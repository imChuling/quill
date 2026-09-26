"""演奏法库 ArticulationLibrary + .tplib 读写 —— FR-7（阶段5 Day4-5）。

按 Part B 契约把已标注事件提炼成**可重定向的控制曲线**（论点：演奏法库不是音频切片）：
    pluck  : RMS 三桶颗粒（granular, Roads 2001）
    slide  : 归一化样条控制点 g(τ)∈[0,1]（音程×时长归一化）
    legato : ±50ms@250Hz 幅度/亮度过渡曲线（25 点）
    strum  : 弦序时差向量 offsets_ms + 方向
    sustain: TimbreSnapshot 引用（内嵌，自包含）

**v2(冲刺本子 Sprint4 / 技术母本 §4.1)——行为=归一化连续轨迹,非标签**:
    每个事件条目新增 traj 块,四条固定 48 点轨迹(τ∈[0,1] 随音长归一化):
      pitch      Δpitch_cents(τ)   相对该音中位 f0(cents)——转移时按目标 interval 重标定
      energy     Δlog_energy(τ)    相对中位 log-RMS——去源绝对响度,应用到目标 velocity
      brightness Δbrightness(τ)    相对中位 log2 谱质心(oct)——residual,不复制绝对质心
      noise      Δnoise_ratio(τ)   相对中位 log10 谱平坦度——气声/摩擦占比的起伏
    每条轨迹带 provenance("extracted"|"inferred"|"user-authored")与 confidence∈[0,1];
    低 confidence 分量由消费端回退为 MIDI gesture/规则模板(§4.2)。
    v1 文件照常加载(traj=None);FORMAT_VERSION=2。

.tplib = zip(manifest.json + arrays.npz)。save/load 往返无损（阶段5 硬出口）。
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2
_ATTACK_LEN = 2646        # 60ms@44.1k
_LEGATO_N = 25            # ±50ms @ 250Hz
_SLIDE_CTRL = 8
_TRAJ_N = 48              # v2:每条行为轨迹的固定采样点数(τ∈[0,1])
_TRAJ_HOP = 512           # 轨迹分析帧移(11.6ms@44.1k)
_TRAJ_KEYS = ("pitch", "energy", "brightness", "noise")


def _seg(y, sr, t0, t1):
    a = max(0, int(t0 * sr)); b = min(len(y), int(t1 * sr))
    return y[a:b] if b > a else y[a:a + 1]


def _rms_bucket(vel, n_buckets=3):
    return int(min(n_buckets - 1, max(0, int(vel * n_buckets))))


def _grain(y, sr, t_on, length=_ATTACK_LEN):
    a = int(t_on * sr)
    g = np.zeros(length, dtype=np.float32)
    seg = y[a:a + length]
    g[:len(seg)] = seg
    fade = max(1, int(0.002 * sr))               # 首尾 2ms 淡变
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade)))
    g[:fade] *= ramp; g[-fade:] *= ramp[::-1]
    return g


def _slide_curve_ctrl(track, n=_SLIDE_CTRL):
    """f0 轨迹 -> 归一化形状 g(τ)∈[0,1] 的 n 个控制点。"""
    if len(track) < 2:
        return np.linspace(0, 1, n).astype(np.float32)
    t = np.array([p[0] for p in track], dtype=np.float64)
    hz = np.array([p[1] for p in track], dtype=np.float64)
    cents = 1200 * np.log2(hz / hz[0])
    span = cents[-1] - cents[0]
    tau = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    g = (cents - cents[0]) / span if abs(span) > 1e-6 else np.zeros_like(cents)
    return np.interp(np.linspace(0, 1, n), tau, g).astype(np.float32)


def _legato_curves(y, sr, t_join, n=_LEGATO_N, half_ms=50.0):
    """join 处 ±half_ms 的幅度与亮度（质心）曲线，各 n 点。"""
    import librosa
    a = t_join - half_ms / 1000.0
    b = t_join + half_ms / 1000.0
    seg = _seg(y, sr, a, b)
    if len(seg) < 64:
        return (np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32))
    # 幅度包络
    rms = librosa.feature.rms(y=seg, frame_length=256, hop_length=64)[0]
    amp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(rms)), rms)
    amp = amp / (np.max(amp) + 1e-9)
    # 亮度（质心，归一到 Nyquist）
    cen = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=64)[0]
    tilt = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(cen)), cen) / (sr / 2)
    return amp.astype(np.float32), tilt.astype(np.float32)


def _resample_traj(vals, n=_TRAJ_N):
    """帧序列 → 固定 n 点(线性;无有效帧返回零)。"""
    v = np.asarray(vals, np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.zeros(n, np.float32), 0.0
    idx = np.arange(len(v), dtype=np.float64)
    v_filled = np.interp(idx, idx[ok], v[ok])           # 缺帧内插
    out = np.interp(np.linspace(0, len(v) - 1, n), idx, v_filled)
    return out.astype(np.float32), float(ok.mean())


def event_trajectories(y, sr, t_on, t_off, f0_track=None):
    """v2 核心:一个事件段 → 四条 Δ轨迹 + provenance/confidence(技术母本 §4.1)。

    全部为**残差**(减各自中位)——转移时不携带源的绝对响度/绝对质心(§4.2)。
    过短段(<3 帧)全部返回 inferred/低置信,由消费端回退规则模板。
    """
    seg = _seg(y, sr, t_on, t_off).astype(np.float64)
    hop = _TRAJ_HOP
    nfr = max(1, (len(seg) - hop) // hop)
    out = {}

    if nfr < 3:
        for k in _TRAJ_KEYS:
            out[k] = {"pts": np.zeros(_TRAJ_N, np.float32),
                      "provenance": "inferred", "confidence": 0.0}
        return out

    frames = np.lib.stride_tricks.sliding_window_view(
        seg[:nfr * hop + hop], hop * 2)[::hop][:nfr]
    win = np.hanning(hop * 2)

    # energy:log-RMS 残差(永远可提取)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loge = np.log(np.maximum(rms, 1e-7))
    e_pts, _ = _resample_traj(loge - np.median(loge))
    out["energy"] = {"pts": e_pts, "provenance": "extracted", "confidence": 1.0}

    # brightness/noise:谱质心(log2 oct)与谱平坦度(log10)残差;静帧记 NaN
    spec = np.abs(np.fft.rfft(frames * win, axis=1))
    freqs = np.fft.rfftfreq(hop * 2, 1 / sr)
    pw = spec.sum(axis=1)
    quiet = pw < (np.median(pw) * 0.05 + 1e-12)
    cent = (spec * freqs).sum(axis=1) / np.maximum(pw, 1e-12)
    cent_l2 = np.where(quiet | (cent <= 20), np.nan, np.log2(np.maximum(cent, 21)))
    b_pts, b_conf = _resample_traj(cent_l2 - np.nanmedian(cent_l2))
    out["brightness"] = {"pts": b_pts, "provenance": "extracted", "confidence": b_conf}
    flat = np.exp(np.log(np.maximum(spec, 1e-12)).mean(axis=1)) / np.maximum(spec.mean(axis=1), 1e-12)
    flat_l = np.where(quiet, np.nan, np.log10(np.maximum(flat, 1e-9)))
    n_pts, n_conf = _resample_traj(flat_l - np.nanmedian(flat_l))
    out["noise"] = {"pts": n_pts, "provenance": "extracted", "confidence": n_conf}

    # pitch:cents 残差。优先用分割器带来的 f0_track [(t,hz)];否则 PESTO 现算
    hz = None
    if f0_track is not None and len(f0_track) >= 3:
        tr = np.asarray(f0_track, np.float64)
        tt = np.linspace(t_on, t_off, nfr)
        hz = np.interp(tt, tr[:, 0], tr[:, 1])
    else:
        # 轨迹提取必须用**连续** f0:PESTO(track_f0)输出按 1/3 半音 bin 量化(33.3c 台阶),
        # 取单音中位(产品取色)无害,但会把揉弦/滑音轨迹整段抹平(7.31 实锤:人声段
        # 420 帧只剩 3 个唯一值→pitch 轨迹恒零)。pyin 连续输出;离线库构建,慢一点无妨。
        try:
            import librosa
            hz_p, vflag, _ = librosa.pyin(seg.astype(np.float32), sr=sr,
                                          fmin=70.0, fmax=1400.0,
                                          frame_length=2048, hop_length=hop)
            hz = np.where(vflag, hz_p, np.nan).astype(np.float64)
        except Exception:
            try:
                from analysis.harmonic import track_f0     # 兜底:量化但胜于无
                hz = np.asarray(track_f0(seg.astype(np.float32), sr=sr)[0], np.float64)
            except Exception:
                hz = None
    if hz is not None and np.isfinite(hz).sum() >= 3 and np.nanmedian(hz) > 0:
        cents = np.where(hz > 0, 1200 * np.log2(np.maximum(hz, 1e-6)), np.nan)
        if len(cents) >= 5:
            cents = cents.copy()
            cents[0] = cents[-1] = np.nan       # 掐首尾帧(PESTO 窗口暖机毛刺,实测首帧 -66c)
            m = cents.copy()                     # 3 点中值滤波:杀孤立倍频错帧
            for j in range(1, len(cents) - 1):
                m[j] = np.nanmedian(cents[j - 1:j + 2])
            cents = m
        p_pts, p_conf = _resample_traj(cents - np.nanmedian(cents))
        out["pitch"] = {"pts": p_pts, "provenance": "extracted", "confidence": p_conf}
    else:
        out["pitch"] = {"pts": np.zeros(_TRAJ_N, np.float32),
                        "provenance": "inferred", "confidence": 0.0}
    return out


def build_library(events, y, sr, sustain_snapshot=None, instrument_guess="guitar",
                  source="", n_buckets=3):
    """已标注事件 + 源音频 -> ArticulationLibrary（dict, Part B 契约）。"""
    lib = {"pluck": [], "slide": [], "legato": [], "strum": [], "sustain": None,
           "meta": {}}
    ev_sorted = sorted(events, key=lambda e: e["t_on"])

    # strum 分组：连续 strum 标签聚成一簇
    i = 0
    while i < len(ev_sorted):
        e = ev_sorted[i]
        lab = e.get("label", "")
        if lab == "pluck":
            lib["pluck"].append({
                "rms_bucket": _rms_bucket(e.get("velocity_est", 0.5), n_buckets),
                "grain": _grain(y, sr, e["t_on"]),
                "traj": event_trajectories(y, sr, e["t_on"], e["t_off"],
                                           e.get("f0_track"))})
            i += 1
        elif lab == "slide":
            track = e.get("f0_track", [])
            interval = 0.0
            if len(track) >= 2:
                interval = 12 * np.log2(track[-1][1] / track[0][1]) if track[0][1] > 0 else 0.0
            lib["slide"].append({
                "interval_st": float(interval),
                "dur_s": float(e["t_off"] - e["t_on"]),
                "curve_ctrl": _slide_curve_ctrl(track),
                "traj": event_trajectories(y, sr, e["t_on"], e["t_off"], track)})
            i += 1
        elif lab == "legato":
            amp_c, tilt_c = _legato_curves(y, sr, e["t_on"])
            lib["legato"].append({"amp_curve": amp_c, "tilt_curve": tilt_c,
                                  "traj": event_trajectories(y, sr, e["t_on"], e["t_off"],
                                                             e.get("f0_track"))})
            i += 1
        elif lab == "strum":
            grp = [e]
            j = i + 1
            while j < len(ev_sorted) and ev_sorted[j].get("label") == "strum" and \
                    ev_sorted[j]["t_on"] - e["t_on"] < 0.12:
                grp.append(ev_sorted[j]); j += 1
            t0 = grp[0]["t_on"]
            offsets = np.array([(g["t_on"] - t0) * 1000.0 for g in grp], dtype=np.float32)
            pitches = [g.get("pitch_midi") or 0 for g in grp]
            direction = "up" if pitches[-1] >= pitches[0] else "down"
            lib["strum"].append({"offsets_ms": offsets, "direction": direction,
                                 "traj": event_trajectories(
                                     y, sr, t0, max(g["t_off"] for g in grp))})
            i = j
        else:
            i += 1

    if sustain_snapshot is not None:
        lib["sustain"] = sustain_snapshot

    lib["meta"] = {
        "instrument_guess": instrument_guess, "source": source,
        "format_version": FORMAT_VERSION,
        "stats": {k: len(lib[k]) for k in ("pluck", "slide", "legato", "strum")},
    }
    return lib


# --------------------------------------------------------------------------- #
# .tplib 序列化（zip = manifest.json + arrays.npz）往返无损
# --------------------------------------------------------------------------- #
def save_tplib(path, lib):
    path = Path(path)
    if path.suffix != ".tplib":
        path = path.with_suffix(".tplib")
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    manifest = {"meta": lib.get("meta", {}), "pluck": [], "slide": [],
                "legato": [], "strum": [], "sustain": None}

    def _pack_traj(entry, prefix, mrec):
        """v2:四条轨迹入 arrays,provenance/confidence 入 manifest。"""
        traj = entry.get("traj")
        if not traj:
            return
        mrec["traj"] = {}
        for tk in _TRAJ_KEYS:
            tr = traj.get(tk)
            if tr is None:
                continue
            ak = f"{prefix}_traj_{tk}"
            arrays[ak] = np.asarray(tr["pts"], np.float32)
            mrec["traj"][tk] = {"pts": ak, "provenance": tr["provenance"],
                                "confidence": round(float(tr["confidence"]), 4)}

    for idx, p in enumerate(lib.get("pluck", [])):
        k = f"pluck_{idx}_grain"; arrays[k] = np.asarray(p["grain"], np.float32)
        rec = {"rms_bucket": int(p["rms_bucket"]), "grain": k}
        _pack_traj(p, f"pluck_{idx}", rec)
        manifest["pluck"].append(rec)
    for idx, s in enumerate(lib.get("slide", [])):
        k = f"slide_{idx}_curve"; arrays[k] = np.asarray(s["curve_ctrl"], np.float32)
        rec = {"interval_st": float(s["interval_st"]),
               "dur_s": float(s["dur_s"]), "curve_ctrl": k}
        _pack_traj(s, f"slide_{idx}", rec)
        manifest["slide"].append(rec)
    for idx, lg in enumerate(lib.get("legato", [])):
        ka, kt = f"legato_{idx}_amp", f"legato_{idx}_tilt"
        arrays[ka] = np.asarray(lg["amp_curve"], np.float32)
        arrays[kt] = np.asarray(lg["tilt_curve"], np.float32)
        rec = {"amp_curve": ka, "tilt_curve": kt}
        _pack_traj(lg, f"legato_{idx}", rec)
        manifest["legato"].append(rec)
    for idx, st in enumerate(lib.get("strum", [])):
        k = f"strum_{idx}_offsets"; arrays[k] = np.asarray(st["offsets_ms"], np.float32)
        rec = {"offsets_ms": k, "direction": st["direction"]}
        _pack_traj(st, f"strum_{idx}", rec)
        manifest["strum"].append(rec)

    sus = lib.get("sustain")
    if sus is not None:
        manifest["sustain"] = {"f0_ref": float(sus["f0_ref"]),
                               "meta": sus.get("meta", {}),
                               "has_partials": sus.get("partial_freqs") is not None,
                               "vibrato": sus.get("vibrato")}
        arrays["sustain_H_env"] = np.asarray(sus["H_env"], np.float32)
        arrays["sustain_N_env"] = np.asarray(sus["N_env"], np.float32)
        arrays["sustain_attack_res"] = np.asarray(sus["attack_res"], np.float32)
        if sus.get("partial_freqs") is not None:
            arrays["sustain_partial_freqs"] = np.asarray(sus["partial_freqs"], np.float32)
        if sus.get("noise_centers_hz") is not None:
            arrays["sustain_noise_centers_hz"] = np.asarray(sus["noise_centers_hz"], np.float32)

    npz_buf = io.BytesIO()
    np.savez(npz_buf, **arrays)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("arrays.npz", npz_buf.getvalue())
    return path


def load_tplib(path):
    with zipfile.ZipFile(path, "r") as z:
        manifest = json.loads(z.read("manifest.json"))
        with io.BytesIO(z.read("arrays.npz")) as buf:
            data = np.load(buf)
            arrays = {k: data[k] for k in data.files}

    def _unpack_traj(mrec):
        """v2 → traj dict;v1 文件无 traj 字段 → None(消费端回退)。"""
        tm = mrec.get("traj")
        if not tm:
            return None
        return {tk: {"pts": arrays[tr["pts"]], "provenance": tr["provenance"],
                     "confidence": tr["confidence"]}
                for tk, tr in tm.items()}

    lib = {"pluck": [], "slide": [], "legato": [], "strum": [],
           "sustain": None, "meta": manifest.get("meta", {})}
    for p in manifest["pluck"]:
        lib["pluck"].append({"rms_bucket": p["rms_bucket"],
                             "grain": arrays[p["grain"]],
                             "traj": _unpack_traj(p)})
    for s in manifest["slide"]:
        lib["slide"].append({"interval_st": s["interval_st"], "dur_s": s["dur_s"],
                             "curve_ctrl": arrays[s["curve_ctrl"]],
                             "traj": _unpack_traj(s)})
    for lg in manifest["legato"]:
        lib["legato"].append({"amp_curve": arrays[lg["amp_curve"]],
                              "tilt_curve": arrays[lg["tilt_curve"]],
                              "traj": _unpack_traj(lg)})
    for st in manifest["strum"]:
        lib["strum"].append({"offsets_ms": arrays[st["offsets_ms"]],
                             "direction": st["direction"],
                             "traj": _unpack_traj(st)})
    sm = manifest.get("sustain")
    if sm is not None:
        sus = {"f0_ref": sm["f0_ref"], "meta": sm.get("meta", {}),
               "vibrato": sm.get("vibrato"),
               "H_env": arrays["sustain_H_env"], "N_env": arrays["sustain_N_env"],
               "attack_res": arrays["sustain_attack_res"],
               "partial_freqs": arrays.get("sustain_partial_freqs"),
               "noise_centers_hz": arrays.get("sustain_noise_centers_hz")}
        lib["sustain"] = sus
    return lib
