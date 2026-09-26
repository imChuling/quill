"""云端渲染服务 —— take_schema_v1 契约的可托管实现(FastAPI)。

定位:companion 网页"任何平台都能跑"的渲染后端。桌面版 companion_api.py
(127.0.0.1 铁律)原封不动;本模块是并行的部署目标,同一契约:

  GET  /status                 版本/可用音色
  POST /render                 notes+bpm+title(+timbre) → 同步渲染 → take_id
  GET  /takes/{id}             take 元数据
  GET  /takes/{id}/audio.wav   WAV 字节

与桌面版的差异(有意为之):
- 渲染走神经路径(ConditionedNN 检索 + DDSP 音色),不依赖实时引擎;
- POST /render 同步完成(短 region 秒级),返回 state=ready——避免
  无状态容器上的后台线程/轮询竞态;companion 轮询循环天然兼容;
- CORS 由 QUILL_ALLOWED_ORIGINS 环境变量控制(缺省 *,demo 用)。

本地试跑: uvicorn ui.render_service:app --port 8724
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

QUILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUILL))

from fastapi import FastAPI, HTTPException, Request   # noqa: E402
from fastapi.middleware.cors import CORSMiddleware    # noqa: E402
from fastapi.responses import Response                # noqa: E402

from ui.companion_api import TakeStore                # noqa: E402  复用落盘仓库

CKPT = Path(os.environ.get("QUILL_CKPT_DIR", QUILL / "checkpoints"))
TAKES_DIR = Path(os.environ.get("QUILL_TAKES_DIR", "/tmp/quill_takes"))
ASSETS = Path(os.environ.get("QUILL_ASSETS_DIR", "/tmp/quill_assets"))
TONES_DIR, BEHS_DIR = ASSETS / "tones", ASSETS / "behaviors"
VERSION = "render-service/2.1"

# ---- 渲染/输入契约常量(原散落的魔法数)---- #
SR_OUT = 16000                 # DDSP 输出采样率
GAP_S = 0.08                   # 乐句切分阈值:空隙 ≥ 此值即断句
EDGE_FADE_S = 0.005            # 乐句边缘淡入淡出,防拼接咔哒
PEAK_TARGET = 0.9              # 归一化目标峰值
MAX_RENDER_S = 120.0           # 单次渲染时间轴上限
MAX_UPLOAD_BYTES = 30_000_000  # /capture 单次上传上限
CAPTURE_MIN_S, CAPTURE_MAX_S = 1.0, 90.0
BEH_CACHE_MAX = 8              # 行为库 LRU 上限(容器内存有限)
MAX_STORED_TONES = int(os.environ.get("QUILL_MAX_ASSETS", "60"))  # 资产总量上限


def _persist(warnings: list | None = None):
    """Modal Volume 显式落盘(本地跑时为 no-op)。

    失败必须可见:静默 pass 会让 /capture 返回 ok 而资产在下一个容器蒸发。
    """
    if not os.environ.get("QUILL_VOLUME_NAME"):
        return
    try:
        import modal
        modal.Volume.from_name(os.environ["QUILL_VOLUME_NAME"]).commit()
    except Exception as e:                                      # noqa: BLE001
        msg = f"volume commit failed: {str(e)[:80]}"
        print(f"[quill] {msg}", file=sys.stderr)
        if warnings is not None:
            warnings.append(msg)


def _evict_assets(warnings: list | None = None):
    """资产目录按 mtime 淘汰最旧的,防持久卷无限增长(/capture 无鉴权)。"""
    for d in (TONES_DIR, BEHS_DIR):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        for p in files[:-MAX_STORED_TONES] if len(files) > MAX_STORED_TONES else []:
            try:
                p.unlink()
                if warnings is not None:
                    warnings.append(f"evicted oldest asset {p.stem}")
            except OSError:
                pass

# 音色注册表:名字 → (DDSP 权重, 轨迹库, envelope)。glass 用 style 库当轨迹库。
TIMBRES = {
    "violin":  ("ddsp_urmp_vn_12k.pt",  "urmp_vn_trajectories.npz",  "sustain"),
    "trumpet": ("ddsp_urmp_tpt_12k.pt", "urmp_tpt_trajectories.npz", "sustain"),
    "guitar":  ("ddsp_guitarset.pt",    "guitarset_trajectories.npz", "pluck"),
    "glass":   ("ddsp_glass.pt",        "glass_style.npz",           "sustain"),
}
ART_MAP = {"pluck": 0, "plain": 0, "legato": 0, "vibrato": 1, "slide": 2}
# 只有这些旋钮真的作用于渲染;未列出的一律拒收,避免"装饰性控件"。
#   noise→DDSP 噪声端  master→输出增益  reverb→算法混响
#   brightness→谱倾斜  artint→traj_gain(轨迹作用强度,研究参数)
SYNTH_KEYS = {"noise", "master", "reverb", "brightness", "artint"}

_models = {}
_lock = threading.Lock()


def _load(timbre: str):
    """懒加载并缓存 (DDSP, ConditionedNN, envelope)。"""
    with _lock:
        if timbre in _models:
            return _models[timbre]
        import torch
        from neural.ddsp import DDSP
        from neural.neural_take import ConditionedNN
        ck, tr, env = TIMBRES[timbre]
        m = DDSP(dropout=0.0)
        m.load_state_dict(torch.load(str(CKPT / ck), map_location="cpu",
                                     weights_only=True))
        m.eval()
        d = np.load(str(CKPT / tr))
        nn = ConditionedNN(d["features"], d["trajectories"])
        _models[timbre] = (m, nn, env)
        return _models[timbre]


def _render_tone_l0(notes: list, tone_id: str):
    """captured tone → L0 引擎离线渲染(取色即演奏,与桌面版同链路)。

    Engine 事件带绝对时间,时间轴天然保真;slide/legato 沿用桌面语义。
    """
    import io as _io
    import soundfile as sf
    from snapshot import load_snapshot
    from synth.instrument import build_instrument
    from runtime.engine import Engine
    snap = load_snapshot(str(TONES_DIR / f"{tone_id}.npz"))
    tex = bool((snap.get("meta", {}) or {}).get("texture"))
    inst = build_instrument(snap, noise_level=0.6 if tex else 0.02)
    eng = Engine(inst, eval_mode=True)
    seq = sorted(notes, key=lambda x: float(x["start"]))
    evs, prev = [], -1
    for nt in seq:
        m, v = int(nt["midi"]), int(nt.get("vel", 100))
        art = str(nt.get("art", "pluck"))
        t0 = float(nt["start"]); t1 = t0 + float(nt["len"])
        if art == "slide" and prev >= 0:
            evs.append((t0, "slide", m, v, 0.08, prev))
        elif art == "legato" and prev >= 0:
            evs.append((t0, "legato", m, v, 0.08, prev))
        else:
            evs.append((t0, "on", m, v, 0.0, -1))
        evs.append((t1, "off", m, 0, 0.0, -1))
        prev = m
    dur = max(t0 + float(nt["len"]) for nt in seq for t0 in [float(nt["start"])]) + 0.5
    y = np.asarray(eng.render_offline(evs, dur), np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = (y / peak * 0.9).astype(np.float32)
    out_peak = float(np.max(np.abs(y))) if y.size else 0.0
    buf = _io.BytesIO()
    sf.write(buf, y, eng.sr, format="WAV")
    return (buf.getvalue(),
            (20.0 * np.log10(out_peak) if out_peak > 0 else -120.0),
            round(len(y) / eng.sr, 3))


def _apply_reverb(y: np.ndarray, sr: int, wet: float) -> np.ndarray:
    """Simple algorithmic reverb via exponentially decaying comb filters."""
    from scipy.signal import fftconvolve
    rt60 = 1.2 + wet * 1.8
    ir_len = int(sr * rt60)
    ir = np.zeros(ir_len, np.float64)
    ir[0] = 1.0
    for delay_ms in [29, 37, 43, 53, 67, 79]:
        d = int(delay_ms * sr / 1000)
        n_echoes = ir_len // d
        for k in range(1, n_echoes):
            ir[k * d] += 0.6 ** k * (0.7 + 0.3 * np.sin(k * 0.7))
    ir /= np.max(np.abs(ir)) + 1e-12
    reverb = fftconvolve(y, ir, mode="full")[:len(y)]
    return y * (1.0 - wet) + reverb * wet


def _apply_tilt(y: np.ndarray, sr: int, brightness: float) -> np.ndarray:
    """一阶谱倾斜(brightness 0.5=直通,>0.5 提高频 <0.5 提低频)。

    单极点高通/低通的加权混合——比参数均衡便宜,足够作 tone 控制。
    """
    from scipy.signal import butter, sosfilt
    tilt = (float(brightness) - 0.5) * 2.0                      # -1..+1
    if abs(tilt) < 0.02:
        return y
    sos = butter(1, 1200.0 / (sr / 2.0), btype="high", output="sos")
    hi = sosfilt(sos, y)
    return y + tilt * 0.85 * (hi if tilt > 0 else (y - hi))


def _render_audio(notes: list, timbre: str, behavior: str | None = None,
                  synth_params: dict | None = None) -> tuple[bytes, float, float]:
    """companion notes({midi,start,len,art}) → 单声部化 → 引擎 score → WAV。

    返回 (wav_bytes, peak_dbfs, duration_s)。

    DAW 来的 region 常含和弦与 let-ring 重叠;引擎是单声部的,复用
    midi_dialect 的规则(同 onset 保最高音、重叠截断)统一在服务端处理。
    behavior 取值:
      "beh:<id>"   captured 行为库(解耦主张的网页实证)
      "distilled"  Plan D 蒸馏残差学生(MIDI-DDSP 教师 → Quill 行为残差)
    synth_params 见 SYNTH_KEYS:noise/master/reverb/brightness 走后处理与
    DDSP 噪声端,artint 走 traj_gain(轨迹作用强度,研究参数)。
    """
    import soundfile as sf
    import torch
    from analysis.midi_dialect import ImportReport, _monophonize
    from neural.perform import control_from_behavior
    if timbre.startswith("tone:"):
        return _render_tone_l0(notes, timbre[5:])
    model, nn, env = _load(timbre)
    if behavior == "distilled":
        nn = _load_distilled()
    elif behavior and behavior.startswith("beh:"):
        nn = _load_behavior_nn(behavior[4:])
    events = [(float(n["start"]), float(n["len"]), int(n["midi"]),
               ART_MAP.get(str(n.get("art", "pluck")), 0), None) for n in notes]
    mono = _monophonize(events, ImportReport())
    score = [{"midi": int(m), "dur": max(0.05, float(d)), "onset": float(o),
              "art": int(a)} for o, d, m, a, _x in mono]

    # 引擎按乐句拼接段落,onset 只进检索特征不进输出时序 → 时间轴保真由
    # 本层负责:空隙 ≥GAP_S 切乐句,乐句内整体渲染(保 crossfade/legato
    # 连贯),乐句音频按 onset 精确摆进时间轴缓冲,空隙即静音。
    phrases, cur = [], [score[0]]
    for a, b in zip(score, score[1:]):
        if b["onset"] - (a["onset"] + a["dur"]) >= GAP_S:
            phrases.append(cur)
            cur = []
        cur.append(b)
    phrases.append(cur)

    sp = synth_params or {}
    noise_g = float(sp.get("noise", 0.6))
    master_g = float(sp.get("master", 0.62)) * 1.4
    # INTENSITY 旋钮 → traj_gain:检索轨迹的作用强度(0.3 收敛 … 2.0 夸张)
    traj_gain = 0.3 + float(sp.get("artint", 0.41)) * 1.7

    rendered = []
    for ph in phrases:
        f0, loud = control_from_behavior(ph, nn, pitch_glide=False, envelope=env,
                                         traj_gain=traj_gain)
        with torch.no_grad():
            y = model(f0, loud, noise_gain=noise_g, noise_gate_db=-1.0).numpy()[0]
        rendered.append((ph[0]["onset"], y))

    end = max(o + len(y) / SR_OUT for o, y in rendered) + 0.05
    buf_y = np.zeros(int(np.ceil(end * SR_OUT)), np.float64)
    fade = max(1, int(EDGE_FADE_S * SR_OUT))
    for onset, y in rendered:
        y = y.astype(np.float64).copy()
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
        i0 = int(round(onset * SR_OUT))
        buf_y[i0:i0 + len(y)] += y                     # overlap-add(乐句彼此不重叠)
    buf_y *= master_g
    buf_y = _apply_tilt(buf_y, SR_OUT, float(sp.get("brightness", 0.5)))
    reverb_wet = float(sp.get("reverb", 0.0))
    if reverb_wet > 0.01:
        buf_y = _apply_reverb(buf_y, SR_OUT, reverb_wet)
    peak = float(np.max(np.abs(buf_y))) if buf_y.size else 0.0
    if peak > 0:
        buf_y = buf_y / peak * PEAK_TARGET
    y16 = buf_y.astype(np.float32)
    out_peak = float(np.max(np.abs(y16))) if y16.size else 0.0
    buf = io.BytesIO()
    sf.write(buf, y16, SR_OUT, format="WAV")
    return (buf.getvalue(),
            (20.0 * np.log10(out_peak) if out_peak > 0 else -120.0),
            round(len(y16) / SR_OUT, 3))


_beh_cache: "OrderedDict[str, object]" = __import__("collections").OrderedDict()

DISTILLED_CKPT = CKPT / "behavior_distilled.pt"
RESIDUAL_STRENGTH = 1.5        # 耳测拍板值(见 docs/changelog Plan D)
_distilled = None


def _load_distilled():
    """Plan D 蒸馏行为学生(MIDI-DDSP 教师 → Quill 行为残差)。

    接口与 ConditionedNN 同构(predict/use_art),故可直接顶替检索模型。
    """
    global _distilled
    if _distilled is None:
        from neural.behavior_distill import DistilledBehavior
        _distilled = DistilledBehavior(DISTILLED_CKPT, device="cpu",
                                       residual_strength=RESIDUAL_STRENGTH)
    return _distilled


def _load_behavior_nn(beh_id: str):
    """captured behavior npz → ConditionedNN(LRU 缓存,容器内存有限)。"""
    if beh_id in _beh_cache:
        _beh_cache.move_to_end(beh_id)
        return _beh_cache[beh_id]
    from neural.neural_take import ConditionedNN
    d = np.load(str(BEHS_DIR / f"{beh_id}.npz"))
    nn = ConditionedNN(d["features"], d["trajectories"])
    _beh_cache[beh_id] = nn
    while len(_beh_cache) > BEH_CACHE_MAX:
        _beh_cache.popitem(last=False)
    return nn


def _slug(name: str, blob: bytes) -> str:
    import hashlib
    import re
    base = re.sub(r"[^\w\-]+", "-", (name or "capture").strip().lower()).strip("-")[:24] or "capture"
    return f"{base}-{hashlib.sha1(blob).hexdigest()[:8]}"


def _decode_audio(blob: bytes):
    """任意常见格式 → mono float32。librosa/audioread(镜像带 ffmpeg)。"""
    import io as _io
    import librosa
    y, sr = librosa.load(_io.BytesIO(blob), sr=None, mono=True)
    return y.astype(np.float32), int(sr)


def _extract_behavior(blob: bytes, out_npz: Path):
    """extract_style 管线(路径进出);返回 (n_notes, art_counts) 或 None。"""
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location(
        "quill_extract_style", str(QUILL / "tools" / "extract_style.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        import soundfile as _sf
        y, sr = _decode_audio(blob)
        _sf.write(f.name, y, sr)
        tmp = f.name
    try:
        mod.extract(tmp, str(out_npz), verbose=False)
    finally:
        Path(tmp).unlink(missing_ok=True)
    if not out_npz.exists():
        return None
    d = np.load(str(out_npz))
    n = int(d["features"].shape[0])
    if n < 2:
        out_npz.unlink(missing_ok=True)
        return None
    arts = d["features"][:, 6].astype(int)
    return n, {"plain": int((arts == 0).sum()), "vibrato": int((arts == 1).sum()),
               "slide": int((arts == 2).sum())}


# -------------------------------------------------------------------- #
app = FastAPI(title="Quill Render Service")
_origins = [o for o in os.environ.get("QUILL_ALLOWED_ORIGINS", "*").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=_origins or ["*"],
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
_takes = TakeStore(TAKES_DIR)


@app.middleware("http")
async def _no_cache_html(request: "Request", call_next):
    """入口 HTML 禁缓存:每次部署换 bundle 哈希,旧 HTML 会引用不存在的
    bundle 或旧代码。哈希资产(/assets/*)不动,浏览器随便缓存。"""
    resp = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/status")
def status():
    return {"ok": True, "version": VERSION, "engine": "neural-ddsp",
            "timbres": [t for t in TIMBRES if (CKPT / TIMBRES[t][0]).exists()]}


@app.post("/capture")
async def capture(request: "Request", name: str = "capture"):
    """一段录音 → 双资产:音色快照(tone)+ 演奏法风格库(behavior)。"""
    blob = await request.body()
    if not blob or len(blob) < 1000:
        raise HTTPException(400, "empty audio")
    if len(blob) > 30_000_000:
        raise HTTPException(400, "audio too large (30MB max)")
    try:
        y, sr = _decode_audio(blob)
    except Exception as e:                                      # noqa: BLE001
        raise HTTPException(400, f"cannot decode audio: {str(e)[:50]}")
    dur = len(y) / sr
    if not (1.0 <= dur <= 90.0):
        raise HTTPException(400, f"audio must be 1-90s (got {dur:.1f}s)")
    TONES_DIR.mkdir(parents=True, exist_ok=True)
    BEHS_DIR.mkdir(parents=True, exist_ok=True)
    aid = _slug(name, blob)
    out = {"ok": True, "id": aid, "duration_s": round(dur, 2),
           "tone": None, "behavior": None, "warnings": []}
    try:
        from analysis.timbre import analyze_timbre
        from snapshot import save_snapshot
        snap = analyze_timbre(y, sr, name=name)
        snap.setdefault("meta", {})["disp"] = name          # 资产列表用显示名
        save_snapshot(str(TONES_DIR / aid), snap)
        f0 = float(snap.get("f0_ref", 0.0))
        midi = 69 + 12 * np.log2(f0 / 440.0) if f0 > 0 else 0
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note = f"{names[int(round(midi)) % 12]}{int(round(midi)) // 12 - 1}" if f0 > 0 else "—"
        h = snap.get("H_env")
        out["tone"] = {"id": aid, "name": name, "f0_hz": round(f0, 1), "note": note,
                       "partials": (int(np.asarray(h).shape[0]) if h is not None else 0)}
    except Exception as e:                                      # noqa: BLE001
        out["warnings"].append(f"tone: {str(e)[:60]}")
    try:
        r = _extract_behavior(blob, BEHS_DIR / f"{aid}.npz")
        if r:
            n, arts = r
            out["behavior"] = {"id": aid, "name": name, "n_notes": n, "arts": arts}
        else:
            out["warnings"].append("behavior: fewer than 2 notes detected")
    except Exception as e:                                      # noqa: BLE001
        out["warnings"].append(f"behavior: {str(e)[:60]}")
    if not out["tone"] and not out["behavior"]:
        raise HTTPException(422, "; ".join(out["warnings"]) or "capture failed")
    _evict_assets(out["warnings"])
    _persist(out["warnings"])
    return out


@app.get("/assets")
def assets():
    """已捕获资产列表(tones + behaviors)。"""
    tones, behs = [], []
    if TONES_DIR.is_dir():
        for p in sorted(TONES_DIR.glob("*.npz"), key=lambda x: x.stat().st_mtime):
            try:
                from snapshot import load_snapshot
                disp = (load_snapshot(str(p)).get("meta", {}) or {}).get("disp") or p.stem
            except Exception:
                disp = p.stem
            tones.append({"id": p.stem, "name": str(disp)})
    if BEHS_DIR.is_dir():
        for p in sorted(BEHS_DIR.glob("*.npz"), key=lambda x: x.stat().st_mtime):
            try:
                d = np.load(str(p))
                behs.append({"id": p.stem, "name": p.stem.rsplit("-", 1)[0],
                             "n_notes": int(d["features"].shape[0])})
            except Exception:
                continue
    # Plan D 蒸馏行为学生:与 captured 行为库并列,可直接换上去对比
    if DISTILLED_CKPT.exists():
        behs.insert(0, {"id": "distilled", "name": "MIDI-DDSP distilled",
                        "n_notes": 0, "kind": "model"})
    return {"tones": tones, "behaviors": behs}


@app.post("/render")
def render(body: dict):
    """契约同 companion_api.do_POST:校验一致;新增可选 timbre 字段。"""
    notes = body.get("notes") or []
    bpm = float(body.get("bpm", 120.0))
    title = str(body.get("title", "") or "Inked region")[:80]
    timbre = str(body.get("timbre", "violin"))
    behavior = str(body.get("behavior", "") or "")
    _safe_id = lambda s: all(c.isalnum() or c in "-_" for c in s) and len(s) <= 80
    if timbre.startswith("tone:"):
        tid = timbre[5:]
        if not _safe_id(tid):
            raise HTTPException(400, "invalid tone id")
        if not (TONES_DIR / f"{tid}.npz").exists():
            raise HTTPException(400, f"unknown captured tone {tid!r}")
    elif timbre not in TIMBRES or not (CKPT / TIMBRES[timbre][0]).exists():
        raise HTTPException(400, f"unknown timbre {timbre!r}")
    if behavior in ("distilled", "beh:distilled"):
        behavior = "distilled"
        if not DISTILLED_CKPT.exists():
            raise HTTPException(400, "distilled behavior model not deployed")
    elif behavior.startswith("beh:"):
        bid = behavior[4:]
        if not _safe_id(bid):
            raise HTTPException(400, "invalid behavior id")
        if not (BEHS_DIR / f"{bid}.npz").exists():
            raise HTTPException(400, f"unknown behavior {bid!r}")
    elif behavior:
        raise HTTPException(400, f"unknown behavior {behavior!r}")
    if not notes:
        raise HTTPException(400, "empty notes")
    end = 0.0
    try:
        for nt in notes:
            m, s, ln = int(nt["midi"]), float(nt["start"]), float(nt["len"])
            if not (0 <= m <= 127) or ln <= 0 or s < 0:
                raise HTTPException(400, f"bad note midi={m}")
            end = max(end, s + ln)
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as e:   # 缺字段/类型错是 400 不是 500
        raise HTTPException(400, f"malformed notes: {str(e)[:50]}")
    if end > MAX_RENDER_S:
        raise HTTPException(400, f">{MAX_RENDER_S:.0f}s")

    synth_params = {k: max(0.0, min(1.0, float(v)))
                    for k, v in (body.get("synth_params") or {}).items()
                    if k in SYNTH_KEYS}
    clean = [{"midi": int(n["midi"]), "start": float(n["start"]),
              "len": float(n["len"]), "vel": int(n.get("vel", 100)),
              "art": str(n.get("art", "pluck"))} for n in notes]
    take = _takes.stage(
        title=title, notes=clean, bpm=bpm,
        timbre={"name": timbre + (f" × {behavior}" if behavior else ""),
                "f0_ref_hz": 0.0, "noise_level": 0.0},
        duration_s=end, sr=SR_OUT,
        provenance={"quill_version": VERSION, "engine": "neural-ddsp",
                    "atlas": None, "source_region": None},
        # 旋钮影响音频却不在其余字段里,必须进 take 身份,否则改完旋钮
        # 重渲会误命中幂等缓存、返回旧音频(旋钮看起来"没作用")。
        variant=json.dumps(synth_params, sort_keys=True, separators=(",", ":")))
    if take["audio"]["state"] == "ready":                       # 幂等命中
        return {"ok": True, "take_id": take["take_id"], "state": "ready"}
    try:
        wav, peak_db, audio_s = _render_audio(clean, timbre, behavior or None, synth_params)
    except HTTPException:
        raise
    except Exception as e:                                      # noqa: BLE001
        _takes.fail(take["take_id"], str(e))
        raise HTTPException(500, f"render_failed: {str(e)[:60]}")
    _takes.attach_audio(take["take_id"], wav, peak_db)
    # duration_s 改为真实时间轴长度(乐句摆位后),companion 以它折算
    # musicDurationTicks —— 与源 region 的节奏严格同长,不被拉伸。
    _takes.set_duration(take["take_id"], audio_s)
    return {"ok": True, "take_id": take["take_id"], "state": "ready"}


@app.get("/takes")
def takes_list():
    return {"takes": _takes.list()}


def _valid_tid(tid: str) -> bool:
    """take_id 是内容哈希 tk_+12hex;路径拼接前先验格式(纵深防御)。"""
    import re
    return bool(re.fullmatch(r"tk_[0-9a-f]{12}", tid))


@app.get("/takes/{tid}")
def take_meta(tid: str):
    take = _takes.get(tid) if _valid_tid(tid) else None
    if take is None:
        raise HTTPException(404, "not_found")
    return take


@app.get("/takes/{tid}/audio.wav")
def take_audio(tid: str):
    take = _takes.get(tid) if _valid_tid(tid) else None
    if take is None:
        raise HTTPException(404, "not_found")
    if take["audio"]["state"] != "ready":
        raise HTTPException(409, take["audio"]["state"])
    return Response(_takes.wav_path(tid).read_bytes(), media_type="audio/wav")


# ---- 同源托管 companion 静态页(QUILL_WEB_DIR 指向 vite dist)----
# mount 在路由之后注册,/status 等 API 优先匹配;页面与 API 同源 → 零 CORS。
_web = os.environ.get("QUILL_WEB_DIR", "")
if _web and Path(_web).is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
