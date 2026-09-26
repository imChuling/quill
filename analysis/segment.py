"""事件切分：Basic Pitch + 自建 onset/f0 双前端求并集 —— FR-5（阶段4 Day1-2）。

两个互补前端：
  * M3 Basic Pitch（Spotify, ICASSP 2022）：唯一带 pitch-bend 的轻量复音转录
  * 自建 `librosa.onset.onset_detect(backtrack=True)` + PESTO 帧级 f0 分段

**双前端求并集**：两者都命中（同音高、同起点）= 高置信；只一方命中 = 进 disagreements
供人工裁决（方法论写报告）。

坑位：吉他泛音让 Basic Pitch 出幽灵高八度音 -> 置信度阈值 + 与 PESTO f0 一致性过滤。
"""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np

try:
    from quill_config import CFG
    _SR = int(CFG["audio"]["sr"])
except Exception:  # pragma: no cover
    _SR = 44100


# --------------------------------------------------------------------------- #
# 音高/频率换算
# --------------------------------------------------------------------------- #
def hz_to_midi(f):
    f = np.asarray(f, dtype=np.float64)
    out = np.full_like(f, np.nan)
    pos = f > 0
    out[pos] = 69.0 + 12.0 * np.log2(f[pos] / 440.0)
    return out



def _f0_median_in(f0, f0_times, t0, t1, skip_ms=10.0):
    """事件窗 [t0+skip, t1] 内 PESTO f0 的中位数（Hz），无有效值返回 NaN。"""
    f0 = np.asarray(f0); f0_times = np.asarray(f0_times)
    a = t0 + skip_ms / 1000.0
    m = (f0_times >= a) & (f0_times <= max(t1, a + 1e-3)) & np.isfinite(f0) & (f0 > 0)
    return float(np.median(f0[m])) if np.any(m) else float("nan")


# --------------------------------------------------------------------------- #
# 前端 1：Basic Pitch
# --------------------------------------------------------------------------- #
def basic_pitch_events(y, sr=_SR, onset_thresh=0.5, frame_thresh=0.3,
                       min_note_len_ms=60.0):
    """M3 Basic Pitch -> [{t_on,t_off,pitch_midi,amp,bend}]（bend 为半音偏移轨迹）。"""
    import soundfile as sf
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            tmp = fh.name
        sf.write(tmp, np.asarray(y, dtype=np.float32), sr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, note_events = predict(
                tmp, ICASSP_2022_MODEL_PATH,
                onset_threshold=onset_thresh, frame_threshold=frame_thresh,
                minimum_note_length=min_note_len_ms)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

    out = []
    for ne in note_events:
        t_on, t_off, pitch, amp = float(ne[0]), float(ne[1]), int(ne[2]), float(ne[3])
        bend = ne[4] if len(ne) > 4 else None
        out.append({"t_on": t_on, "t_off": t_off, "pitch_midi": float(pitch),
                    "amp": amp,
                    "bend": (None if bend is None else np.asarray(bend, dtype=np.float32)),
                    "source": "bp"})
    out.sort(key=lambda e: e["t_on"])
    return out


# --------------------------------------------------------------------------- #
# 前端 2：自建 onset + f0 分段
# --------------------------------------------------------------------------- #
def onset_f0_events(y, sr=_SR, f0=None, f0_times=None, hop=512):
    """librosa onset(backtrack) + PESTO f0 分段 -> [{t_on,t_off,pitch_midi,...}]。"""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop,
                                        backtrack=True, units="time")
    onsets = np.asarray(onsets, dtype=np.float64)
    dur = len(y) / sr
    if onsets.size == 0:
        return []

    # RMS 包络用于定 offset（能量回落）
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    floor = 0.15 * float(np.max(rms)) if rms.size else 0.0

    events = []
    for i, t_on in enumerate(onsets):
        next_on = onsets[i + 1] if i + 1 < len(onsets) else dur
        # offset = 能量跌破 floor 或下一个 onset，取先到者
        seg = (rms_t >= t_on) & (rms_t < next_on)
        t_off = next_on
        if np.any(seg):
            below = np.where(seg & (rms < floor))[0]
            if below.size:
                t_off = float(rms_t[below[0]])
        t_off = max(t_off, t_on + 0.03)
        pitch_hz = _f0_median_in(f0, f0_times, t_on, min(t_on + 0.2, t_off)) \
            if f0 is not None else float("nan")
        pitch_midi = float(hz_to_midi(np.array([pitch_hz]))[0]) if pitch_hz > 0 else float("nan")
        events.append({"t_on": float(t_on), "t_off": float(t_off),
                       "pitch_midi": pitch_midi, "amp": float("nan"),
                       "bend": None, "source": "onset"})
    return events


# --------------------------------------------------------------------------- #
# 幽灵高八度过滤（坑位）
# --------------------------------------------------------------------------- #
def ghost_octave_filter(events, f0, f0_times, amp_thresh=0.0,
                        tol_semitone=1.5):
    """剔除幽灵八度音 + 弱事件。与 PESTO f0 比对：

      事件音高 p 与窗内 PESTO 期望音高 e 比较，若 |p-e|>tol 但 p±12 命中 e
      （高八度泛音幽灵 或 低八度次谐波幽灵）=> 丢弃；
      另剔除振幅 < amp_thresh 的弱事件（Basic Pitch 置信度阈值）。
    """
    kept, dropped = [], []
    for ev in events:
        if ev.get("amp") is not None and np.isfinite(ev["amp"]) and ev["amp"] < amp_thresh:
            dropped.append({**ev, "drop_reason": "low_amp"})
            continue
        e_hz = _f0_median_in(f0, f0_times, ev["t_on"], ev["t_off"])
        if e_hz > 0 and np.isfinite(ev["pitch_midi"]):
            e_midi = float(hz_to_midi(np.array([e_hz]))[0])
            p = ev["pitch_midi"]
            far = abs(p - e_midi) > tol_semitone
            octave_ghost = (abs((p - 12) - e_midi) <= tol_semitone or   # 高八度泛音
                            abs((p + 12) - e_midi) <= tol_semitone)     # 低八度次谐波
            if far and octave_ghost:
                dropped.append({**ev, "drop_reason": "ghost_octave"})
                continue
        kept.append(ev)
    return kept, dropped


# --------------------------------------------------------------------------- #
# 双前端求并集
# --------------------------------------------------------------------------- #
def union_events(bp_events, onset_events, onset_tol_s=0.05, pitch_tol=1.0):
    """并集：两前端都命中（同音高±pitch_tol、起点±onset_tol）标 agree；
    只一方命中进 disagreements。返回 (events, disagreements)。

    以 Basic Pitch 事件为主（带音高/offset/bend），自建前端用于补漏 + 一致性确认。
    """
    bp = sorted(bp_events, key=lambda e: e["t_on"])
    on = sorted(onset_events, key=lambda e: e["t_on"])
    used_on = [False] * len(on)

    merged, disagree = [], []
    for e in bp:
        match = None
        for j, o in enumerate(on):
            if used_on[j]:
                continue
            if abs(o["t_on"] - e["t_on"]) <= onset_tol_s and (
                    not np.isfinite(o["pitch_midi"]) or
                    abs(o["pitch_midi"] - e["pitch_midi"]) <= pitch_tol):
                match = j
                break
        ev = dict(e)
        ev["agree"] = match is not None
        if match is not None:
            used_on[match] = True
            ev["t_on"] = min(ev["t_on"], on[match]["t_on"])  # onset backtrack 更准
        else:
            disagree.append({**e, "only": "bp"})
        merged.append(ev)

    # 自建前端独有的（BP 漏掉的）补进来
    for j, o in enumerate(on):
        if not used_on[j] and np.isfinite(o["pitch_midi"]):
            ev = dict(o); ev["agree"] = False
            merged.append(ev)
            disagree.append({**o, "only": "onset"})

    merged.sort(key=lambda e: e["t_on"])
    return merged, disagree
