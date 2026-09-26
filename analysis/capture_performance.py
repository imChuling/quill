"""演奏式取色(协议改造)—— 一段录音,同时取出"色"与"笔法"。

**为什么改**(docs/competitive_map_2026.md §3e/§3f):
旧协议要求"稳定持续音"——利于谱分析,但按 Siedenburg/Saitis/McAdams (2019):
*"a single sound-producing object can give rise to a **universe of timbres**"*,
一个稳态音只是该物件音色宇宙里的**一个点**;而乐器身份最强的信息(起音瞬态、谱波动、
激励手势)恰好被这个协议扔掉。信息不在录音里,任何 DSP 或模型都恢复不了。

**新协议**:用户录一小段**演奏这个物件**(数个音,有起音/强弱/手势)→ 从同一段录音抽:
  1. TimbreSnapshot —— 取**含起音的完整音**(不掐中段),保留瞬态与 H_env 的时变(谱波动);
  2. BehaviorTrajectories(.tplib v2)—— 每个事件四条连续残差轨迹。
两者出自同一次录音、同一件物件 → "取色笔 + 取笔法"。

引擎零改动:输出仍是标准 snapshot dict(+ 可选 behavior 列表)。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from analysis.harmonic import f0_scalar, track_f0        # noqa: E402
from analysis.timbre import analyze_timbre               # noqa: E402
from analysis.tplib import event_trajectories            # noqa: E402

SR = 44100
_MIN_EV = 0.18           # 事件最短时长(短于此不足以取色/抽轨迹)
_PRE_ROLL = 0.03         # 起音前留 30ms(瞬态完整性 > 切口干净)


def load_any(path) -> np.ndarray:
    """wav/m4a/mp3 皆可;统一 44.1k 单声道。"""
    try:
        import soundfile as sf
        y, sr = sf.read(str(path))
        y = y.mean(axis=1) if getattr(y, "ndim", 1) > 1 else y
        if sr != SR:
            import librosa
            y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=SR)
    except Exception:
        import librosa
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    return np.asarray(y, np.float32)


def find_events(y, sr=SR, max_events=12):
    """演奏段 → [(t_on, t_off)];失败退回整段。"""
    spans = []
    try:
        from analysis.segment import onset_f0_events
        for e in onset_f0_events(y, sr=sr):
            t0, t1 = float(e.get("t_on", 0)), float(e.get("t_off", 0))
            if t1 - t0 >= _MIN_EV:
                spans.append((max(0.0, t0 - _PRE_ROLL), t1))
    except Exception:
        pass
    if not spans:
        import librosa
        on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
        for i, t0 in enumerate(on):
            t1 = on[i + 1] if i + 1 < len(on) else len(y) / sr
            if t1 - t0 >= _MIN_EV:
                spans.append((max(0.0, t0 - _PRE_ROLL), min(t1, t0 + 3.0)))
    return spans[:max_events] or [(0.0, len(y) / sr)]


def _spec_flux(seg, sr):
    """谱波动:Grey (1977) 三个知觉维度之一,**是音色身份的主轴,不是杂质**。"""
    n_fft, hop = 2048, 512
    if len(seg) < n_fft * 2:
        return 0.0
    fr = np.lib.stride_tricks.sliding_window_view(seg, n_fft)[::hop]
    S = np.abs(np.fft.rfft(fr * np.hanning(n_fft), axis=1))
    S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-12)     # 每帧归一化=只看形状变化
    return float(np.mean(np.abs(np.diff(np.log(np.maximum(S, 1e-9)), axis=0))))


def _score_event(y, sr, t0, t1):
    """选"取色代表音"的打分(v2,2026-08-01 修正)。

    v1 的错误:惩罚音高不稳定(cv)、时长指数仅 0.3 —— 结果专挑**又短又死板**的音,
    把手势和谱波动全躲开了,实测新协议谱波动反而低于旧协议(0.767 vs 1.072)。
    v2:**奖励**谱波动与时长(Grey 1977:谱波动是音色三维之一),只用能量做门槛。
    """
    seg = y[int(t0 * sr):int(t1 * sr)].astype(np.float64)
    if len(seg) < int(_MIN_EV * sr):
        return -1.0, None
    e = float(np.sqrt((seg ** 2).mean()))
    if e < 1e-4:
        return -1.0, None
    f0, _, _ = track_f0(seg.astype(np.float32), sr=sr, cross_check=False)
    v = np.asarray(f0, np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 5:
        return -1.0, None
    dur = t1 - t0
    flux = _spec_flux(seg, sr)
    return (dur ** 0.7) * e * (0.3 + flux), float(np.median(v))


def capture_from_performance(path_or_audio, sr=SR, name="performance",
                             quasi_harmonic=False, verbose=True):
    """演奏段 → (snapshot, behaviors, info)。

    snapshot:  取自**含起音的完整代表音**(window=None → analyze_timbre 用整段,
               保留瞬态与 H_env 时变;与旧协议 window="auto" 掐中段相反);
    behaviors: 每个事件的 .tplib v2 四轨迹(pitch/energy/brightness/noise)。
    """
    y = (load_any(path_or_audio) if isinstance(path_or_audio, (str, Path))
         else np.asarray(path_or_audio, np.float32))
    spans = find_events(y, sr)
    scored = []
    for (t0, t1) in spans:
        s, f0m = _score_event(y, sr, t0, t1)
        if s > 0:
            scored.append((s, t0, t1, f0m))
    if not scored:
        raise ValueError("演奏段里没找到可取色的音(太短/无音高?)")
    scored.sort(reverse=True)
    _, bt0, bt1, bf0 = scored[0]

    seg = y[int(bt0 * sr):int(bt1 * sr)]
    snap = analyze_timbre(seg, sr=sr, name=name, quasi_harmonic=quasi_harmonic,
                          denoise=True, window=None, cross_check=False)

    behaviors = []
    for (t0, t1) in spans:
        tj = event_trajectories(y, sr, t0, t1)
        if tj["energy"]["confidence"] > 0:
            behaviors.append(tj)

    H = np.asarray(snap["H_env"], np.float64)
    flux = float(np.mean(np.abs(np.diff(np.log(np.maximum(H, 1e-9)), axis=1))))
    info = {"n_events": len(spans), "n_behaviors": len(behaviors),
            "pick_span": (round(bt0, 2), round(bt1, 2)), "pick_f0": round(bf0 or 0, 1),
            "H_frames": H.shape[1], "spectral_flux": round(flux, 4)}
    if verbose:
        print(f"[perf-capture] {info['n_events']} 事件 → 代表音 {info['pick_span']}s "
              f"@{info['pick_f0']}Hz;行为轨迹 {info['n_behaviors']} 条;"
              f"谱波动={info['spectral_flux']}")
    return snap, behaviors, info
