"""2×2 stimuli 生成器 —— .tplib v2 的第一个消费者(冲刺本子 §7.1;技术母本 §4.2)。

行为转移规则(§4.2,全部作用于**残差**轨迹,不携带源的绝对量):
  pitch(cents Δ)      → 逐帧 f0:f0_note · 2^(Δp(τ)/1200)      (揉弦/滑/勾全保留)
  energy(ln-RMS Δ)    → 逐帧幅度:exp(Δe(τ))                    (呼吸/起伏)
  brightness(oct Δ)   → 逐帧倾斜:gain_k ×(f_k/1kHz)^Δb(τ)      (亮度运动)
  noise(v1 暂不用)    → 记 manifest,消费端后续接

合成路径 = 快照稳态包络(共振峰保持取 g_k)+ resynth 逐帧 f0/H —— 与 L0 同一数学,
无行为轨迹时(全零)退化为静态渲染 = 对照组,天然 α=0。

产出:同一乐句 × {Voice, Glass 音色} × {Vocal, Guitar 行为} 4 格 + 2 个静态对照,
RMS 对齐,落 assets/ear_tests/stimuli_2x2_v1/。
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from analysis.tplib import event_trajectories  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.additive import resynth  # noqa: E402

SR = 44100
HOP = 512
# 共享乐句(midi, t_on, dur_s):五音,含长短对比与大跳
PHRASE = [(60, 0.00, 0.55), (62, 0.60, 0.55), (64, 1.20, 0.90), (67, 2.20, 0.55), (72, 2.80, 1.40)]
TAIL = 0.6


def _steady_gains(snap: dict, f0_note: float, K: int) -> np.ndarray:
    """共振峰保持:稳态包络在 k·f0_note 绝对频率处采样(同 voice._compute_gain_k 语义)。"""
    H = np.asarray(snap["H_env"], np.float64)
    T = H.shape[1]
    lo, hi = int(T * 0.25), max(int(T * 0.75), int(T * 0.25) + 1)
    Hs = np.median(H[:, lo:hi], axis=1)
    pf = snap.get("partial_freqs")
    xp = (
        np.asarray(pf, np.float64)[: len(Hs)]
        if pf is not None and np.asarray(pf).size
        else np.arange(1, len(Hs) + 1) * float(snap["f0_ref"])
    )
    tgt = np.arange(1, K + 1) * f0_note
    g = np.interp(tgt, xp, Hs, left=Hs[0], right=Hs[-1])
    hi_f = xp[-1]
    above = tgt > hi_f
    if np.any(above) and Hs[-1] > 0:
        octv = np.log2(np.maximum(tgt[above], hi_f) / hi_f)
        g[above] = Hs[-1] * 10 ** (-12.0 * octv / 20.0)
    return g


def synth_note(snap: dict, midi: int, dur_s: float, traj: dict | None) -> np.ndarray:
    """一个音:行为轨迹驱动的加法重合成。traj=None → 静态对照(仅 8ms 边缘淡变)。"""
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    n = int(dur_s * SR)
    T = max(3, n // HOP)
    K = int(min(60, (SR / 2 - 800) / f0))
    g = _steady_gains(snap, f0, K)

    tau = np.linspace(0, 1, T)

    def tr(key, scale=1.0):
        if traj is None:
            return np.zeros(T)
        pts = np.asarray(traj[key]["pts"], np.float64)
        return np.interp(tau, np.linspace(0, 1, len(pts)), pts) * scale

    dp = tr("pitch")  # cents
    de = tr("energy")  # ln RMS
    db = np.clip(tr("brightness"), -1.0, 1.0)  # oct

    f0_t = f0 * 2 ** (dp / 1200.0)
    amp_t = np.exp(de)
    fk = np.arange(1, K + 1)[:, None] * f0  # [K,1] 名义频率
    tilt = (fk / 1000.0) ** db[None, :]  # [K,T] 亮度倾斜
    H = g[:, None] * amp_t[None, :] * tilt
    # 全局起收保护(静态对照也有):8ms 起 + 60ms 收
    env = np.ones(T)
    a = max(1, int(0.008 * SR / HOP))
    r = max(2, int(0.06 * SR / HOP))
    env[:a] *= np.linspace(0, 1, a)
    env[-r:] *= np.linspace(1, 0, r) ** 1.5
    H *= env[None, :]
    return resynth(H.astype(np.float32), f0_t, sr=SR, hop=HOP, length=n)


def behavior_events(wav_path: Path, max_events=8) -> list:
    """行为源 → 每事件四轨迹。先试 onset 分割,失败退回整段单事件。"""
    try:
        import soundfile as sf

        y, sr = sf.read(str(wav_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if sr != SR:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    except Exception:  # m4a 等压缩格式走 librosa/audioread
        import librosa

        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        y = y.astype(np.float32)
    spans = []
    try:
        from analysis.segment import onset_f0_events

        evs = onset_f0_events(y, sr=SR)
        spans = [
            (e["t_on"], e["t_off"]) for e in evs if e.get("t_off", 0) - e.get("t_on", 0) > 0.15
        ][:max_events]
    except Exception:
        pass
    if not spans:
        spans = [(0.0, len(y) / SR)]
    out = []
    for t0, t1 in spans:
        tj = event_trajectories(y, SR, t0, t1)
        if tj["energy"]["confidence"] > 0:
            out.append(tj)
    return out or [event_trajectories(y, SR, 0.0, len(y) / SR)]


def render_cell(snap: dict, events: list | None) -> np.ndarray:
    """乐句**连续线**合成(A1,2026-08-01):逐音独立合成再相加=采样器的失败模式
    (音间幅度归零、每音重新起振)。这里把整条乐句建成一条连续的 f0/H 时间线,
    相邻音之间 f0 在 60ms 内滑行、能量不归零——与引擎 legato 同语义。"""
    total = int((PHRASE[-1][1] + PHRASE[-1][2] + TAIL) * SR)
    T = max(8, total // HOP)
    t_fr = np.arange(T) * HOP / SR
    K_max = 60

    # 1) 名义 f0 时间线:音内保持,音间 60ms log 域滑行(=legato 连接)
    f0_nom = np.zeros(T)
    glide = 0.06
    for i, (midi, t_on, dur) in enumerate(PHRASE):
        f_here = 440.0 * 2 ** ((midi - 69) / 12)
        if i == 0:
            f0_nom[t_fr < t_on + dur] = f_here
            continue
        prev_f = 440.0 * 2 ** ((PHRASE[i - 1][0] - 69) / 12)
        m_gl = (t_fr >= t_on) & (t_fr < t_on + glide)
        u = np.clip((t_fr[m_gl] - t_on) / glide, 0, 1)
        f0_nom[m_gl] = prev_f * (f_here / prev_f) ** u  # log 域线性=听感线性
        f0_nom[(t_fr >= t_on + glide) & (t_fr < t_on + dur)] = f_here
    f0_nom[f0_nom <= 0] = f0_nom[f0_nom > 0][-1] if np.any(f0_nom > 0) else 440.0

    # 2) 行为轨迹铺到时间线(每音一条,取该音所属事件)
    dp = np.zeros(T)
    de = np.zeros(T)
    db = np.zeros(T)
    for i, (_midi, t_on, dur) in enumerate(PHRASE):
        tj = events[i % len(events)] if events else None
        m = (t_fr >= t_on) & (t_fr < t_on + dur)
        if not m.any() or tj is None:
            continue
        tau = np.linspace(0, 1, int(m.sum()))
        for key, dst in (("pitch", dp), ("energy", de), ("brightness", db)):
            pts = np.asarray(tj[key]["pts"], np.float64)
            dst[m] = np.interp(tau, np.linspace(0, 1, len(pts)), pts)
    db = np.clip(db, -1.0, 1.0)

    # 3) 一次性合成整条线(共振峰保持:每帧按当前名义音高取稳态增益)
    f0_t = f0_nom * 2 ** (dp / 1200.0)
    amp_t = np.exp(de)
    H = np.zeros((K_max, T))
    for i, (_midi, t_on, _dur) in enumerate(PHRASE):  # 分段取 g(音内音高恒定)
        # 覆盖延到**下一音起点**(乐句谱面上音间有空隙,但 legato 演奏是连着的:
        # 空隙留零=逐音重触发=采样器行为。这里让声音连续跨过空隙)
        t_next = PHRASE[i + 1][1] if i + 1 < len(PHRASE) else 1e9
        m = (t_fr >= t_on) & (t_fr < t_next)
        if not m.any():
            continue
        f_here = 440.0 * 2 ** ((midi - 69) / 12)
        K = int(min(K_max, (SR / 2 - 800) / f_here))
        g = _steady_gains(snap, f_here, K)
        fk = np.arange(1, K + 1)[:, None] * f_here
        H[:K, m] = g[:, None] * (fk / 1000.0) ** db[None, m]
    H *= amp_t[None, :]
    # 乐句整体起收(不是逐音!)+ 收尾自然衰减
    env = np.ones(T)
    a = max(1, int(0.01 * SR / HOP))
    env[:a] = np.linspace(0, 1, a)
    tail_fr = int(TAIL * SR / HOP)
    env[-tail_fr:] = np.linspace(1, 0, tail_fr) ** 2.0
    H *= env[None, :]

    y = resynth(H.astype(np.float32), f0_t, sr=SR, hop=HOP, length=total).astype(np.float64)
    r = np.sqrt((y**2).mean())
    return (y * (0.08 / max(r, 1e-9))).astype(np.float32)


def main():
    """2×2(定位修订 2026-08-01:人声出局)= {Glass, Guitar 音色} × {Glass, Guitar 行为}。
    glass 行为源(杯沿摩擦演奏)待用户录音;当前先出 guitar 行为列 + 两个静态对照。"""
    import soundfile as sf

    timbres = {
        "glass": ROOT / "assets/snapshots/wineglass.npz",
        "guitar": ROOT / "assets/snapshots/guitar.npz",
    }
    behaviors = {"guitar": ROOT / "assets/recordings/performance/guitar_perf_01.m4a"}
    glass_perf = ROOT / "assets/recordings/performance/glass_perf_01.wav"
    if glass_perf.exists():  # 用户录好即自动进 2×2
        behaviors["glass"] = glass_perf
    out = ROOT / "assets/ear_tests/stimuli_2x2_v1"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"phrase": PHRASE, "cells": {}}
    beh_events = {}
    for bname, bpath in behaviors.items():
        beh_events[bname] = behavior_events(bpath)
        print(f"behavior[{bname}]: {len(beh_events[bname])} events  ({bpath.name})")
    for tname, tpath in timbres.items():
        snap = load_snapshot(str(tpath))
        y = render_cell(snap, None)
        sf.write(str(out / f"{tname}_static.wav"), y, SR)
        manifest["cells"][f"{tname}_static"] = {"timbre": tname, "behavior": None}
        for bname in behaviors:
            y = render_cell(snap, beh_events[bname])
            sf.write(str(out / f"{tname}_x_{bname}.wav"), y, SR)
            manifest["cells"][f"{tname}_x_{bname}"] = {"timbre": tname, "behavior": bname}
            print(f"  {tname} × {bname} ✓")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest['cells'])} files → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
