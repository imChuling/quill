"""P1 A/B:表现层(谱形轨迹+亮度随电平)开 vs 关,同乐句渲染 —— 耳测用。

乐句设计给"风琴效应"最难的场景:长持续单音(6s) + 按住的三和弦(4s)。
输出 assets/demos/p1_{tone}_{off,on}.wav + 客观指标(持续段谱通量,on 应显著更高)。

用法:python -m eval.p1_ab
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from synth.voice import set_liveness  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "assets" / "demos"
SR = 44100
DUR = 11.5

TONES = {"voice": "voice.npz", "pad": "synth_pad.npz", "wineglass": "wineglass.npz"}

# 长单音(6s) + 三和弦按住(4s)——静态频谱最藏不住的乐句
EVENTS = [
    (0.2, "on", 57, 96, 0.0, -1),
    (6.2, "off", 57, 0, 0.0, -1),
    (6.6, "on", 53, 92, 0.0, -1),
    (6.62, "on", 57, 88, 0.0, -1),
    (6.64, "on", 60, 90, 0.0, -1),
    (10.8, "off", 53, 0, 0.0, -1),
    (10.8, "off", 57, 0, 0.0, -1),
    (10.8, "off", 60, 0, 0.0, -1),
]


def octave_shift(f0_ref):
    midi_ref = 69 + 12 * np.log2(max(float(f0_ref), 1.0) / 440.0)
    return int(round((midi_ref - 57.0) / 12.0)) * 12


def sustain_motion(y, a, b):
    """持续段的谱运动:(谱通量, 谱质心std Hz)。风琴=双低;活=至少一项明显。"""
    seg = y[int(a * SR) : int(b * SR)]
    n_fft, hop = 2048, 512
    frames = [
        np.abs(np.fft.rfft(seg[i : i + n_fft] * np.hanning(n_fft)))
        for i in range(0, len(seg) - n_fft, hop)
    ]
    S = np.array(frames)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    cent = (S * freqs).sum(axis=1) / (S.sum(axis=1) + 1e-12)
    Sn = S / (S.max() + 1e-12)
    return float(np.mean(np.abs(np.diff(Sn, axis=0))) * 1e3), float(np.std(cent))


def main():
    for name, f in TONES.items():
        snap = load_snapshot(str(ROOT / "assets" / "snapshots" / f))
        sh = octave_shift(snap["f0_ref"])
        evs = [(t, k, (n + sh if n >= 0 else n), v, val, n2) for (t, k, n, v, val, n2) in EVENTS]
        row, ys = {}, {}
        for mode, (spec, dark, br) in {
            "off": (0.0, 0.0, 0.0),
            "on": (0.85, 0.0, 0.8),
            "max": (1.8, 0.0, 1.6),
        }.items():  # max=夸张教学档:听懂"呼吸"用
            inst = build_instrument(snap, noise_level=0.2)
            eng = Engine(inst, eval_mode=True)
            set_liveness(spec, dark, br)  # 必须在 Engine 之后(引擎 init 会按 config 重设)
            y = eng.render_offline(evs, DUR)
            sf.write(str(DEMO / f"p1_{name}_{mode}.wav"), y, SR)
            ys[mode] = y
            row[mode] = sustain_motion(y, 2.0, 5.8)
        # 真咔哒判据 = 残差导数**峰度**(正常音频~3-10;孤立咔哒会 >30)。
        # 注:脉冲计数在调制深度大时失效(残差本身成音频级信号),7.2 调查证实误报。
        kur = {}
        for m in ("on", "max"):
            dr = np.diff(ys[m] - ys["off"])
            m2 = (dr**2).mean() + 1e-30
            kur[m] = float((dr**4).mean() / m2**2 - 3.0)
        F = inst.spec_traj.shape[1] if inst.spec_traj is not None else 0
        (_, c0), (_, c1), (_, c2) = row["off"], row["on"], row["max"]
        amp_dev = float(np.abs(inst.amp_traj - 1).mean()) if inst.amp_traj is not None else 0.0
        print(
            f"[{name:10s}] F={F:3d} 响度呼吸幅={amp_dev:.3f}  质心std {c0:5.1f}→{c1:5.1f}→{c2:5.1f} Hz  "
            f"残差峰度 on={kur['on']:.1f} max={kur['max']:.1f} → {'干净' if kur['on'] < 30 else '有咔哒!'}"
        )
    set_liveness(0.85, 0.0, 0.8)  # 还原默认
    print("\n耳测:assets/demos/p1_{voice,pad,wineglass}_{off,on}.wav(先 off 后 on)")


if __name__ == "__main__":
    main()
