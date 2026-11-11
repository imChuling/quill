"""A/B 对比生成器 —— 呈现铁律:没有对比不出声音(2026-08-01,回应"同质化"批评)。

每个素材出一对:
  sampler 臂:单点采样器的诚实仿真——快照在参考音高的重合成 = "录下的样本",
             逐音 librosa pitch_shift + 触发包络,音与音之间**断开**(采样器语义);
  quill 臂:  引擎 render_offline 全链路(演奏法 legato/slide + P1 呼吸 + 起音颗粒
             + 共振峰保持)——同一乐句被"演奏"而不是"播放"。

乐句设计 = 采样器死穴:连音上行跑句 → 大滑音 → 长音(呼吸/渐强)。
两臂 RMS 对齐;信息量公平(都只用同一个快照,无外部素材)。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from synth.render import render  # noqa: E402

SR = 44100
# 采样器死穴乐句:(midi, t_on, dur, art)
# 连音跑句(legato)→ 大滑音(slide)→ 顶点长音(听呼吸)→ 收尾滑落
PHRASE = [
    (55, 0.00, 0.42, "pluck"),
    (57, 0.45, 0.42, "legato"),
    (59, 0.90, 0.42, "legato"),
    (60, 1.35, 0.42, "legato"),
    (67, 1.90, 2.30, "slide"),  # 大七度滑上去,长音顶点
    (60, 4.40, 1.20, "slide"),  # 滑回来收束
]


def sampler_arm(snap: dict) -> np.ndarray:
    """单点采样器仿真:参考音高样本 + 逐音移调 + AD 触发包络,无连接、无呼吸。"""
    import librosa

    f0_ref = float(snap["f0_ref"])
    sample = render(
        snap, f0_ref, length=int(2.8 * SR), mode="formant", add_attack=True
    )  # "录音棚里录下的那一个样本"
    total = int((PHRASE[-1][1] + PHRASE[-1][2] + 0.8) * SR)
    y = np.zeros(total, np.float64)
    ref_midi = 69 + 12 * np.log2(f0_ref / 440.0)
    for midi, t_on, dur, _art in PHRASE:  # 采样器不懂演奏法:art 全忽略
        semis = midi - ref_midi
        seg = librosa.effects.pitch_shift(sample.astype(np.float32), sr=SR, n_steps=float(semis))[
            : int(dur * SR)
        ]
        n = len(seg)
        env = np.ones(n)
        a = max(1, int(0.005 * SR))
        r = max(2, int(0.12 * SR))
        env[:a] = np.linspace(0, 1, a)
        if n > r:
            env[-r:] = np.linspace(1, 0, r) ** 1.2  # 干巴巴的释放
        s0 = int(t_on * SR)
        y[s0 : s0 + n] += seg[:n] * env
    return y


def quill_arm(snap: dict) -> np.ndarray:
    """引擎全链路:同乐句按演奏法**演奏**(legato 连接/slide 滑行/P1 呼吸/起音颗粒)。"""
    inst = build_instrument(snap, noise_level=0.02)  # 产品默认噪声 2%
    eng = Engine(inst, eval_mode=True)
    eng.reset()
    events, prev = [], -1
    for midi, t_on, dur, art in PHRASE:
        if art == "legato" and prev >= 0:
            events.append((t_on, "legato", midi, 96, 0.04, prev))
        elif art == "slide" and prev >= 0:
            events.append((t_on, "slide", midi, 105, 0.12, prev))
        else:
            events.append((t_on, "on", midi, 100, 0.0, -1))
        events.append((t_on + dur, "off", midi, 0, 0.0, -1))
        prev = midi
    dur_total = PHRASE[-1][1] + PHRASE[-1][2] + 0.8
    return eng.render_offline(events, dur_total).astype(np.float64)


def rms(y, r=0.08):
    v = np.sqrt((np.asarray(y, np.float64) ** 2).mean())
    return (y * (r / max(v, 1e-9))).astype(np.float32)


def main():
    import soundfile as sf

    out = ROOT / "assets/ear_tests/ab_vs_sampler"
    out.mkdir(parents=True, exist_ok=True)
    for name, path in [
        ("glass", "assets/snapshots/wineglass.npz"),
        ("guitar", "assets/snapshots/guitar_rich.npz"),
    ]:
        snap = load_snapshot(str(ROOT / path))
        sf.write(str(out / f"{name}_sampler.wav"), rms(sampler_arm(snap)), SR)
        sf.write(str(out / f"{name}_quill.wav"), rms(quill_arm(snap)), SR)
        print(f"{name}: sampler/quill 对 ✓")
    print(f"→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
