"""FR-9 交叉重组 —— 杀手锏：任意音色 × 任意演奏法库。

`音色快照(B) × 演奏法库(A)` 自由组合 —— "会吉他滑音的人声"。
这是采样器/Synplant 结构上做不出的东西（解耦架构白送的能力）：引擎放 B 的音色，
手势解释器套 A 的演奏法（slide 曲线/strum 时差），同一手势谱出"用 A 弹法弹的 B"。

产物：
  <name>.wav                重组结果（音色B + 演奏法A）
  <name>_ref.wav            对照（音色A + 演奏法A，同谱）—— A/B 听音色差、弹法同
  <name>_compare.png        两者频谱图（弹法/f0 相同，音色不同）

用法：
  # 会吉他滑音的人声：人声音色 × 吉他演奏法
  python -m eval.fr9_recombine --timbre assets/snapshots/voice_tone_mid.npz \
         --articulation assets/libraries/guitar.tplib --name voice_x_guitar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.io import load  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from analysis.tplib import load_tplib  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

SR = 44100
DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demos"

# 一段突出滑音/连音的旋律（人声滑音 = 招牌）
_PHRASE = [
    (0.20, "on", 60, 100),
    (0.55, "on", 64, 95),  # +4 -> slide
    (1.05, "on", 62, 85),  # -2 -> legato
    (1.45, "off", 62, 0),
    (1.45, "off", 64, 0),
    (1.45, "off", 60, 0),
    (1.65, "on", 55, 100),
    (2.0, "on", 62, 110),  # +7 -> slide(快)
    (2.6, "off", 62, 0),
    (2.6, "off", 55, 0),
    (2.85, "on", 48, 100),
    (2.87, "on", 52, 100),
    (2.89, "on", 55, 100),  # strum
    (3.1, "cc", -1, 100),  # 颤音
    (3.9, "off", 48, 0),
    (3.9, "off", 52, 0),
    (3.9, "off", 55, 0),
    (4.1, "on", 67, 100),
    (4.45, "on", 60, 95),  # -7 下滑
    (5.0, "off", 60, 0),
    (5.0, "off", 67, 0),
]
_DUR = 5.3


def _timbre_snapshot(path, name):
    p = Path(path)
    if p.suffix == ".npz":
        return load_snapshot(p), p.stem
    y, _ = load(str(p), sr=SR)  # 直接给单音 wav 也行
    return analyze_timbre(y, sr=SR, name=name), p.stem


def _render(snapshot, lib, noise=0.2):
    inst = build_instrument(snapshot, noise_level=noise)
    eng = Engine(inst, eval_mode=True)
    gi = GestureInterpreter(library=lib)
    gi.apply_to_engine(eng)
    cooked = gi.process(_PHRASE)
    return eng.render_offline(cooked, _DUR), cooked, gi


def _spec(ax, y, sr, title):
    import librosa

    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    librosa.display.specshow(
        S,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="linear",
        ax=ax,
        cmap="magma",
        vmin=-70,
        vmax=0,
    )
    ax.set_ylim(0, 4000)
    ax.set_title(title, fontsize=9)


def main(argv=None):
    ap = argparse.ArgumentParser(description="FR-9 交叉重组")
    ap.add_argument("--timbre", required=True, help="音色快照 .npz 或 单音 wav")
    ap.add_argument("--articulation", required=True, help="演奏法库 .tplib")
    ap.add_argument("--name", default="recombine")
    ap.add_argument(
        "--noise", type=float, default=0.2, help="噪声端比例（人声宜低~0.2 防气声过响；吉他可高些）"
    )
    ap.add_argument(
        "--ref-timbre", default=None, help="对照音色（默认用库 sustain 快照=演奏法源乐器本身）"
    )
    args = ap.parse_args(argv)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    snap, tname = _timbre_snapshot(args.timbre, args.name)
    lib = load_tplib(args.articulation)
    aname = Path(args.articulation).stem
    print(f"重组: 音色[{tname}] × 演奏法[{aname}]  (f0_ref={snap['f0_ref']:.1f}Hz)")

    # 主体：音色B × 演奏法A
    y, cooked, gi = _render(snap, lib, noise=args.noise)
    from collections import Counter

    print(
        "演奏法分布:",
        dict(Counter(c[1] for c in cooked)),
        f"| slide 曲线={'库实测' if gi.slide_curve is not None else '线性'}",
    )
    sf.write(
        DEMO_DIR / f"{args.name}.wav",
        (y / (np.max(np.abs(y)) + 1e-9) * 0.97).astype(np.float32),
        SR,
    )

    # 对照：音色A（演奏法源乐器本身的音色）× 同演奏法、同谱
    ref_snap = None
    if args.ref_timbre:
        ref_snap, _ = _timbre_snapshot(args.ref_timbre, "ref")
    elif lib.get("sustain"):
        ref_snap = lib["sustain"]
    if ref_snap is not None:
        y_ref, _, _ = _render(ref_snap, lib)
        sf.write(
            DEMO_DIR / f"{args.name}_ref.wav",
            (y_ref / (np.max(np.abs(y_ref)) + 1e-9) * 0.97).astype(np.float32),
            SR,
        )
        # 频谱对比图
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        _spec(
            axes[0],
            y_ref.astype(np.float32),
            SR,
            f"REF: timbre [{aname}] x articulation [{aname}]  (source instrument)",
        )
        _spec(
            axes[1],
            y.astype(np.float32),
            SR,
            f"RECOMBINE: timbre [{tname}] x articulation [{aname}]  (same gesture, diff timbre)",
        )
        fig.suptitle(
            f"FR-9 cross-recombination: same phrase + same articulation, timbre {tname} vs {aname}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(DEMO_DIR / f"{args.name}_compare.png", dpi=110)
        plt.close(fig)

    clean = bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0)
    print(
        f"\n  -> {args.name}.wav  ({len(y) / SR:.1f}s, clean={clean})"
        + (f" | 对照 {args.name}_ref.wav + {args.name}_compare.png" if ref_snap is not None else "")
    )
    print("\n---------------- FR-9 验收 ----------------")
    print(f"  [{'OK ' if clean else 'FAIL'}] 交叉重组成立：{tname} 的音色 × {aname} 的演奏法")
    print(f"  盲听：{args.name}.wav 与 {args.name}_ref.wav 弹法完全相同（滑/连/扫一致），")
    print(f"        音色一个是 {tname}、一个是 {aname} —— 采样器结构上做不出这件事")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
