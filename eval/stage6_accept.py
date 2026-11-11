"""阶段6 验收 —— FR-8 ｜ 吉他链路集成 demo。

全链路:真吉他音色快照(.tplib sustain) + 真吉他演奏法库 + 手势解释器 + 实时引擎,
用一段 MIDI 手势谱弹出带 拨/连/滑/扫/颤 的"那把吉他"。

产物:assets/demos/stage6_guitar_chain.wav + stage6_f0_trajectory.png

用法:
  python -m eval.stage6_accept                       # 用 assets/libraries/guitar.tplib
  python -m eval.stage6_accept --lib path/to.tplib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.tplib import load_tplib  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402
from synth.instrument import Instrument, build_instrument  # noqa: E402

SR = 44100
DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demos"
LIB_DIR = Path(__file__).resolve().parents[1] / "assets" / "libraries"

# 手势谱（raw MIDI）：拨 -> 连音 -> 滑音 -> 扫弦 -> 颤音
_PHRASE = [
    (0.20, "on", 52, 100),
    (0.70, "off", 52, 0),  # pluck E3
    (0.85, "on", 55, 95),  # pluck G3
    (1.05, "on", 57, 85),  # +2 半音 -> legato
    (1.45, "off", 57, 0),
    (1.45, "off", 55, 0),
    (1.65, "on", 52, 90),  # pluck
    (1.85, "on", 59, 110),  # +7 半音 -> slide(力度大=快)
    (2.45, "off", 59, 0),
    (2.45, "off", 52, 0),
    (2.70, "on", 48, 100),
    (2.715, "on", 52, 100),
    (2.73, "on", 55, 100),  # strum C
    (2.9, "cc", -1, 100),  # CC1 -> 颤音
    (3.7, "off", 48, 0),
    (3.7, "off", 52, 0),
    (3.7, "off", 55, 0),
    (3.9, "on", 60, 105),
    (4.0, "bend", -1, 0, 2.0),  # pluck + pitch bend +2
    (4.6, "off", 60, 0),
]
_DUR = 4.9


def _load_guitar(lib_path):
    if Path(lib_path).exists():
        lib = load_tplib(lib_path)
        if lib.get("sustain"):
            inst = build_instrument(lib["sustain"], noise_level=0.3)
            return inst, lib, f".tplib({Path(lib_path).name}) sustain 快照 + 演奏法库"
    # 兜底：合成吉他状乐器 + 空库
    K = 40
    inst = Instrument(
        1.0 / np.arange(1, K + 1),
        np.arange(1, K + 1, dtype=float),
        np.zeros(0),
        np.zeros(0),
        146.83,
        SR,
        noise_level=0.2,
    )
    return inst, None, "合成兜底乐器（无库）"


def _simple_f0(y, sr=SR, fmin=80.0, fmax=700.0):
    """librosa.pyin 逐帧基频（自相关，能找弱基频；PESTO 对纯重合成 conf 偏低，
    FFT 主峰又会被吉他强上谐波带偏）。"""
    import librosa

    f0, _, _ = librosa.pyin(y.astype(np.float32), sr=sr, fmin=fmin, fmax=fmax, hop_length=256)
    t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
    m = np.isfinite(f0) & (f0 > 0)
    return t[m], f0[m]


def _plot_f0(y, cooked, name="stage6"):
    ft, f0 = _simple_f0(y)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ft, f0, ".", ms=4, color="C0", label="rendered f0 (pyin)")
    # 标注手势
    cmap = {"on": "C1", "slide": "C3", "legato": "C2", "vibrato": "C4", "bend": "C5"}
    seen = set()
    for c in cooked:
        k = c[1]
        if k in cmap:
            lab = {
                "on": "pluck/strum",
                "slide": "slide",
                "legato": "legato",
                "vibrato": "vibrato(CC1)",
                "bend": "pitchbend",
            }[k]
            ax.axvline(c[0], color=cmap[k], lw=0.9, alpha=0.6, label=lab if k not in seen else None)
            seen.add(k)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f0 (Hz)")
    ax.set_title(f"{name}: guitar chain — gesture -> articulation (f0 trajectory)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEMO_DIR / f"{name}_f0_trajectory.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill 阶段6 验收（FR-8 吉他链路）")
    ap.add_argument("--lib", default=str(LIB_DIR / "guitar.tplib"))
    args = ap.parse_args(argv)

    inst, lib, src = _load_guitar(args.lib)
    print(f"乐器/库来源: {src}  (K={inst.K}, f0_ref={inst.f0_ref:.1f}Hz)")

    gi = GestureInterpreter(library=lib)
    print(
        f"slide 曲线: {'库实测 ' + str(len(gi.slide_curve)) + ' 点' if gi.slide_curve is not None else '默认线性'}"
    )
    cooked = gi.process(_PHRASE)
    from collections import Counter

    print("手势 -> 演奏法:", dict(Counter(c[1] for c in cooked)))
    for c in cooked:
        if c[1] in ("slide", "legato"):
            print(
                f"  {c[1]}: note{c[5]}->note{c[2]} ({abs(c[2] - c[5])}半音) dur={c[4] * 1000:.0f}ms"
            )

    eng = Engine(inst, eval_mode=True)
    gi.apply_to_engine(eng)
    y = eng.render_offline(cooked, _DUR)

    import soundfile as sf

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    out = DEMO_DIR / "stage6_guitar_chain.wav"
    sf.write(out, (y / (np.max(np.abs(y)) + 1e-9) * 0.97).astype(np.float32), SR)
    _plot_f0(y, cooked, "stage6")

    clean = bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0 and not eng.error_flag)
    has_all = all(
        k in Counter(c[1] for c in cooked) for k in ("on", "legato", "slide", "vibrato", "bend")
    )
    print(f"\n  demo wav -> {out.name}  ({len(y) / SR:.1f}s, clean={clean})")
    print("  f0 轨迹图 -> stage6_f0_trajectory.png")

    print("\n---------------- 阶段6 验收汇总 ----------------")
    print(
        f"  [{'OK ' if has_all else 'WARN'}] 五类手势齐发（pluck/legato/slide/strum/vibrato/bend）"
    )
    print(f"  [{'OK ' if clean else 'FAIL'}] 吉他链路集成渲染干净有界")
    print("  盲听 stage6_guitar_chain.wav：拨→连→滑→扫→颤一气呵成 = '用键盘弹出他的弹法'")
    return 0 if (clean and has_all) else 1


if __name__ == "__main__":
    raise SystemExit(main())
