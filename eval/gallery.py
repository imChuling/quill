"""重组画廊 —— 批量产出"任意音色 × 同一演奏法"的 demo（视频素材 + FR-9 泛化展示）。

同一手势谱 + 同一吉他演奏法库，套到不同音色快照上：吉他本身 / 人声 / 酒杯……
弹法完全相同、音色各异 —— 一眼（一耳）看出音色与演奏法解耦。

用法：python -m eval.gallery   [--lib assets/libraries/guitar.tplib]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval.fr9_recombine as fr9  # noqa: E402
from analysis.tplib import load_tplib  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

SR = 44100
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "assets" / "demos"
SNAP_DIR = ROOT / "assets" / "snapshots"

# (快照文件, 展示名, 噪声端比例)
_TIMBRES = [
    ("guitar.npz", "guitar", 0.3),  # 源乐器本身（对照）
    ("voice_tone_mid.npz", "voice", 0.2),  # 会吉他滑音的人声
    ("wineglass.npz", "wineglass", 0.15),  # 会吉他滑音的酒杯
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="FR-9 重组画廊")
    ap.add_argument("--lib", default=str(ROOT / "assets" / "libraries" / "guitar.tplib"))
    args = ap.parse_args(argv)

    if not Path(args.lib).exists():
        print(f"[缺] 演奏法库不存在: {args.lib}（先跑 eval.build_library）")
        return 2
    lib = load_tplib(args.lib)
    aname = Path(args.lib).stem
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"演奏法库: {aname}  | 同一手势谱套到不同音色：")
    made = []
    for fn, label, noise in _TIMBRES:
        p = SNAP_DIR / fn
        if not p.exists():
            print(f"  [跳过] {label}: 缺快照 {fn}")
            continue
        snap = load_snapshot(p)
        inst = build_instrument(snap, noise_level=noise)
        eng = Engine(inst, eval_mode=True)
        gi = GestureInterpreter(library=lib)
        gi.apply_to_engine(eng)
        y = eng.render_offline(gi.process(fr9._PHRASE), fr9._DUR)
        out = DEMO_DIR / f"gallery_{label}_x_{aname}.wav"
        sf.write(out, (y / (np.max(np.abs(y)) + 1e-9) * 0.97).astype(np.float32), SR)
        clean = bool(np.all(np.isfinite(y)) and not eng.error_flag)
        made.append((label, out.name, clean))
        print(f"  ✓ {label:10s} × {aname}  -> {out.name}  ({'clean' if clean else 'WARN'})")

    print(f"\n画廊产出 {len(made)} 条，目录 {DEMO_DIR}")
    print(
        "视频用法：三条并排播——弹法逐音对齐、音色 guitar/voice/wineglass 各异 = 解耦+重组一目了然"
    )
    return 0 if made else 2


if __name__ == "__main__":
    raise SystemExit(main())
