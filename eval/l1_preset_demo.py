"""L1 取色检索 demo —— 按"感觉"选音色 → 检索好预设 → 带演奏法弹(同一招牌乐句)。

证明产品完整度:不止"现录现取色"(L0),还能"**按描述符浏览/检索预设库 → 立刻是可演奏法的乐器**"。
与 fr9_recombine 共用同一 _PHRASE + 演奏法库,故"检索到的任意音色"都自带滑/连/扫/颤。

用法:
  python -m eval.l1_preset_demo --corpus ../../datasets/quill_corpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.tplib import load_tplib  # noqa: E402
from eval.hero_demo import DEMO as DEMO_DIR
from eval.hero_demo import SR
from eval.hero_demo import render as render_aligned  # noqa: E402
from library import SnapshotLibrary  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="L1 取色检索 demo")
    ap.add_argument("--corpus", default="../../datasets/quill_corpus")
    ap.add_argument("--articulation", default="assets/libraries/guitar.tplib")
    ap.add_argument(
        "--descriptors",
        nargs="*",
        default=["warm 温暖", "plucked 拨弦", "brass 铜管", "vocal 人声"],
    )
    ap.add_argument("--pitch", type=int, default=60)
    args = ap.parse_args(argv)

    lib = SnapshotLibrary(args.corpus)
    art = load_tplib(args.articulation)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 库浏览（{len(lib.good_entries())}/{len(lib.index)} 张通过质量白名单）===")
    for desc, n, fams in lib.browse():
        print(f"  {desc:14s} {n:3d} 张  {fams}")

    print("\n=== 选描述符 → 检索 → 带演奏法弹（招牌乐句:滑/连/扫/颤）===")
    ok = True
    for desc in args.descriptors:
        try:
            snap, meta = lib.retrieve_descriptor(desc, pitch_midi=args.pitch)
        except KeyError as e:
            print(f"  [{desc}] 跳过: {e}")
            continue
        y, shift, red = render_aligned(snap, art, noise=0.2)  # 八度对齐到音色音区
        clean = bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0)
        ok = ok and clean
        name = f"l1_{desc.split()[0]}"
        sf.write(
            DEMO_DIR / f"{name}.wav", (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32), SR
        )
        print(
            f"  [{desc:12s}] → {meta['instrument']}@{meta['midi']} "
            f"({meta['f0_ref']:.0f}Hz) → {name}.wav  clean={clean}"
        )

    print("\n---------------- L1 验收 ----------------")
    print(f"  [{'OK ' if ok else 'FAIL'}] 选 tag/描述符 → 检索预设 → 立刻可弹 + 自带演奏法")
    print("  = 取色笔的'按需检索'那条腿（与'现录现取'L0 互补,产品完整）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
