"""L1 capture retrieval demo: choose a timbre by "feel" -> retrieve a good preset -> play with articulations (the same signature phrase).

Shows product completeness: not only "record and capture on the spot" (L0), but also "**browse/retrieve the preset library by descriptor -> immediately a playable instrument with articulations**".
Shares the same _PHRASE + articulation library with fr9_recombine, so "any retrieved timbre" comes with slide/legato/strum/vibrato.

Usage:
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
    ap = argparse.ArgumentParser(description="L1 capture retrieval demo")
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

    print(f"=== Library browse ({len(lib.good_entries())}/{len(lib.index)} pass the quality whitelist) ===")
    for desc, n, fams in lib.browse():
        print(f"  {desc:14s} {n:3d}  {fams}")

    print("\n=== Choose descriptor -> retrieve -> play with articulations (signature phrase: slide/legato/strum/vibrato) ===")
    ok = True
    for desc in args.descriptors:
        try:
            snap, meta = lib.retrieve_descriptor(desc, pitch_midi=args.pitch)
        except KeyError as e:
            print(f"  [{desc}] skipped: {e}")
            continue
        y, shift, red = render_aligned(snap, art, noise=0.2)  # octave-aligned to the timbre's register
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

    print("\n---------------- L1 acceptance ----------------")
    print(f"  [{'OK ' if ok else 'FAIL'}] choose tag/descriptor -> retrieve preset -> immediately playable + articulations included")
    print("  = the 'retrieve on demand' leg of the capture pen (complements 'record and capture' L0; the product is complete)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
