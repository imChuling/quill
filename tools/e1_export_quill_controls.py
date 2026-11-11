"""E1 Step 1: Export Quill behavior controls (f0 + loudness) for each melody.

Runs in the `mt` conda environment. Outputs {melody}_f0.npy and {melody}_loudness.npy
for use by tools/mididdsp_render.py (which runs in the mididdsp environment).

Usage:
    conda run -n mt python tools/e1_export_quill_controls.py
    conda run -n mt python tools/e1_export_quill_controls.py --no-slide
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
OUT = CKPT / "eval" / "e1_quill_controls"

SR = 16000
FPS = 250

MELODIES = {
    "lyrical": [
        (62, 0.8, 1), (65, 0.4, 0), (67, 1.2, 1), (65, 0.4, 0),
        (69, 0.4, 2), (72, 1.6, 1), (67, 0.8, 0), (62, 1.6, 1),
    ],
    "scale_run": [
        (62, 0.2, 0), (64, 0.2, 0), (65, 0.2, 0), (67, 0.2, 0),
        (69, 0.2, 0), (71, 0.2, 0), (72, 0.2, 0), (74, 0.2, 0),
        (76, 0.4, 0), (74, 0.4, 0), (72, 1.6, 1),
    ],
    "leaps": [
        (57, 0.4, 0), (64, 0.4, 0), (62, 0.4, 2), (69, 0.8, 1),
        (65, 0.4, 0), (72, 0.4, 2), (71, 0.8, 0), (64, 1.2, 1),
    ],
    "folk": [
        (69, 0.4, 0), (71, 0.2, 0), (72, 0.4, 0), (69, 0.4, 0),
        (67, 0.8, 1), (64, 0.4, 0), (67, 0.4, 2), (69, 1.2, 1),
        (67, 0.4, 0), (64, 0.4, 0), (62, 1.6, 1),
    ],
}


def strip_slides(melodies):
    """Replace art=2 (slide) with art=0 (plain)."""
    return {k: [(m, d, 0 if a == 2 else a) for m, d, a in v]
            for k, v in melodies.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-slide", action="store_true",
                        help="Replace slide (art=2) with plain (art=0)")
    args = parser.parse_args()

    from neural.neural_take import ConditionedNN
    from neural.perform import control_from_behavior

    np.random.seed(42)
    torch.manual_seed(42)

    melodies = strip_slides(MELODIES) if args.no_slide else MELODIES
    if args.no_slide:
        print("--no-slide: art=2 → art=0")

    ck = CKPT / "ddsp_urmp_vn_12k.pt"
    tr = CKPT / "urmp_vn_trajectories.npz"
    for p in [ck, tr]:
        if not p.exists():
            print(f"MISSING: {p}")
            sys.exit(1)

    d = np.load(str(tr))
    nn = ConditionedNN(d["features"], d["trajectories"])
    print(f"Loaded behavior library: {tr.name}")

    OUT.mkdir(parents=True, exist_ok=True)

    for mname, score in melodies.items():
        notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
        with torch.no_grad():
            f0, loud = control_from_behavior(
                notes, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0,
            )
        f0_np = f0.numpy()[0, :, 0]
        loud_np = loud.numpy()[0, :, 0]

        np.save(str(OUT / f"{mname}_f0.npy"), f0_np)
        np.save(str(OUT / f"{mname}_loudness.npy"), loud_np)
        print(f"  {mname}: f0 shape={f0_np.shape}, loudness shape={loud_np.shape}, "
              f"f0 range=[{f0_np.min():.1f}, {f0_np.max():.1f}] Hz")

    print(f"\nExported to {OUT}/")


if __name__ == "__main__":
    main()
