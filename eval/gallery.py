"""Recombination gallery: batch-produce "any timbre × the same articulation" demos (video material + FR-9 generalisation showcase).

The same gesture score + the same guitar articulation library applied to different timbre snapshots: the guitar itself / voice / wine glass...
Identical technique, different timbres: timbre and articulation are visibly (audibly) decoupled at a glance.

Usage: python -m eval.gallery   [--lib assets/libraries/guitar.tplib]
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

# (snapshot file, display name, noise-side ratio)
_TIMBRES = [
    ("guitar.npz", "guitar", 0.3),  # the source instrument itself (reference)
    ("voice_tone_mid.npz", "voice", 0.2),  # a voice that can do guitar slides
    ("wineglass.npz", "wineglass", 0.15),  # a wine glass that can do guitar slides
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="FR-9 recombination gallery")
    ap.add_argument("--lib", default=str(ROOT / "assets" / "libraries" / "guitar.tplib"))
    args = ap.parse_args(argv)

    if not Path(args.lib).exists():
        print(f"[missing] articulation library not found: {args.lib} (run eval.build_library first)")
        return 2
    lib = load_tplib(args.lib)
    aname = Path(args.lib).stem
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Articulation library: {aname}  | same gesture score applied to different timbres:")
    made = []
    for fn, label, noise in _TIMBRES:
        p = SNAP_DIR / fn
        if not p.exists():
            print(f"  [skipped] {label}: missing snapshot {fn}")
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

    print(f"\nGallery produced {len(made)} clips, directory {DEMO_DIR}")
    print(
        "Video use: play the three side by side: technique aligned note by note, timbres guitar/voice/wineglass differ = decoupling + recombination at a glance"
    )
    return 0 if made else 2


if __name__ == "__main__":
    raise SystemExit(main())
