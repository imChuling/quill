"""Cross-instrument behaviour transfer demo: counterfactual combinations of decoupled behaviour (retrieval library + envelope) and timbre (DDSP).

2×2 matrix, the same MIDI:
  vnB_vnT   violin behaviour × violin timbre (the original)
  gtB_gtT   guitar behaviour × guitar timbre (the original)
  gtB_vnT   guitar behaviour × violin timbre <- "violin played like a guitar" (pluck envelope + guitar micro-dynamics)
  vnB_gtT   violin behaviour × guitar timbre <- "guitar bowed like a violin" (sustain envelope + vibrato)

Narrative (ISMIR LBD): behaviour trajectories are an instrument-independent representation that can drive any timbre model.
Run: cd quill && python tools/transfer_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from neural.neural_take import NeuralTake

OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "transfer_demo"

# medium-tempo melody: has both short notes (clear guitar pluck feel) and long notes (clear violin vibrato/sustain feel)
MELODY = [
    {"midi": 62, "start": 0.0, "len": 0.4, "vel": 100},
    {"midi": 65, "start": 0.4, "len": 0.4, "vel": 105},
    {"midi": 69, "start": 0.8, "len": 0.8, "vel": 110},
    {"midi": 67, "start": 1.6, "len": 0.4, "vel": 100},
    {"midi": 65, "start": 2.0, "len": 0.4, "vel": 100},
    {"midi": 62, "start": 2.4, "len": 1.2, "vel": 108, "art": "vibrato"},
    # phrase 2 (after 0.3s)
    {"midi": 69, "start": 3.9, "len": 0.4, "vel": 105},
    {"midi": 72, "start": 4.3, "len": 0.4, "vel": 110, "art": "slide"},
    {"midi": 74, "start": 4.7, "len": 1.6, "vel": 115, "art": "vibrato"},
]

COMBOS = [
    ("vnB_vnT", "violin", "violin"),
    ("gtB_gtT", "guitar", "guitar"),
    ("gtB_vnT", "guitar", "violin"),
    ("vnB_gtT", "violin", "guitar"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nt = NeuralTake()
    for name, behavior, timbre in COMBOS:
        t0 = time.time()
        y, sr = nt.render(MELODY, instrument=timbre, behavior=behavior)
        out = OUT_DIR / f"{name}.wav"
        sf.write(str(out), y, sr)
        print(
            f"  {name}: behavior={behavior:7s} timbre={timbre:7s} "
            f"{len(y) / sr:.2f}s in {time.time() - t0:.2f}s -> {out.name}"
        )
    print(f"\n=== Transfer Demo Complete -> {OUT_DIR} ===")
    print(
        "Ear test: gtB_vnT should have plucked attacks + guitar micro-dynamics but a violin timbre; vnB_gtT should have a sustain envelope + vibrato but a guitar timbre"
    )


if __name__ == "__main__":
    main()
