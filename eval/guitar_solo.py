"""Solo electric guitar demo: DDSP Fender + fast-attack sustain template + blues solo with bend/vibrato bloom.

Recipe (finalised by the user's ear test, 6.23):
  - timbre: DDSP Fender Strat (clean).
  - envelope: **14ms crisp attack (convex curve, punchy) + smooth exponential decay from the peak with tau≈3.5s**.
    Not note_template's raw slow climb (172ms to peak -> "swell then fade"), and the floor is not flattened either (-> "out of breath").
  - articulation: bend reaches pitch in 170ms (render_midi default), vibrato bloom (solo starts earlier).
Usage: python -m eval.guitar_solo
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural.perform import FPS, note_template, render_midi, render_strum  # noqa: E402

FENDER = "assets/demos/ddsp_fender_strat.pt"
EM_CHORD = [52, 55, 59, 64]  # Em(E3 G3 B3 E4)
DATASETS = os.environ.get("QUILL_DATASETS", str(Path.home() / "Desktop/MT/datasets"))  # IDMT-SMT-Guitar lives under $QUILL_DATASETS
NOTE_GLOB = (
    DATASETS + "/IDMT-SMT-GUITAR_V2/"
    "dataset1/Fender Strat Clean Neck SC/audio/*.wav"
)

# blues solo: pluck -> whole-tone bend + vibrato (cry) -> hammer-on -> slide -> long bend + heavy vibrato (climax)
SOLO = [
    {"midi": 59, "dur": 0.45},
    {"midi": 62, "dur": 0.95, "art": "bend", "bend": 2, "vibrato": 40},
    {"midi": 59, "dur": 0.40},
    {"midi": 57, "dur": 0.40, "art": "legato"},
    {"midi": 55, "dur": 0.50},
    {"midi": 59, "dur": 0.50, "art": "slide"},
    {"midi": 62, "dur": 0.45},
    {"midi": 64, "dur": 1.90, "art": "bend", "bend": 2, "vibrato": 50},
]


def solo_template(note_src, tau=3.5, atk_ms=14, n_frames=900):
    """Solo loudness template (log-loudness) with a fast attack + smooth exponential decay from the peak.

    tau   = decay time constant (seconds, larger sings more); atk_ms = attack duration (smaller is crisper).
    The peak level of a real Fender note is used as reference, but **the shape is fully synthetic** (avoids template slow-climb/drop artefacts).
    """
    peak = float(np.exp(note_template(note_src)).max())
    a = max(2, int(atk_ms / 1000.0 * FPS))
    out = np.empty(n_frames)
    out[:a] = peak * np.linspace(0, 1, a) ** 0.4  # fast attack (convex, punchy)
    tp = np.arange(n_frames - a) / FPS
    out[a:] = peak * np.exp(-tp / tau)  # smooth decay from the peak
    return np.log(out + 1e-6)


def _to44(y):
    return AF.resample(torch.tensor(np.asarray(y, np.float32))[None], 16000, 44100).numpy()[0]


def tail_fade(y, sr=44100, start_s=3.5):
    """Raised-cosine ramp-down 1->0 at the tail, smoothly bringing the DDSP ring-out noise floor to silence (keeps the natural decay before it)."""
    y = y.copy()
    t0 = min(len(y) - 1, int(start_s * sr))
    nf = len(y) - t0
    if nf > 1:
        y[t0:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, nf)))
    return y


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Solo electric guitar demo (solo + strum)")
    ap.add_argument("--tau", type=float, default=3.5, help="solo decay time constant (seconds)")
    args = ap.parse_args(argv)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    notes = sorted(glob.glob(NOTE_GLOB))
    if not notes:
        print("missing a real Fender note (IDMT-SMT-GUITAR) as peak reference")
        return 1
    note = notes[0]

    # (1) blues solo
    y = _to44(render_midi(FENDER, SOLO, solo_template(note, tau=args.tau), dev))
    sf.write(
        "assets/demos/guitar_solo.wav",
        (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32),
        44100,
    )
    print(f"  solo  -> guitar_solo.wav ({len(y) / 44100:.1f}s)")

    # (2) Em strum, full decay + tail ramp-down to finish cleanly
    tmpl_long = solo_template(note, tau=2.2, n_frames=2200)
    yc = _to44(render_strum(FENDER, EM_CHORD, tmpl_long, dur=7.5, spread_ms=70, device=dev))
    yc = tail_fade(yc, 44100, start_s=3.5)
    sf.write(
        "assets/demos/guitar_strum.wav",
        (yc / (np.max(np.abs(yc)) + 1e-9) * 0.92).astype(np.float32),
        44100,
    )
    print(f"  strum -> guitar_strum.wav ({len(yc) / 44100:.1f}s, Em full decay + clean ending)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
