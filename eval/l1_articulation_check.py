"""Foundation check for the L1 performance half (sprint_7.8 top priority, done retroactively).

Problem: the "performance half" of the articulation engine was never systematically verified through the real chain: hero_unified uses explicit cooked
events and bypasses gesture detection; the app handles things separately. This script goes through the **complete real chain**:
    raw MIDI (with overlapping keys/chord clusters) -> GestureInterpreter.process (FR-8 detection + templates)
    → cooked → Engine.render_offline → wav
Renders the same full-vocabulary phrase (pluck/legato/slide/vibrato/strum) on 4 L0 captures (voice/pad/wine glass/guitar),
output to assets/demos/l1_check_*.wav so a human ear can judge whether "articulation × any timbre" holds.

Usage: python -m eval.l1_articulation_check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.tplib import load_tplib  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "assets" / "demos"
SR = 44100

# full-vocabulary phrase (raw MIDI, overlapping keys trigger legato/slide, an 80ms cluster triggers strum). Centre ≈ MIDI 62.
RAW = [
    (0.00, "on", 60, 95),
    (0.75, "off", 60, 0),  # (1) pluck
    (0.85, "on", 60, 92),
    (1.60, "on", 62, 90),
    (1.64, "off", 60, 0),  # (2) legato (overlap +2)
    (2.35, "off", 62, 0),
    (2.45, "on", 62, 90),
    (3.20, "on", 69, 105),
    (3.24, "off", 62, 0),  # (3) slide (overlap +7)
    (4.10, "off", 69, 0),
    (4.30, "on", 67, 88),
    (4.55, "cc", -1, 0, 90),
    (5.90, "cc", -1, 0, 0),  # (4) vibrato (CC1)
    (6.00, "off", 67, 0),
    (6.20, "on", 55, 100),
    (6.222, "on", 59, 96),
    (6.246, "on", 62, 92),  # (5) strum (3 notes within 80ms)
    (7.60, "off", 55, 0),
    (7.60, "off", 59, 0),
    (7.60, "off", 62, 0),
]
DUR = 8.4
CENTER = 62.0

TONES = {  # name -> snapshot
    "voice": "voice.npz",
    "pad": "synth_pad.npz",
    "wineglass": "wineglass.npz",
    "guitar": "guitar_rich.npz",  # a known weak area of L0, listed honestly
}


def octave_shift(f0_ref: float) -> int:
    midi_ref = 69 + 12 * np.log2(max(float(f0_ref), 1.0) / 440.0)
    return int(round((midi_ref - CENTER) / 12.0)) * 12


def shifted_raw(shift: int):
    out = []
    for e in RAW:
        t, kind, note = e[0], e[1], e[2]
        out.append(
            (t, kind, note + shift if (kind in ("on", "off") and note >= 0) else note, *e[3:])
        )
    return out


def f0_med(y, a, b):
    """Median f0 within the segment (librosa yin; rough check of pitch/slide landing)."""
    import librosa

    seg = y[int(a * SR) : int(b * SR)]
    if len(seg) < 2048:
        return float("nan")
    f = librosa.yin(seg.astype(np.float64), fmin=50, fmax=1600, sr=SR)
    f = f[np.isfinite(f)]
    return float(np.median(f)) if f.size else float("nan")


def main():
    lib = load_tplib(str(ROOT / "assets" / "libraries" / "guitar.tplib"))
    gi = GestureInterpreter(library=lib)
    print(
        f"template: guitar.tplib | legato≤{gi.legato_max_st}st | strum {gi.strum_min} notes/{gi.strum_window * 1000:.0f}ms"
    )
    for name, snap_file in TONES.items():
        snap = load_snapshot(str(ROOT / "assets" / "snapshots" / snap_file))
        shift = octave_shift(snap["f0_ref"])
        cooked = gi.process(shifted_raw(shift))
        kinds = [c[1] for c in cooked]
        inst = build_instrument(snap, noise_level=0.2)
        eng = Engine(inst, eval_mode=True)
        gi.apply_to_engine(eng)
        y = eng.render_offline(cooked, DUR)
        path = DEMO / f"l1_check_{name}.wav"
        sf.write(str(path), y, SR)
        # objective sanity: detected vocabulary + slide landing + vibrato modulation
        tgt = 440.0 * 2 ** ((69 + shift - 69) / 12)  # slide target note
        f_slide_end = f0_med(y, 3.85, 4.08)
        det = {k: kinds.count(k) for k in ("on", "legato", "slide", "vibrato")}
        print(
            f"[{name:10s}] shift={shift:+d}  detected={det}  "
            f"slide landing {f_slide_end:.0f}Hz (target {tgt:.0f})  peak {np.abs(y).max():.2f}  -> {path.name}"
        )
    print("\nListening judgement is left to the human ear: assets/demos/l1_check_*.wav")


if __name__ == "__main__":
    main()
