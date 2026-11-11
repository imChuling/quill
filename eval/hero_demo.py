"""7.8 hero demo: the same phrase × several capture timbres, **octave-aligned to each timbre's natural register**.

Fixes the flaw of the fr9 draft: a fixed absolute register forced each timbre to be transposed by [-28,+17] semitones, and formant extrapolation collapsed.
Here each timbre **shifts the whole phrase by an integer number of octaves** to the register closest to its f0_ref (same melody/articulation,
minimal transposition -> every timbre sounds in its own sweet spot; also closer to a real multi-instrument arrangement: bass low, flute high).

Usage: python -m eval.hero_demo
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

SR = 44100
DEMO = Path(__file__).resolve().parents[1] / "assets" / "demos"

# a fairly narrow (~1 octave), pleasant phrase that makes full use of articulations. Centre ~MIDI 57.
PHRASE = [
    (0.20, "on", 55, 100),
    (0.55, "on", 60, 95),  # +5 -> slide
    (1.05, "on", 58, 85),  # -2 -> legato
    (1.45, "off", 58, 0),
    (1.45, "off", 60, 0),
    (1.45, "off", 55, 0),
    (1.65, "on", 53, 100),
    (2.0, "on", 60, 110),  # +7 -> slide (fast)
    (2.6, "off", 60, 0),
    (2.6, "off", 53, 0),
    (2.85, "on", 53, 100),
    (2.87, "on", 57, 100),
    (2.89, "on", 60, 100),  # strum chord
    (3.1, "cc", -1, 100),  # vibrato
    (3.9, "off", 53, 0),
    (3.9, "off", 57, 0),
    (3.9, "off", 60, 0),
    (4.1, "on", 62, 100),
    (4.45, "on", 57, 95),  # -5 slide down
    (5.0, "off", 57, 0),
    (5.0, "off", 62, 0),
]
DUR = 5.3
PHRASE_CENTER = float(np.mean([e[2] for e in PHRASE if e[1] in ("on",) and e[2] > 0]))


def octave_align(phrase, f0_ref):
    """Shift the whole phrase by an integer number of octaves so its centre is closest to the snapshot's natural register (minimal transposition)."""
    midi_ref = 69 + 12 * np.log2(max(f0_ref, 1.0) / 440.0)
    shift = int(round((midi_ref - PHRASE_CENTER) / 12.0)) * 12
    out = []
    for t, typ, m, v in phrase:
        out.append((t, typ, (m + shift if typ in ("on", "off") and m > 0 else m), v))
    return out, shift


def render(snap, lib, noise=0.2):
    align, shift = octave_align(PHRASE, snap["f0_ref"])
    inst = build_instrument(snap, noise_level=noise)
    eng = Engine(inst, eval_mode=True)
    gi = GestureInterpreter(library=lib)
    gi.apply_to_engine(eng)
    cooked = gi.process(align)
    y = eng.render_offline(cooked, DUR)
    # report the transposition range (diagnostic: should all fall within ±9)
    notes = [m for _, typ, m, _ in align if typ == "on" and m > 0]
    midi_ref = round(69 + 12 * np.log2(max(snap["f0_ref"], 1.0) / 440.0))
    red = [min(notes) - midi_ref, max(notes) - midi_ref]
    return y, shift, red


def main(argv=None):
    DEMO.mkdir(parents=True, exist_ok=True)
    lib = load_tplib("assets/libraries/guitar.tplib")
    timbres = [
        ("hero2_guitar", "assets/snapshots/guitar.npz", 0.30, "guitar (anchor)"),
        ("hero2_voice", "assets/snapshots/voice_tone_mid.npz", 0.20, "voice"),
        ("hero2_pad", "assets/snapshots/synth_pad.npz", 0.15, "synth pad"),
        ("hero2_wineglass", "assets/snapshots/wineglass.npz", 0.25, "wine glass"),
    ]
    clips = []
    print("timbre          octave shift   phrase vs natural register   clean")
    for name, path, noise, label in timbres:
        snap = load_snapshot(path)
        y, shift, red = render(snap, lib, noise=noise)
        clean = bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0)
        sf.write(
            DEMO / f"{name}.wav", (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32), SR
        )
        print(f"  {label:10s}  {shift:+4d} st   [{red[0]:+d},{red[1]:+d}] st      {clean}")
        clips.append((name, label, y))

    # RMS-matched montage
    gap = np.zeros(int(0.6 * SR), np.float32)
    seq = []
    for _, _, y in clips:
        yy = y * (0.12 / (np.sqrt(np.mean(y**2)) + 1e-9))
        seq.append(np.clip(yy, -0.99, 0.99).astype(np.float32))
        seq.append(gap)
    sf.write(DEMO / "hero2_montage.wav", np.concatenate(seq).astype(np.float32), SR)
    print(
        f"\n  -> hero2_montage.wav  ({sum(len(c[2]) for c in clips) / SR + 0.6 * len(clips):.1f}s)"
    )
    print("     " + " → ".join(c[1] for c in clips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
