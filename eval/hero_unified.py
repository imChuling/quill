"""7.8 hero demo (unified phrase version v2): four timbres play **the same monophonic score** (note-for-note identical).

Conflict: L0 is 16-voice polyphonic (can strum), the DDSP guitar is monophonic (cannot handle chords) -> when the previous version unified everything through the
gesture engine, the guitar notes were all wrong. Solution: **a monophonic phrase + two independent deterministic paths from the same score** (both build f0 directly from the score,
without the error-prone gesture detection):
  - guitar: control_from_score(score, real note template) -> DDSP Fender (winning recipe, correct notes).
  - voice/pad/wine glass: score -> explicit cooked events (slide/vibrato written directly, not detected) -> L0 engine,
                  octave-aligned to each timbre's register.
Four clips = the same performance × different captures.

Usage: python -m eval.hero_unified
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

from analysis.tplib import load_tplib  # noqa: E402
from neural.perform import control_from_score, note_template, render  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

SR = 44100
DEMO = Path(__file__).resolve().parents[1] / "assets" / "demos"
FENDER = "assets/demos/ddsp_fender_strat.pt"
DATASETS = os.environ.get("QUILL_DATASETS", str(Path.home() / "Desktop/MT/datasets"))  # IDMT-SMT-Guitar lives under $QUILL_DATASETS
NOTE_GLOB = (
    DATASETS + "/IDMT-SMT-GUITAR_V2/"
    "dataset1/Fender Strat Clean Neck SC/audio/*.wav"
)

# monophonic signature phrase (no chords): pluck + slide + vibrato. Native guitar register MIDI 53–62.
SCORE = [
    {"midi": 55, "dur": 0.5},
    {"midi": 60, "dur": 0.5, "slide": True},  # slide +5
    {"midi": 58, "dur": 0.5},
    {"midi": 53, "dur": 0.5},
    {"midi": 60, "dur": 0.6, "slide": True},  # slide +7
    {"midi": 57, "dur": 0.5},
    {"midi": 62, "dur": 0.9, "vibrato": 30},  # held note + vibrato
    {"midi": 57, "dur": 0.8, "slide": True},  # slide down
]
DUR = sum(s["dur"] for s in SCORE) + 1.0
CENTER = float(np.mean([s["midi"] for s in SCORE]))


def _octave_shift(f0_ref):
    midi_ref = 69 + 12 * np.log2(max(f0_ref, 1.0) / 440.0)
    return int(round((midi_ref - CENTER) / 12.0)) * 12


def score_to_cooked(score, shift, vib_max_cents, vel=100, t0=0.1):
    """Monophonic score -> explicit cooked events (slide/vibrato written directly, no gesture detection)."""
    cooked, cum, cur = [], t0, None
    for seg in score:
        midi = seg["midi"] + shift
        if seg.get("slide") and cur is not None:
            cooked.append((cum, "slide", midi, vel, 0.12, cur))  # continue and slide to midi
        else:
            if cur is not None:
                cooked.append((cum - 0.006, "off", cur, 0, 0.0, -1))
            cooked.append((cum, "on", midi, vel, 0.0, -1))
        if seg.get("vibrato"):
            depth = min(1.0, seg["vibrato"] / max(vib_max_cents, 1e-6))
            cooked.append((cum + 0.08, "vibrato", -1, 0, depth, -1))
        cur = midi
        cum += seg["dur"]
    cooked.append((cum, "off", cur, 0, 0.0, -1))
    cooked.sort(key=lambda x: x[0])
    return cooked


def render_L0(snap, lib, noise=0.02):  # noise layer 2% (decided by the user's ear test on 7.3, same as the app default)
    gi = GestureInterpreter(library=lib)
    inst = build_instrument(snap, noise_level=noise)
    eng = Engine(inst, eval_mode=True)
    gi.apply_to_engine(eng)  # inject slide curve + vibrato rate
    shift = _octave_shift(snap["f0_ref"])
    cooked = score_to_cooked(SCORE, shift, gi.vib_max_cents)
    return eng.render_offline(cooked, DUR), shift


def render_guitar(lib, device=None):
    note_src = sorted(glob.glob(NOTE_GLOB))[0]
    tmpl = note_template(note_src)
    f0, lo, _ = control_from_score(SCORE, template=tmpl)  # same score
    y = render(FENDER, f0, lo, device)  # 16k DDSP
    return AF.resample(torch.tensor(np.asarray(y, np.float32))[None], 16000, 44100).numpy()[0]


def main(argv=None):
    DEMO.mkdir(parents=True, exist_ok=True)
    lib = load_tplib("assets/libraries/guitar.tplib")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    clips = []

    gy = render_guitar(lib, device=dev)
    sf.write(
        DEMO / "hero3_guitar.wav", (gy / (np.max(np.abs(gy)) + 1e-9) * 0.95).astype(np.float32), SR
    )
    print(f"  guitar (DDSP control_from_score, same score)  {len(gy) / SR:.1f}s")
    clips.append(("guitar (DDSP)", gy))

    for path, noise, fid, label in [
        ("assets/snapshots/voice_tone_mid.npz", 0.20, "voice", "voice"),
        ("assets/snapshots/synth_pad.npz", 0.15, "pad", "synth pad"),
        ("assets/snapshots/wineglass.npz", 0.25, "wineglass", "wine glass"),
    ]:
        snap = load_snapshot(path)
        y, shift = render_L0(snap, lib, noise=noise)
        sf.write(
            DEMO / f"hero3_{fid}.wav",
            (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32),
            SR,
        )
        print(f"  {label:8s}(L0, same score, octave {shift:+d})  {len(y) / SR:.1f}s")
        clips.append((label, y))

    gap = np.zeros(int(0.6 * SR), np.float32)
    seq = []
    for _, y in clips:
        yy = y * (0.13 / (np.sqrt(np.mean(y**2)) + 1e-9))
        seq.append(np.clip(yy, -0.99, 0.99).astype(np.float32))
        seq.append(gap)
    sf.write(DEMO / "hero_unified_montage.wav", np.concatenate(seq).astype(np.float32), SR)
    print("\n  -> hero_unified_montage.wav  " + " → ".join(c[0] for c in clips))
    print("  four clips = the same monophonic score (note-for-note identical) × different captures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
