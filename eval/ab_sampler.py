"""A/B comparison generator: presentation rule, no sound without a comparison (2026-08-01, responding to the "homogeneity" critique).

One pair per source:
  sampler arm: an honest simulation of a single-point sampler: the snapshot resynthesised at the reference pitch = "the recorded sample",
             per-note librosa pitch_shift + trigger envelope, notes are **disconnected** (sampler semantics);
  quill arm:  full engine render_offline chain (articulation legato/slide + P1 breathing + attack grain
             + formant preservation): the same phrase is "performed" rather than "played back".

Phrase design = the sampler's weak spot: legato ascending run -> large slide -> long note (breathing/crescendo).
Both arms RMS-matched; information is fair (both use only the same snapshot, no external material).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from synth.render import render  # noqa: E402

SR = 44100
# phrase targeting the sampler's weak spot: (midi, t_on, dur, art)
# legato run (legato) -> large slide (slide) -> long top note (listen for breathing) -> closing slide down
PHRASE = [
    (55, 0.00, 0.42, "pluck"),
    (57, 0.45, 0.42, "legato"),
    (59, 0.90, 0.42, "legato"),
    (60, 1.35, 0.42, "legato"),
    (67, 1.90, 2.30, "slide"),  # slide up a major seventh, long top note
    (60, 4.40, 1.20, "slide"),  # slide back down to close
]


def sampler_arm(snap: dict) -> np.ndarray:
    """Single-point sampler simulation: reference-pitch sample + per-note transposition + AD trigger envelope, no connection, no breathing."""
    import librosa

    f0_ref = float(snap["f0_ref"])
    sample = render(
        snap, f0_ref, length=int(2.8 * SR), mode="formant", add_attack=True
    )  # "the one sample recorded in the studio"
    total = int((PHRASE[-1][1] + PHRASE[-1][2] + 0.8) * SR)
    y = np.zeros(total, np.float64)
    ref_midi = 69 + 12 * np.log2(f0_ref / 440.0)
    for midi, t_on, dur, _art in PHRASE:  # the sampler knows nothing about articulation: art is ignored
        semis = midi - ref_midi
        seg = librosa.effects.pitch_shift(sample.astype(np.float32), sr=SR, n_steps=float(semis))[
            : int(dur * SR)
        ]
        n = len(seg)
        env = np.ones(n)
        a = max(1, int(0.005 * SR))
        r = max(2, int(0.12 * SR))
        env[:a] = np.linspace(0, 1, a)
        if n > r:
            env[-r:] = np.linspace(1, 0, r) ** 1.2  # dry release
        s0 = int(t_on * SR)
        y[s0 : s0 + n] += seg[:n] * env
    return y


def quill_arm(snap: dict) -> np.ndarray:
    """Full engine chain: the same phrase **performed** with articulation (legato connection/slide glide/P1 breathing/attack grain)."""
    inst = build_instrument(snap, noise_level=0.02)  # product default noise 2%
    eng = Engine(inst, eval_mode=True)
    eng.reset()
    events, prev = [], -1
    for midi, t_on, dur, art in PHRASE:
        if art == "legato" and prev >= 0:
            events.append((t_on, "legato", midi, 96, 0.04, prev))
        elif art == "slide" and prev >= 0:
            events.append((t_on, "slide", midi, 105, 0.12, prev))
        else:
            events.append((t_on, "on", midi, 100, 0.0, -1))
        events.append((t_on + dur, "off", midi, 0, 0.0, -1))
        prev = midi
    dur_total = PHRASE[-1][1] + PHRASE[-1][2] + 0.8
    return eng.render_offline(events, dur_total).astype(np.float64)


def rms(y, r=0.08):
    v = np.sqrt((np.asarray(y, np.float64) ** 2).mean())
    return (y * (r / max(v, 1e-9))).astype(np.float32)


def main():
    import soundfile as sf

    out = ROOT / "assets/ear_tests/ab_vs_sampler"
    out.mkdir(parents=True, exist_ok=True)
    for name, path in [
        ("glass", "assets/snapshots/wineglass.npz"),
        ("guitar", "assets/snapshots/guitar_rich.npz"),
    ]:
        snap = load_snapshot(str(ROOT / path))
        sf.write(str(out / f"{name}_sampler.wav"), rms(sampler_arm(snap)), SR)
        sf.write(str(out / f"{name}_quill.wav"), rms(quill_arm(snap)), SR)
        print(f"{name}: sampler/quill pair ✓")
    print(f"→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
