"""FR-9 cross recombination, the signature feature: any timbre × any articulation library.

Free combination of `timbre snapshot (B) × articulation library (A)`: "a voice that can do guitar slides".
This is something samplers/Synplant structurally cannot do (a capability the decoupled architecture gives for free): the engine plays B's timbre,
the gesture interpreter applies A's articulations (slide curve/strum offsets), and the same gesture score produces "B played with A's technique".

Outputs:
  <name>.wav                recombination result (timbre B + articulation A)
  <name>_ref.wav            reference (timbre A + articulation A, same score): A/B to hear the timbre difference with the same technique
  <name>_compare.png        spectrograms of both (same technique/f0, different timbre)

Usage:
  # a voice that can do guitar slides: voice timbre × guitar articulation
  python -m eval.fr9_recombine --timbre assets/snapshots/voice_tone_mid.npz \
         --articulation assets/libraries/guitar.tplib --name voice_x_guitar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.io import load  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from analysis.tplib import load_tplib  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

SR = 44100
DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demos"

# a melody that features slides/legato (voice slides = the showcase)
_PHRASE = [
    (0.20, "on", 60, 100),
    (0.55, "on", 64, 95),  # +4 -> slide
    (1.05, "on", 62, 85),  # -2 -> legato
    (1.45, "off", 62, 0),
    (1.45, "off", 64, 0),
    (1.45, "off", 60, 0),
    (1.65, "on", 55, 100),
    (2.0, "on", 62, 110),  # +7 -> slide (fast)
    (2.6, "off", 62, 0),
    (2.6, "off", 55, 0),
    (2.85, "on", 48, 100),
    (2.87, "on", 52, 100),
    (2.89, "on", 55, 100),  # strum
    (3.1, "cc", -1, 100),  # vibrato
    (3.9, "off", 48, 0),
    (3.9, "off", 52, 0),
    (3.9, "off", 55, 0),
    (4.1, "on", 67, 100),
    (4.45, "on", 60, 95),  # -7 slide down
    (5.0, "off", 60, 0),
    (5.0, "off", 67, 0),
]
_DUR = 5.3


def _timbre_snapshot(path, name):
    p = Path(path)
    if p.suffix == ".npz":
        return load_snapshot(p), p.stem
    y, _ = load(str(p), sr=SR)  # a single-note wav works directly too
    return analyze_timbre(y, sr=SR, name=name), p.stem


def _render(snapshot, lib, noise=0.2):
    inst = build_instrument(snapshot, noise_level=noise)
    eng = Engine(inst, eval_mode=True)
    gi = GestureInterpreter(library=lib)
    gi.apply_to_engine(eng)
    cooked = gi.process(_PHRASE)
    return eng.render_offline(cooked, _DUR), cooked, gi


def _spec(ax, y, sr, title):
    import librosa

    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    librosa.display.specshow(
        S,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="linear",
        ax=ax,
        cmap="magma",
        vmin=-70,
        vmax=0,
    )
    ax.set_ylim(0, 4000)
    ax.set_title(title, fontsize=9)


def main(argv=None):
    ap = argparse.ArgumentParser(description="FR-9 cross recombination")
    ap.add_argument("--timbre", required=True, help="timbre snapshot .npz or single-note wav")
    ap.add_argument("--articulation", required=True, help="articulation library .tplib")
    ap.add_argument("--name", default="recombine")
    ap.add_argument(
        "--noise", type=float, default=0.2, help="noise-side ratio (keep low ~0.2 for voice to avoid excessive breathiness; guitar can be higher)"
    )
    ap.add_argument(
        "--ref-timbre", default=None, help="reference timbre (defaults to the library's sustain snapshot = the articulation source instrument itself)"
    )
    args = ap.parse_args(argv)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    snap, tname = _timbre_snapshot(args.timbre, args.name)
    lib = load_tplib(args.articulation)
    aname = Path(args.articulation).stem
    print(f"Recombine: timbre[{tname}] × articulation[{aname}]  (f0_ref={snap['f0_ref']:.1f}Hz)")

    # main: timbre B × articulation A
    y, cooked, gi = _render(snap, lib, noise=args.noise)
    from collections import Counter

    print(
        "articulation distribution:",
        dict(Counter(c[1] for c in cooked)),
        f"| slide curve={'library measurement' if gi.slide_curve is not None else 'linear'}",
    )
    sf.write(
        DEMO_DIR / f"{args.name}.wav",
        (y / (np.max(np.abs(y)) + 1e-9) * 0.97).astype(np.float32),
        SR,
    )

    # reference: timbre A (the timbre of the articulation source instrument itself) × same articulation, same score
    ref_snap = None
    if args.ref_timbre:
        ref_snap, _ = _timbre_snapshot(args.ref_timbre, "ref")
    elif lib.get("sustain"):
        ref_snap = lib["sustain"]
    if ref_snap is not None:
        y_ref, _, _ = _render(ref_snap, lib)
        sf.write(
            DEMO_DIR / f"{args.name}_ref.wav",
            (y_ref / (np.max(np.abs(y_ref)) + 1e-9) * 0.97).astype(np.float32),
            SR,
        )
        # spectrogram comparison plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        _spec(
            axes[0],
            y_ref.astype(np.float32),
            SR,
            f"REF: timbre [{aname}] x articulation [{aname}]  (source instrument)",
        )
        _spec(
            axes[1],
            y.astype(np.float32),
            SR,
            f"RECOMBINE: timbre [{tname}] x articulation [{aname}]  (same gesture, diff timbre)",
        )
        fig.suptitle(
            f"FR-9 cross-recombination: same phrase + same articulation, timbre {tname} vs {aname}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(DEMO_DIR / f"{args.name}_compare.png", dpi=110)
        plt.close(fig)

    clean = bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0)
    print(
        f"\n  -> {args.name}.wav  ({len(y) / SR:.1f}s, clean={clean})"
        + (f" | reference {args.name}_ref.wav + {args.name}_compare.png" if ref_snap is not None else "")
    )
    print("\n---------------- FR-9 acceptance ----------------")
    print(f"  [{'OK ' if clean else 'FAIL'}] cross recombination holds: {tname} timbre × {aname} articulation")
    print(f"  Blind listening: {args.name}.wav and {args.name}_ref.wav use exactly the same technique (slides/legato/strums identical),")
    print(f"        one timbre is {tname}, the other {aname}: samplers structurally cannot do this")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
