"""P1 A/B: expression layer (spectral trajectory + brightness following level) on vs off, same phrase rendered, for ear testing.

The phrase targets the hardest case for the "organ effect": a long sustained single note (6s) + a held triad (4s).
Outputs assets/demos/p1_{tone}_{off,on}.wav + objective metrics (spectral flux of the sustain, on should be clearly higher).

Usage: python -m eval.p1_ab
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from synth.voice import set_liveness  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "assets" / "demos"
SR = 44100
DUR = 11.5

TONES = {"voice": "voice.npz", "pad": "synth_pad.npz", "wineglass": "wineglass.npz"}

# long single note (6s) + held triad (4s): the phrase where a static spectrum has nowhere to hide
EVENTS = [
    (0.2, "on", 57, 96, 0.0, -1),
    (6.2, "off", 57, 0, 0.0, -1),
    (6.6, "on", 53, 92, 0.0, -1),
    (6.62, "on", 57, 88, 0.0, -1),
    (6.64, "on", 60, 90, 0.0, -1),
    (10.8, "off", 53, 0, 0.0, -1),
    (10.8, "off", 57, 0, 0.0, -1),
    (10.8, "off", 60, 0, 0.0, -1),
]


def octave_shift(f0_ref):
    midi_ref = 69 + 12 * np.log2(max(float(f0_ref), 1.0) / 440.0)
    return int(round((midi_ref - 57.0) / 12.0)) * 12


def sustain_motion(y, a, b):
    """Spectral motion of the sustain: (spectral flux, spectral centroid std Hz). Organ = both low; alive = at least one clearly present."""
    seg = y[int(a * SR) : int(b * SR)]
    n_fft, hop = 2048, 512
    frames = [
        np.abs(np.fft.rfft(seg[i : i + n_fft] * np.hanning(n_fft)))
        for i in range(0, len(seg) - n_fft, hop)
    ]
    S = np.array(frames)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    cent = (S * freqs).sum(axis=1) / (S.sum(axis=1) + 1e-12)
    Sn = S / (S.max() + 1e-12)
    return float(np.mean(np.abs(np.diff(Sn, axis=0))) * 1e3), float(np.std(cent))


def main():
    for name, f in TONES.items():
        snap = load_snapshot(str(ROOT / "assets" / "snapshots" / f))
        sh = octave_shift(snap["f0_ref"])
        evs = [(t, k, (n + sh if n >= 0 else n), v, val, n2) for (t, k, n, v, val, n2) in EVENTS]
        row, ys = {}, {}
        for mode, (spec, dark, br) in {
            "off": (0.0, 0.0, 0.0),
            "on": (0.85, 0.0, 0.8),
            "max": (1.8, 0.0, 1.6),
        }.items():  # max = exaggerated teaching level: for learning to hear the "breathing"
            inst = build_instrument(snap, noise_level=0.2)
            eng = Engine(inst, eval_mode=True)
            set_liveness(spec, dark, br)  # must come after Engine (engine init resets it from config)
            y = eng.render_offline(evs, DUR)
            sf.write(str(DEMO / f"p1_{name}_{mode}.wav"), y, SR)
            ys[mode] = y
            row[mode] = sustain_motion(y, 2.0, 5.8)
        # true click criterion = **kurtosis** of the residual derivative (normal audio ~3-10; isolated clicks >30).
        # Note: pulse counting fails at large modulation depth (the residual itself becomes an audio-rate signal); the 7.2 investigation confirmed false positives.
        kur = {}
        for m in ("on", "max"):
            dr = np.diff(ys[m] - ys["off"])
            m2 = (dr**2).mean() + 1e-30
            kur[m] = float((dr**4).mean() / m2**2 - 3.0)
        F = inst.spec_traj.shape[1] if inst.spec_traj is not None else 0
        (_, c0), (_, c1), (_, c2) = row["off"], row["on"], row["max"]
        amp_dev = float(np.abs(inst.amp_traj - 1).mean()) if inst.amp_traj is not None else 0.0
        print(
            f"[{name:10s}] F={F:3d} loudness breathing depth={amp_dev:.3f}  centroid std {c0:5.1f}->{c1:5.1f}->{c2:5.1f} Hz  "
            f"residual kurtosis on={kur['on']:.1f} max={kur['max']:.1f} -> {'clean' if kur['on'] < 30 else 'clicks!'}"
        )
    set_liveness(0.85, 0.0, 0.8)  # restore defaults
    print("\nEar test: assets/demos/p1_{voice,pad,wineglass}_{off,on}.wav (off first, then on)")


if __name__ == "__main__":
    main()
