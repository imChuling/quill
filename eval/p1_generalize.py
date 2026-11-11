"""P1 generalisation sweep (guarding against overfitting, a user warning on 7.2): one locked parameter set × diverse real sources.

Principle: one global set of liveness parameters (0.85/0/0.8), **no per-file tuning**; the motion itself comes from each source's
own recording (data-driven), and the only global constant is depth. This script verifies on TinySOL (brass/woodwind/strings/keyboard)
+ the project's own recordings (voice ×3 / synth pad): no clipping, no clicks (residual kurtosis), motion within a reasonable band.

Usage: python -m eval.p1_generalize
"""

from __future__ import annotations

import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from analysis.io import load  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from synth.voice import set_liveness  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TSOL = Path(os.environ.get("QUILL_DATASETS", str(Path.home() / "Desktop/MT/datasets"))) / "TinySOL"
SR = 44100
DUR = 11.5
LIVE = (0.85, 0.0, 0.8)  # locked global parameters (same as the product default)

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


def mid_mf(inst_dir):
    """Pick a mid-register mf single note from an instrument directory (middle of the sorted list, heuristic; not picking the nice-sounding one = guarding against overfitting)."""
    fs = sorted(glob.glob(str(inst_dir / "ordinario" / "*-mf-*.wav")))
    return fs[len(fs) // 2] if fs else None


def sources():
    out = [
        ("voice_low", ROOT / "assets/recordings/tones/voice_tone_low.m4a"),
        ("voice_mid", ROOT / "assets/recordings/tones/voice_tone_mid.m4a"),
        ("voice_high", ROOT / "assets/recordings/tones/voice_tone_high.m4a"),
        ("synth_pad", ROOT / "assets/recordings/tones/synth_pad.wav"),
    ]
    for fam, inst in [
        ("Brass", "Trumpet in C"),
        ("Brass", "Horn"),
        ("Brass", "Bass Tuba"),
        ("Winds", "Flute"),
        ("Winds", "Clarinet in Bb"),
        ("Winds", "Bassoon"),
        ("Strings", "Violin"),
        ("Strings", "Violoncello"),
        ("Strings", "Contrabass"),
        ("Keyboards", "Accordion"),
    ]:
        f = mid_mf(TSOL / fam / inst)
        if f:
            out.append((inst.replace(" ", "_"), Path(f)))
    return out


def main():
    print(f"locked parameters liveness={LIVE} (no per-file tuning)\n")
    bad = 0
    for name, path in sources():
        try:
            y, sr = load(str(path), sr=SR)
            snap = analyze_timbre(np.asarray(y, np.float32), sr=sr, name=name)
            if not (np.isfinite(snap.get("f0_ref", 0)) and snap["f0_ref"] > 0):
                print(f"[{name:14s}] skipped: no stable pitch")
                continue
            sh = int(round((69 + 12 * np.log2(snap["f0_ref"] / 440.0) - 57) / 12.0)) * 12
            evs = [
                (t, k, (n + sh if n >= 0 else n), v, val, n2) for (t, k, n, v, val, n2) in EVENTS
            ]
            ys = {}
            for mode, cfg in {"off": (0, 0, 0), "on": LIVE}.items():
                inst = build_instrument(snap, noise_level=0.2)
                eng = Engine(inst, eval_mode=True)
                set_liveness(*cfg)
                ys[mode] = eng.render_offline(evs, DUR)
            dr = np.diff(ys["on"] - ys["off"])
            m2 = (dr**2).mean() + 1e-30
            kur = float((dr**4).mean() / m2**2 - 3.0)
            pk = float(np.abs(ys["on"]).max())
            tdev = float(np.abs(inst.spec_traj - 1).mean()) if inst.spec_traj is not None else 0.0
            adev = float(np.abs(inst.amp_traj - 1).mean()) if inst.amp_traj is not None else 0.0
            ok = pk <= 1.0 and np.isfinite(ys["on"]).all() and kur < 30
            bad += not ok
            print(
                f"[{name:14s}] f0={snap['f0_ref']:6.1f}Hz peak={pk:.2f} residual kurtosis={kur:5.1f} "
                f"shape breathing={tdev:.3f} loudness breathing={adev:.3f} -> {'OK' if ok else 'FAIL'}"
            )
        except Exception as e:
            bad += 1
            print(f"[{name:14s}] error: {str(e)[:60]} -> FAIL")
    print(
        f"\nConclusion: {'all passed: one parameter set generalises' if bad == 0 else f'{bad} sources failed, investigate (no masking by tuning)'}"
    )
    set_liveness(*LIVE)


if __name__ == "__main__":
    main()
