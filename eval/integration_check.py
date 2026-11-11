"""Mid-project integration checkpoint (6.25): sustain-only end-to-end full data path.

Following the material capture protocol: single note -> snapshot -> real-time keyboard performance, verifying that the whole analysis -> assets -> runtime
path runs by 6.25 (the first full-chain integration is not left to 6.29). Interface problems are printed and recorded in the stage 6 integration checklist.

Offline local version (runs without a sound card/microphone): a synthetic steady note replaces the live recording, the rest of the path is identical:
  analyze_timbre -> save_snapshot(.npz) -> load_snapshot -> build_instrument -> Engine.render_offline

Usage:
  python -m eval.integration_check                 # run the path with a synthetic note
  python -m eval.integration_check --record 4      # record a 4s note live (needs a microphone)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.io import normalize  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot, save_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

SR = 44100
SNAP_DIR = Path(__file__).resolve().parents[1] / "assets" / "snapshots"


def _synth_tone(f0=196.0, dur=2.0, sr=SR):
    """Steady harmonic single note (replaces the live recording; the rest of the path is identical)."""
    t = np.arange(int(sr * dur)) / sr
    y = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 25))
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (dur - t) / 0.1)
    return normalize((y * env).astype(np.float32))


def _dominant_hz(y, sr):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(S)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill mid-project integration checkpoint (sustain-only)")
    ap.add_argument("--record", type=float, default=0.0, help="live recording length in seconds")
    args = ap.parse_args(argv)

    issues = []  # interface problem list -> stage 6 integration

    # (1) analysis: recording/synthesis -> snapshot
    if args.record > 0:
        from analysis.io import record

        print(f"Recording a {args.record}s note live ...")
        y = record(args.record, sr=SR)
    else:
        print("(offline) synthetic steady note replaces the live recording ...")
        y = _synth_tone()
    snap = analyze_timbre(y, sr=SR, name="integration_check")
    print(
        f"  analyze_timbre -> f0_ref={snap['f0_ref']:.1f}Hz, "
        f"H{snap['H_env'].shape}, N{snap['N_env'].shape}"
    )

    # (2) assets: snapshot serialisation round trip
    p = save_snapshot(SNAP_DIR / "integration_check.npz", snap)
    snap2 = load_snapshot(p)
    if not np.allclose(snap["H_env"], snap2["H_env"]):
        issues.append("snapshot H_env serialisation round trip mismatch")
    print(
        f"  save/load snapshot -> {p.name}  round trip identical={np.allclose(snap['H_env'], snap2['H_env'])}"
    )

    # (3) runtime: snapshot -> instrument -> real-time engine performance (offline rendering on the same process_block path)
    inst = build_instrument(snap2, noise_level=0.3)
    eng = Engine(inst, eval_mode=True)
    note = 64  # E4
    y_play = eng.render_offline([(0.1, "on", note, 100, 0.0), (0.9, "off", note, 0, 0.0)], 1.2)
    f0_target = 440.0 * 2 ** ((note - 69) / 12.0)
    got = _dominant_hz(y_play[int(0.2 * SR) : int(0.8 * SR)], SR)
    cents = 1200 * np.log2(got / f0_target)
    clean = bool(np.all(np.isfinite(y_play)) and np.max(np.abs(y_play)) <= 1.0)
    if abs(cents) > 30:
        issues.append(f"performance pitch deviation {cents:.0f} cents (note {note} expected {f0_target:.1f}Hz)")
    if not clean:
        issues.append("engine output non-finite/out of range")
    if eng.error_flag:
        issues.append("engine error_flag set")
    print(
        f"  engine plays note {note}: f0={got:.1f}Hz (expected {f0_target:.1f}, {cents:+.0f}cents), "
        f"clean={clean}"
    )

    print("\n---------------- Integration check summary ----------------")
    ok = not issues
    print(f"  analysis -> assets -> runtime full path: {'PASS' if ok else 'FAIL'}")
    if issues:
        print("  Interface problems (recorded in the stage 6 integration checklist):")
        for s in issues:
            print(f"   - {s}")
    else:
        print("  No interface problems; sustain-only end to end works (first full-chain integration complete)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
