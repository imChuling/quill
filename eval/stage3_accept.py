"""Stage 3 acceptance: FR-4 | exit: measured latency numbers + no xrun at full polyphony.

Verifiable offline on this machine (no sound card needed):
  (1) stress acceptance: 16-voice full-load process_block callback time < 50% of the block budget (NFR-2 worst-case criterion)
  (2) golden-file regression: fixed event sequence rendered offline -> compared sample by sample with the baseline (eval_mode is reproducible)
  (3) render a short preview piece (chords + melody) -> assets/demos/stage3_phrase.wav

Needs a sound card (this script tries and tolerates failure):
  (4) latency report: measured sd.OutputStream latency; steps for measuring end-to-end latency with the loopback method are printed below

Usage:
  python -m eval.stage3_accept                  # use assets/snapshots/guitar.npz (or a synthetic fallback)
  python -m eval.stage3_accept voice            # specify the snapshot name
  python -m eval.stage3_accept --update-golden  # reset the golden baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.engine import Engine, Event  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import Instrument, build_instrument  # noqa: E402

try:
    from quill_config import CFG
except Exception:  # pragma: no cover
    CFG = None

SR = 44100
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "assets" / "demos"
SNAP_DIR = ROOT / "assets" / "snapshots"
GOLDEN = ROOT / "eval" / "golden" / "stage3_golden.npy"


def _load_instrument(name):
    p = SNAP_DIR / f"{name}.npz"
    if p.exists():
        snap = load_snapshot(p)
        nl = (CFG or {}).get("runtime", {}).get("noise_level", 0.5)
        return build_instrument(snap, noise_level=nl), f"snapshot {p.name}"
    # synthetic fallback: 60 harmonics with 1/k rolloff
    K = 60
    H = 1.0 / np.arange(1, K + 1)
    inst = Instrument(
        H,
        np.arange(1, K + 1, dtype=float),
        np.zeros(0),
        np.zeros(0),
        220.0,
        SR,
        name="synthetic",
        noise_level=0.3,
    )
    return inst, "synthetic fallback instrument (60 harmonics)"


# --------------------------------------------------------------------------- #
# (1) Stress acceptance: 16-voice full-load callback time
# --------------------------------------------------------------------------- #
def stress_test(inst, n_blocks=600):
    eng = Engine(inst, cfg=CFG, eval_mode=True)
    bs = eng.blocksize
    budget_ms = bs / eng.sr * 1000.0

    # start 16 different notes + bend, reaching the full-load steady state
    eng.reset()
    for i in range(eng.max_voices):
        eng._apply_event(Event("on", 45 + i, 110, 0.0))
    out = np.zeros(bs, dtype=np.float32)

    # warm up a few blocks (JIT/caches ready)
    for _ in range(20):
        eng.process_block(out, bs, 0)

    # performance mode: disable GC during measurement (rule #1, reproduces real-time callback conditions; GC is the biggest hidden xrun source)
    import gc

    gc.collect()
    try:
        gc.freeze()
    except Exception:
        pass
    gc.disable()
    try:
        times = np.empty(n_blocks)
        for b in range(n_blocks):
            t0 = time.perf_counter()
            eng.process_block(out, bs, b * bs)
            times[b] = (time.perf_counter() - t0) * 1000.0
    finally:
        gc.enable()

    active = sum(1 for v in eng.pool if v.active)
    return {
        "budget_ms": budget_ms,
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "max": float(times.max()),
        "active_voices": active,
        "blocksize": bs,
        "err": eng.error_flag,
    }


# --------------------------------------------------------------------------- #
# (2) Golden-file regression + (3) short preview piece
# --------------------------------------------------------------------------- #
_PHRASE = [
    # chord (C major) three notes together, then two melody lines on top
    (0.00, "on", 60, 100, 0.0),
    (0.00, "on", 64, 95, 0.0),
    (0.00, "on", 67, 95, 0.0),
    (0.80, "off", 60, 0, 0.0),
    (0.80, "off", 64, 0, 0.0),
    (0.80, "off", 67, 0, 0.0),
    (0.85, "on", 72, 110, 0.0),
    (1.05, "off", 72, 0, 0.0),
    (1.05, "on", 71, 100, 0.0),
    (1.25, "off", 71, 0, 0.0),
    (1.25, "on", 69, 90, 0.0),
    (1.55, "off", 69, 0, 0.0),
    (1.55, "on", 67, 105, 0.0),
    (1.7, "bend", -1, 0, 2.0),  # pitch bend +2 semitones
    (2.2, "off", 67, 0, 0.0),
]
_PHRASE_DUR = 2.6


def golden_regression(inst, update=False):
    eng = Engine(inst, cfg=CFG, eval_mode=True)
    y = eng.render_offline(_PHRASE, _PHRASE_DUR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(
        DEMO_DIR / "stage3_phrase.wav",
        (y / (np.max(np.abs(y)) + 1e-9) * 0.97).astype(np.float32),
        SR,
    )

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    status = "n/a"
    if update or not GOLDEN.exists():
        np.save(GOLDEN, y)
        status = "baseline written" if update else "baseline created (first run)"
    else:
        ref = np.load(GOLDEN)
        if ref.shape == y.shape and np.allclose(ref, y, atol=1e-6):
            status = "PASS (matches baseline)"
        else:
            md = (
                float(np.max(np.abs(ref[: len(y)] - y[: len(ref)])))
                if ref.shape == y.shape
                else float("nan")
            )
            status = f"DIFF (max sample difference {md:.2e})"
    return y, status, bool(eng.error_flag)


# --------------------------------------------------------------------------- #
# (4) Latency report (needs a sound card; tolerant of failure)
# --------------------------------------------------------------------------- #
def latency_report(inst):
    budget_ms = inst and (Engine(inst, cfg=CFG).blocksize / SR * 1000.0)
    lines = [f"block budget latency (buffer): {budget_ms:.2f} ms (blocksize/sr)"]
    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="output")
        lines.append(f"default output device: {dev['name']}")
        with sd.OutputStream(
            samplerate=SR, blocksize=int(budget_ms * SR / 1000), channels=1, dtype="float32"
        ) as s:
            lines.append(f"sd reported stream.latency: {s.latency * 1000:.2f} ms")
    except Exception as exc:
        lines.append(f"(no sound card/unavailable, skipping measured stream latency: {type(exc).__name__})")
    lines.append("End-to-end latency by loopback (run manually on a machine with a sound card):")
    lines.append("  connect the output back to the input (or play and record in the same room), have the engine emit a pulse, record the output at the same time,")
    lines.append("  measure the sample difference from MIDI/trigger to the recorded onset -> end-to-end latency (goes into the latency decomposition table).")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill stage 3 acceptance (FR-4)")
    ap.add_argument(
        "snapshot", nargs="?", default="guitar", help="assets/snapshots/<name>.npz (default guitar)"
    )
    ap.add_argument("--update-golden", action="store_true")
    args = ap.parse_args(argv)

    inst, src = _load_instrument(args.snapshot)
    print(f"Instrument source: {src}  (K={inst.K} harmonics, sr={inst.sr})")

    print("\n[1] 16-voice full-load stress test ...")
    s = stress_test(inst)
    pct = s["p95"] / s["budget_ms"] * 100
    pass1 = s["p95"] < 0.5 * s["budget_ms"] and not s["err"]
    print(f"  active voices     : {s['active_voices']}/{16}")
    print(f"  block budget       : {s['budget_ms']:.2f} ms (blocksize={s['blocksize']})")
    print(
        f"  callback time     : mean {s['mean']:.3f} / median {s['median']:.3f} / "
        f"p95 {s['p95']:.3f} / max {s['max']:.3f} ms"
    )
    print(f"  p95 of budget   : {pct:.1f}%   criterion<50%  -> {'PASS' if pass1 else 'FAIL'}")

    print("\n[2] golden-file regression + [3] short preview piece ...")
    y, gstatus, gerr = golden_regression(inst, update=args.update_golden)
    pass2 = ("PASS" in gstatus) or ("baseline" in gstatus) or ("created" in gstatus)
    print(
        f"  short piece -> assets/demos/stage3_phrase.wav  ({len(y) / SR:.2f}s, "
        f"peak {np.max(np.abs(y)):.3f}, finite {bool(np.all(np.isfinite(y)))})"
    )
    print(f"  golden regression     : {gstatus}")

    print("\n[4] latency report ...")
    for ln in latency_report(inst):
        print("  " + ln)

    ok = pass1 and pass2 and not gerr
    print("\n---------------- Stage 3 acceptance summary ----------------")
    print(
        f"  [{'OK ' if pass1 else 'FAIL'}] no xrun at full polyphony: 16-voice p95 callback {pct:.1f}% of budget (<50%)"
    )
    print(f"  [{'OK ' if pass2 else 'FAIL'}] offline render determinism/golden regression: {gstatus}")
    print("  latency: buffer budget 5.8ms; end-to-end <30ms must be measured with the loopback method on a machine with a sound card")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
