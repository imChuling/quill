"""Expressive timing evaluation: does the timing signature follow the behaviour source (2×2) + ear-test A/B audio.

The "immediate next step" promised in the paper's Conclusion. timing v1 = articulation ratio (sounding/IOI) retrieved
from the library + agogic (local IOI stretching), see neural_take.apply_timing.

Experiment design (mirrors the source-attribution protocol):
  behaviour ∈ {violin, guitar} × timbre ∈ {violin, guitar}, timing=1.0.
  For each render, extract the timing signature from the transformed note table:
    [median articulation ratio, IQR articulation ratio, detach fraction (gaps >80ms)]
  Check that the distance from the signature to "the matching behaviour source's library signature" < the distance to "the other library's signature" (ratio<1 = pass).
  The note table is the ground truth for audio timing: _phrases places phrases at absolute times,
  so gaps are real silence in the rendered audio.

Ear test (batched): checkpoints/eval_timing/*.wav
  {lyrical,folk} × viT × {viB,guB} × timing{on,off} = 8 clips.

Run: conda run -n mt python tools/eval_timing.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from neural.neural_take import NeuralTake, apply_timing
from tools.eval_quill import MELODIES

OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "eval_timing"
RESULTS = Path(__file__).resolve().parent.parent / "checkpoints" / "eval" / "timing_results.json"
DETACH_GAP = 0.08  # neural_take.PHRASE_GAP: a gap above this = a detached phrase boundary
ART_NAMES = {0: "", 1: "vibrato", 2: "slide"}


def melody_notes(score):
    """eval_quill's (midi,dur,art) sequence -> piano-roll note table (legato nominal timing)."""
    notes, t = [], 0.0
    for midi, dur, art in score:
        notes.append({"midi": midi, "start": t, "len": dur, "vel": 100, "art": ART_NAMES[art]})
        t += dur
    return notes


def timing_signature(notes):
    """Note table -> [median art-ratio, IQR art-ratio, detach fraction]."""
    seq = sorted(notes, key=lambda x: float(x["start"]))
    ratios, detached = [], []
    for i in range(len(seq) - 1):
        ioi = float(seq[i + 1]["start"]) - float(seq[i]["start"])
        if ioi <= 0.05:
            continue
        r = min(float(seq[i]["len"]) / ioi, 1.0)
        ratios.append(r)
        detached.append((ioi - float(seq[i]["len"])) > DETACH_GAP)
    ratios = np.array(ratios)
    return np.array(
        [
            float(np.median(ratios)),
            float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
            float(np.mean(detached)),
        ]
    )


def library_signature(nn):
    """The library's reference timing signature (the same triple, computed from the timing table)."""
    r = nn.timing[:, 0].astype(np.float64)
    return np.array(
        [
            float(np.median(r)),
            float(np.percentile(r, 75) - np.percentile(r, 25)),
            float(np.mean(r < (1.0 - 0.15))),
        ]
    )  # shortened by >15% counts as a staccato tendency


SWEEP_STRENGTHS = [0.5, 0.75, 1.0, 1.25]
SWEEP_PAIRS = [("lyrical", "violin"), ("folk", "guitar")]


def run_attribution(nt, behaviors, lib_sigs, strength=1.0):
    checks, details = [], {}
    for beh in behaviors:
        other = [b for b in behaviors if b != beh][0]
        for tim in behaviors:
            sigs = []
            for _mname, score in MELODIES.items():
                notes = melody_notes(score)
                shaped = apply_timing(notes, nt.voice(beh).nn, strength=strength)
                sigs.append(timing_signature(shaped))
            sig = np.mean(sigs, axis=0)
            scale = np.abs(lib_sigs[beh]) + np.abs(lib_sigs[other]) + 1e-6
            d_match = float(np.linalg.norm((sig - lib_sigs[beh]) / scale))
            d_other = float(np.linalg.norm((sig - lib_sigs[other]) / scale))
            ratio = d_match / max(d_other, 1e-9)
            ok = ratio < 1.0
            checks.append(ok)
            details[f"{beh[:2]}B_{tim[:2]}T"] = {
                "signature": np.round(sig, 3).tolist(),
                "d_match": round(d_match, 3),
                "d_other": round(d_other, 3),
                "ratio": round(ratio, 3),
                "pass": bool(ok),
            }
            print(
                f"  {beh[:2]}B_{tim[:2]}T: sig={np.round(sig, 3).tolist()}  "
                f"ratio={ratio:.3f}  {'PASS' if ok else 'FAIL'}"
            )
    print(f"timing attribution (strength={strength}): {sum(checks)}/{len(checks)}")
    return checks, details


def run_sweep(nt, behaviors):
    """strength sweep: render the SWEEP_PAIRS combinations for each strength value, for ear-test calibration."""
    sweep_dir = OUT_DIR / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    wavs = []
    for s in SWEEP_STRENGTHS:
        for mname, beh in SWEEP_PAIRS:
            notes = melody_notes(MELODIES[mname])
            y, sr = nt.render(notes, instrument="violin", behavior=beh, timing=s)
            name = f"{mname}_viT_{beh[:2]}B_s{s:.2f}.wav"
            sf.write(str(sweep_dir / name), y, sr)
            wavs.append(name)
            print(f"  sweep: {name}")
    print(f"strength sweep: {len(wavs)} clips -> {sweep_dir}")
    return wavs


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="strength sweep for ear-test calibration")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nt = NeuralTake(device="cpu")
    behaviors = ["violin", "guitar"]
    lib_sigs = {b: library_signature(nt.voice(b).nn) for b in behaviors}
    print("Library reference signatures [median ratio, IQR, detach-ish frac]:")
    for b, s in lib_sigs.items():
        print(f"  {b}: {np.round(s, 3).tolist()}")

    # ---- 2×2 timing attribution ---- #
    checks, details = run_attribution(nt, behaviors, lib_sigs, strength=1.0)

    # ---- Ear-test A/B audio (timing on/off × behaviour source) ---- #
    wavs = []
    for mname in ["lyrical", "folk"]:
        for beh in behaviors:
            for tstr, tval in [("toff", 0.0), ("ton", 1.0)]:
                notes = melody_notes(MELODIES[mname])
                y, sr = nt.render(notes, instrument="violin", behavior=beh, timing=tval)
                name = f"{mname}_viT_{beh[:2]}B_{tstr}.wav"
                sf.write(str(OUT_DIR / name), y, sr)
                wavs.append(name)
    print(f"ear-test audio {len(wavs)} clips -> {OUT_DIR}")

    # ---- strength sweep (generated additionally with --sweep) ---- #
    sweep_wavs = []
    sweep_attr = {}
    if args.sweep:
        sweep_wavs = run_sweep(nt, behaviors)
        for s in SWEEP_STRENGTHS:
            print(f"\n--- attribution at strength={s} ---")
            _, attr = run_attribution(nt, behaviors, lib_sigs, strength=s)
            sweep_attr[str(s)] = attr

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "library_signatures": {b: np.round(s, 3).tolist() for b, s in lib_sigs.items()},
        "attribution": details,
        "passed": f"{sum(checks)}/{len(checks)}",
        "ear_test_wavs": wavs,
    }
    if sweep_wavs:
        results["sweep"] = {
            "strengths": SWEEP_STRENGTHS,
            "wavs": sweep_wavs,
            "attribution_by_strength": sweep_attr,
        }
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"results -> {RESULTS}")


if __name__ == "__main__":
    main()
