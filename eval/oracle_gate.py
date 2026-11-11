"""Gate 0 · Oracle: the go/no-go prerequisite experiment for the Neural Timbre Atlas.

Sprint notebook §5.8 Gate 0: **if the DSP render of the real target-note snapshot does not exceed the preregistered
static-source improvement threshold, check/strengthen the renderer first and stop the Atlas.**

Question: what the Atlas aims to do is "given 1-3 reference notes of an unseen instrument, predict the parameter correction of that instrument at a new pitch/
velocity relative to the static snapshot". A **perfect** model of this kind can at best predict
"the target note's own snapshot parameters". So measure this upper bound first:

    static : snapshot(reference note A) -> rendered to f0_B        <- current L0 shipping path
    oracle : snapshot(**real target note B**) -> rendered to f0_B  <- upper bound of any parameter prediction model
    sampler: pitch_shift(A -> B)                      <- naive sampler baseline (known audible difference)

All three are compared with the **real recording of target note B** by steady-state log spectral distance (analysis uses the same chain as the product).

If oracle ≈ static: there is nothing to gain in parameter space; the bottleneck is the renderer (synthesiser expressiveness), not
parameter prediction -> training a model is wasted, strengthen the renderer first.
If oracle ≫ static: real headroom exists and the Atlas is worth Sprint 2.

--------------------------------------------------------------------------- #
Preregistered criteria (fixed on 2026-07-29, **before any results**; any change must be recorded in the claim ledger)
--------------------------------------------------------------------------- #
H1 absolute headroom : median (static-oracle)/static  ≥ 0.10   (at least 10% of spectral distance to gain)
H2 calibrated headroom : median (static-oracle)/(sampler-static) ≥ 0.50
                   : the gain is at least half of the static-vs-sampler gap that is "already shown to be audible".
                   This anchors the abstract spectral distance to a scale known to be perceptually meaningful.
H3 consistency        : fraction of (instrument, target note) pairs where oracle beats static ≥ 0.70

Decision:
  GO        = H1 ∧ H3 pass, and H2 passes        -> Atlas goes into Sprint 2
  WEAK      = H1 ∧ H3 pass, H2 fails           -> headroom exists but may not be audible;
                                               Sprint 2 shrinks to a 2-day spike, do ridge first (Gate 1)
  NO-GO     = H1 or H3 fails                   -> stop the Atlas, strengthen the renderer instead; L0 keeps shipping

Usage:
    python eval/oracle_gate.py --dataset tinysol --limit 6      # quick pilot
    python eval/oracle_gate.py --dataset tinysol                # full run
    python eval/oracle_gate.py --dataset guitarset --out results/oracle_gate_gset.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.harmonic import f0_scalar, track_f0  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from eval.fidelity import SR, _sampler_pitch_shift, spectral_distance  # noqa: E402
from synth.render import render  # noqa: E402

# preregistered thresholds (do not change after seeing results)
H1_MIN_REL = 0.10
H2_MIN_CALIB = 0.50
H3_MIN_WINRATE = 0.70


def oracle_rows(samples, name, sr=SR, quasi=False, max_semis=18.0):
    """One instrument: compute the static / oracle / sampler spectral distances for every non-reference note."""
    enriched = []
    for pitch, audio in samples:
        y = np.asarray(audio, np.float32)
        f0v = f0_scalar(track_f0(y, sr=sr)[0])
        if np.isfinite(f0v) and f0v > 0:
            enriched.append((f0v, pitch, y))
    enriched.sort(key=lambda s: s[0])
    if len(enriched) < 2:
        return []

    ref_i = len(enriched) // 2
    f0_ref_real, ref_pitch, ref_audio = enriched[ref_i]
    snap_s = analyze_timbre(ref_audio, sr=sr, name=name, quasi_harmonic=quasi)
    f0_ref = snap_s["f0_ref"]

    rows = []
    for i, (f0_tgt_real, pitch, real) in enumerate(enriched):
        if i == ref_i:
            continue
        semis = 12.0 * np.log2(f0_tgt_real / f0_ref_real)
        if abs(semis) > max_semis:
            continue
        n = len(real)

        # static: reference-note snapshot -> rendered to the target f0 (current shipping path)
        f0_tgt = f0_ref * (f0_tgt_real / f0_ref_real)
        y_static = render(snap_s, f0_tgt, length=n, mode="formant", add_attack=False)

        # oracle: the target note's **own** snapshot -> rendered at its own reference f0 (zero transposition = parameter upper bound)
        try:
            snap_o = analyze_timbre(real, sr=sr, name=f"{name}_oracle", quasi_harmonic=quasi)
            y_oracle = render(snap_o, snap_o["f0_ref"], length=n, mode="formant", add_attack=False)
        except Exception as exc:  # notes whose capture fails are skipped, not silently counted as good results
            rows.append(
                {
                    "pitch": int(pitch),
                    "semis": round(semis, 2),
                    "skipped": f"oracle_capture_failed:{type(exc).__name__}",
                }
            )
            continue

        y_samp = _sampler_pitch_shift(ref_audio, semis, sr)

        d_static = spectral_distance(y_static, real, sr)
        d_oracle = spectral_distance(y_oracle, real, sr)
        d_samp = spectral_distance(y_samp, real, sr)
        rows.append(
            {
                "pitch": int(pitch),
                "semis": round(semis, 2),
                "d_static": d_static,
                "d_oracle": d_oracle,
                "d_sampler": d_samp,
                "rel_headroom": (d_static - d_oracle) / d_static if d_static > 0 else 0.0,
                "calib_headroom": (
                    (d_static - d_oracle) / (d_samp - d_static)
                    if (d_samp - d_static) > 1e-9
                    else float("nan")
                ),
                "oracle_better": bool(d_oracle < d_static),
            }
        )
    return rows


def summarize(all_rows):
    ok = [r for r in all_rows if "d_static" in r]
    if not ok:
        return {"error": "no valid rows"}
    rel = np.array([r["rel_headroom"] for r in ok])
    win = np.array([r["oracle_better"] for r in ok], dtype=float)
    calib = np.array([r["calib_headroom"] for r in ok], dtype=float)
    calib_ok = calib[np.isfinite(calib)]

    h1 = float(np.median(rel))
    h2 = float(np.median(calib_ok)) if calib_ok.size else float("nan")
    h3 = float(win.mean())
    p1, p3 = h1 >= H1_MIN_REL, h3 >= H3_MIN_WINRATE
    p2 = bool(np.isfinite(h2) and h2 >= H2_MIN_CALIB)

    verdict = "GO" if (p1 and p3 and p2) else ("WEAK" if (p1 and p3) else "NO-GO")
    return {
        "n_pairs": len(ok),
        "n_skipped": len(all_rows) - len(ok),
        "H1_median_rel_headroom": h1,
        "H1_pass": bool(p1),
        "H1_min": H1_MIN_REL,
        "H2_median_calib_headroom": h2,
        "H2_pass": p2,
        "H2_min": H2_MIN_CALIB,
        "H3_oracle_winrate": h3,
        "H3_pass": bool(p3),
        "H3_min": H3_MIN_WINRATE,
        "median_d_static": float(np.median([r["d_static"] for r in ok])),
        "median_d_oracle": float(np.median([r["d_oracle"] for r in ok])),
        "median_d_sampler": float(np.median([r["d_sampler"] for r in ok])),
        "verdict": verdict,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tinysol", choices=["tinysol", "guitarset", "nsynth"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N instruments (pilot)")
    ap.add_argument("--max-pitch", type=int, default=7)
    ap.add_argument("--out", default="results/oracle_gate.json")
    a = ap.parse_args(argv)

    dsets = Path.home() / "Desktop" / "MT" / "datasets"
    if a.dataset == "tinysol":
        from eval.fidelity import tinysol_groups

        groups = tinysol_groups(a.root or dsets / "TinySOL", max_pitch=a.max_pitch)
    elif a.dataset == "nsynth":
        from eval.fidelity import nsynth_groups

        groups = nsynth_groups(a.root or dsets / "nsynth-test", max_pitch=a.max_pitch)
    else:
        from eval.fidelity import idmt_samples

        s = idmt_samples(a.root or dsets / "GuitarSet_mic")
        groups = [("GuitarSet/mic", s)] if s else []

    if not groups:
        print(f"[oracle-gate] dataset is empty: {a.dataset}")
        return 1
    if a.limit:
        groups = groups[: a.limit]

    print(
        f"[oracle-gate] {a.dataset}: {len(groups)} instruments "
        f"(preregistered H1≥{H1_MIN_REL} H2≥{H2_MIN_CALIB} H3≥{H3_MIN_WINRATE})"
    )
    t0 = time.time()
    all_rows, per_inst = [], []
    for name, samples in groups:
        rows = oracle_rows(samples, name)
        ok = [r for r in rows if "d_static" in r]
        if ok:
            m_rel = float(np.median([r["rel_headroom"] for r in ok]))
            wr = float(np.mean([r["oracle_better"] for r in ok]))
            print(f"  {name:42s} n={len(ok):2d} rel_headroom={m_rel:+.3f} winrate={wr:.2f}")
            per_inst.append(
                {
                    "instrument": name,
                    "n": len(ok),
                    "median_rel_headroom": m_rel,
                    "oracle_winrate": wr,
                }
            )
        else:
            print(f"  {name:42s} (no valid samples)")
        for r in rows:
            r["instrument"] = name
        all_rows += rows

    summ = summarize(all_rows)
    summ["dataset"] = a.dataset
    summ["elapsed_s"] = round(time.time() - t0, 1)
    summ["n_instruments"] = len(per_inst)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summ, "per_instrument": per_inst, "rows": all_rows}, indent=2)
    )

    print("\n" + "=" * 66)
    print(
        f"H1 absolute headroom  median (static-oracle)/static = {summ['H1_median_rel_headroom']:+.3f} "
        f"(≥{H1_MIN_REL}) {'PASS' if summ['H1_pass'] else 'FAIL'}"
    )
    print(
        f"H2 calibrated headroom  median /(sampler-static)      = {summ['H2_median_calib_headroom']:+.3f} "
        f"(≥{H2_MIN_CALIB}) {'PASS' if summ['H2_pass'] else 'FAIL'}"
    )
    print(
        f"H3 oracle win rate                              = {summ['H3_oracle_winrate']:.3f} "
        f"(≥{H3_MIN_WINRATE}) {'PASS' if summ['H3_pass'] else 'FAIL'}"
    )
    print(
        f"median spectral distance: static={summ['median_d_static']:.4f} "
        f"oracle={summ['median_d_oracle']:.4f} sampler={summ['median_d_sampler']:.4f}"
    )
    print(
        f"\n>>> GATE 0 decision: {summ['verdict']}   "
        f"(n={summ['n_pairs']} pairs, {summ['n_instruments']} instruments, {summ['elapsed_s']}s)"
    )
    print("=" * 66)
    print(f"artifact: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
