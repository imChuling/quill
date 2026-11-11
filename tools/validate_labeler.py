"""Labeler validation: MIDI-DDSP controlled rendering -> per-note ground truth vs classify_articulation output.

Data is generated on NBE_1 by export_labeler_validation.py (each piece × vibrato ∈ {0.0,0.3,0.6}
re-rendered in full, synthesis noise seeds matched across levels), synced to --data-dir and run locally:

    conda run -n mt python tools/validate_labeler.py --data-dir <dir>

The scoring path is identical to production (the per-segment calls of extract_style.extract):
  event_trajectories(y, sr, t_on, t_off) → classify_articulation(pitch_traj, conf)
note boundaries use the ground truth (isolating the labeler from the segmenter); the segmenter is tested separately.

Ground-truth mapping: vibrato=0.0 -> expect plain; vibrato=0.6 -> expect vibrato.
The 0.3 level is a sensitivity reference (the vibrato ceiling of natural renders is ~0.32), reported separately and not counted in the main metric.

Manipulation check: the 3.5–9Hz modulation band energy of the rendered f0 must rise monotonically with the vibrato
parameter, otherwise that piece's control is invalid (the teacher does not obey the control), and it is excluded from the main metric and reported.

Outputs: checkpoints/eval/labeler_validation.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

QUILL = Path(__file__).resolve().parent.parent
ART_NAMES = {0: "plain", 1: "vibrato", 2: "slide"}
FRAME_RATE = 250.0
MIN_DUR_FULL = 0.25  # notes shorter than one vibrato cycle do not count toward the vibrato recall main metric


def mod_band_energy(f0_hz, fps=FRAME_RATE, lo=3.5, hi=9.0):
    """f0 (Hz frame sequence) -> 3.5–9Hz band energy after cents detrending (for the manipulation check, same criterion as the paper)."""
    hz = np.asarray(f0_hz, np.float64)
    hz = hz[hz > 20]
    if len(hz) < int(fps * 0.3):
        return np.nan
    cents = 1200.0 * np.log2(hz / np.median(hz))
    cents = cents - np.polyval(np.polyfit(np.arange(len(cents)), cents, 1), np.arange(len(cents)))
    win = np.hanning(len(cents))
    spec = np.abs(np.fft.rfft(cents * win)) ** 2
    freqs = np.fft.rfftfreq(len(cents), 1.0 / fps)
    band = (freqs >= lo) & (freqs <= hi)
    return float(spec[band].sum() / max(len(cents), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=str(QUILL / "checkpoints/eval/labeler_validation.json"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    import soundfile as sf

    from analysis.tplib import event_trajectories
    from tools.extract_style import classify_articulation

    manifest = json.loads((data_dir / "manifest.json").read_text())
    print(f"{len(manifest)} renders in manifest")

    # ---- Manipulation check: per-note paired f0 modulation band energy (note: the whole piece cannot be used:
    # the note change rate at 130-200BPM itself falls in the 3.5-9Hz band and would swamp the vibrato signal) ----
    mel_level_energy = {}
    gt_notes_by_mel = {}
    for rec in manifest:
        tag = f"vib{rec['level']:.2f}".replace(".", "p")
        f0 = np.load(data_dir / f"{rec['name']}__{tag}_f0.npz")["f0_hz"]
        gt = json.loads((data_dir / f"{rec['name']}__{tag}_gt.json").read_text())
        gt_notes_by_mel.setdefault(rec["name"], gt["notes"])
        energies = []
        for note in gt["notes"]:
            if note["t_off"] - note["t_on"] < MIN_DUR_FULL:
                continue
            a = int(note["t_on"] * FRAME_RATE) + 10
            b = int(note["t_off"] * FRAME_RATE) - 5
            if b - a < int(FRAME_RATE * 0.2):
                continue
            energies.append(mod_band_energy(f0[a:b]))
        mel_level_energy.setdefault(rec["name"], {})[rec["level"]] = (
            float(np.nanmedian(energies)) if energies else np.nan
        )
    valid_melodies = set()
    print("\nManipulation check (median per-note 3.5-9Hz band energy of long notes, must rise monotonically with the level):")
    for name, by_level in sorted(mel_level_energy.items()):
        levels = sorted(by_level)
        vals = [by_level[l] for l in levels]
        ok = all(np.isfinite(v) for v in vals) and all(
            vals[i] < vals[i + 1] for i in range(len(vals) - 1)
        )
        print(
            f"  {name:28s} "
            + "  ".join(f"{l}:{v:10.1f}" for l, v in zip(levels, vals))
            + ("  OK" if ok else "  FAIL (excluded)")
        )
        if ok:
            valid_melodies.add(name)

    # ---- Per-note labelling ----
    rows = []
    for rec in manifest:
        tag = f"vib{rec['level']:.2f}".replace(".", "p")
        gt = json.loads((data_dir / f"{rec['name']}__{tag}_gt.json").read_text())
        y, sr = sf.read(str(data_dir / rec["wav"]), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        for note in gt["notes"]:
            t_on, t_off = note["t_on"], note["t_off"]
            if t_off - t_on < 0.10 or t_off > len(y) / sr:
                continue
            traj = event_trajectories(y, sr, t_on, t_off)
            pitch = traj["pitch"]
            pts = np.asarray(pitch["pts"])
            pred = classify_articulation(pts, float(pitch["confidence"]))
            rng = float(pts.max() - pts.min())
            d = np.diff(pts)
            sc = int(np.sum(np.abs(np.diff(np.sign(d[np.abs(d) > 1e-3]))) > 0))
            rows.append(
                {
                    "melody": rec["name"],
                    "level": rec["level"],
                    "t_on": t_on,
                    "dur": t_off - t_on,
                    "pred": pred,
                    "pred_name": ART_NAMES[pred],
                    "pitch_conf": float(pitch["confidence"]),
                    "rng": round(rng, 1),
                    "sign_changes": sc,
                }
            )
        print(f"  labeled {rec['name']} {tag}: {len(gt['notes'])} notes")

    # ---- Summary ----
    def subset(level, min_dur=0.0, valid_only=True):
        return [
            r
            for r in rows
            if r["level"] == level
            and r["dur"] >= min_dur
            and (not valid_only or r["melody"] in valid_melodies)
        ]

    def dist(rs):
        n = max(len(rs), 1)
        return {name: sum(1 for r in rs if r["pred"] == k) / n for k, name in ART_NAMES.items()} | {
            "n": len(rs)
        }

    report = {
        "schema": "quill.labeler_validation.v1",
        "valid_melodies": sorted(valid_melodies),
        "excluded_melodies": sorted(set(mel_level_energy) - valid_melodies),
        "manipulation_check": {
            name: {str(l): v for l, v in by.items()} for name, by in mel_level_energy.items()
        },
        "results": {},
    }

    print("\n=== Main metric (pieces that passed the manipulation check) ===")
    for level, expect in [(0.0, "plain"), (0.6, "vibrato")]:
        for min_dur, tag in [(MIN_DUR_FULL, f"dur>={MIN_DUR_FULL}s"), (0.0, "all")]:
            rs = subset(level, min_dur)
            d = dist(rs)
            acc = d.get(expect, 0.0)
            key = f"level{level}_{tag}"
            report["results"][key] = d | {"expected": expect, "accuracy": acc}
            print(
                f"  vibrato={level} ({tag:12s}) n={d['n']:3d}  "
                f"plain {d['plain']:.2f}  vibrato {d['vibrato']:.2f}  "
                f"slide {d['slide']:.2f}   -> {expect} hit rate {acc:.2f}"
            )
    rs = subset(0.3, MIN_DUR_FULL)
    report["results"]["level0.3_sensitivity"] = dist(rs)
    d = dist(rs)
    print(
        f"  vibrato=0.3 (sensitivity reference) n={d['n']:3d}  "
        f"plain {d['plain']:.2f}  vibrato {d['vibrato']:.2f}  slide {d['slide']:.2f}"
    )

    # ---- Threshold-free analysis: feature separability (0.0 vs 0.6, long notes) ----
    # production rule = sign_changes>=4 AND rng>=30c; 30c is calibrated on URMP violin (±50-100c),
    # while the MIDI-DDSP 0.6 level is a gentle vibrato (±25c). Report feature AUC and the optimal threshold.
    pos = subset(0.6, MIN_DUR_FULL)
    neg = subset(0.0, MIN_DUR_FULL)
    if pos and neg:

        def auc_of(score_fn):
            scores = np.array([score_fn(r) for r in neg + pos])
            truth = np.array([0] * len(neg) + [1] * len(pos))
            order = np.argsort(scores)
            ranks = np.empty(len(scores))
            ranks[order] = np.arange(1, len(scores) + 1)
            P, N = truth.sum(), len(truth) - truth.sum()
            return float((ranks[truth == 1].sum() - P * (P + 1) / 2) / (P * N))

        auc_sc = auc_of(lambda r: r["sign_changes"])
        auc_rng = auc_of(lambda r: r["rng"])
        auc_gated = auc_of(lambda r: r["rng"] if r["sign_changes"] >= 4 else 0.0)
        best_acc, best_rng = 0.0, 0.0
        for thr in sorted({r["rng"] for r in pos + neg}):
            acc = float(
                np.mean(
                    [(r["rng"] >= thr and r["sign_changes"] >= 4) for r in pos]
                    + [not (r["rng"] >= thr and r["sign_changes"] >= 4) for r in neg]
                )
            )
            if acc > best_acc:
                best_acc, best_rng = acc, float(thr)
        report["results"]["feature_analysis"] = {
            "auc_sign_changes": auc_sc,
            "auc_range": auc_rng,
            "auc_rule_score": auc_gated,
            "best_rng_thr_cents": best_rng,
            "acc_at_best_thr": best_acc,
            "n_pos": len(pos),
            "n_neg": len(neg),
        }
        print(f"\n=== Feature separability (0.0 vs 0.6, dur>={MIN_DUR_FULL}s) ===")
        print(f"  AUC: sign_changes {auc_sc:.3f}  range {auc_rng:.3f}  rule score {auc_gated:.3f}")
        print(f"  optimal range threshold {best_rng:.0f}c (keeping the sc>=4 gate) -> accuracy {best_acc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["rows"] = rows
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
