"""Listening test v2 statistical analysis, carried out strictly according to prereg_v2.md.

Input: raw_codes_v2.txt (answer codes, one line per participant) + study_v2_manifest.json
Output: listening_stats_v2.json + a reconciliation table in the terminal (the paper's numbers follow this)

Preregistered rules (prereg_v2.md, finalised 2026-08-09):
- M1/M2/M4/M6 one-sided binomial (direction: M1->v0, M2->labeled, M4->diff, M6->diff); M3 two-sided.
- M5 paired Wilcoxon (Quill v0 = stimulus 0 vs MIDI-DDSP = stimulus 7).
- ABX: unsure excluded from accuracy but the count is reported; X1–X6 pooled one-sided binomial;
  per-trial binomial + Bonferroni (0.05/8); per-participant accuracy + Wilcoxon vs 0.5.
  X7/X8 reported separately, not pooled with X1–X6.
- duplicate IDs: the first submission is kept; participants whose code cannot be decoded are excluded entirely and reported.

Run: python3 eval/listening/analyze_study_v2.py
"""

import json
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "study_v2_manifest.json").read_text())
ABX_KEY = {t["qid"]: t for t in MANIFEST["abx"]}

# preregistered directional hypotheses: qid -> (correct/hypothesised value, alternative)
MAIN_HYPOTH = {
    "M1": ("0", "greater"),  # v0 retrieved behaviour (stimulus 0)
    "M2": ("2", "greater"),  # labeled (stimulus 2)
    "M3": ("4", "two-sided"),  # v2 parametric (stimulus 4), exploratory, two-sided
    "M4": ("diff", "greater"),
    "M6": ("diff", "greater"),
}
ABX_MAIN = ["X1", "X2", "X3", "X4", "X5", "X6"]  # cross-instrument claim
ABX_GLASS = ["X7", "X8"]  # reported separately
BONFERRONI_M = 8  # 0.05/8, preregistered


def parse_codes():
    lines = [l for l in (HERE / "raw_codes_v2.txt").read_text().splitlines() if l.strip()]
    seen, rows, bad = set(), [], 0
    for l in lines:
        parts = l.strip().split("|")
        if not parts or parts[0] != "QV2":
            bad += 1
            continue
        d = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
        if "id" not in d or d["id"] in seen:
            if "id" not in d:
                bad += 1
            continue
        seen.add(d["id"])
        rows.append(d)
    return rows, len(lines), bad


def binom(k, n, alternative):
    return float(stats.binomtest(k, n, 0.5, alternative=alternative).pvalue)


def main():
    rows, n_lines, bad = parse_codes()
    n = len(rows)
    out = {"n_lines": n_lines, "n_participants": n, "n_undecodable": bad}
    print(f"total {n_lines} lines -> N={n} (after deduplication), undecodable {bad}")

    # ---- Main listening test binary questions ----
    out["main"] = {}
    for qid, (target, alt) in MAIN_HYPOTH.items():
        votes = [d[qid] for d in rows if qid in d]
        k = sum(1 for v in votes if v == target)
        p = binom(k, len(votes), alt)
        out["main"][qid] = {"k": k, "n": len(votes), "alternative": alt, "p": p}
        print(f"  {qid}: {k}/{len(votes)}  ({alt}) p={p:.6f}")

    # ---- M5 paired ratings ----
    q = [int(d["M5r0"]) for d in rows if "M5r0" in d and "M5r7" in d]
    m = [int(d["M5r7"]) for d in rows if "M5r0" in d and "M5r7" in d]
    if q:
        try:
            w = stats.wilcoxon(q, m, alternative="two-sided", method="exact")
            p_w = float(w.pvalue)
        except ValueError:  # when all differences are 0
            p_w = 1.0
        out["main"]["M5"] = {
            "n": len(q),
            "quill_median": float(sorted(q)[len(q) // 2]),
            "mididdsp_median": float(sorted(m)[len(m) // 2]),
            "wilcoxon_p": p_w,
        }
        print(
            f"  M5: Quill median {out['main']['M5']['quill_median']:.0f} vs "
            f"MIDI-DDSP {out['main']['M5']['mididdsp_median']:.0f}  "
            f"Wilcoxon p={p_w:.6f}"
        )

    # ---- ABX ----
    out["abx"] = {"per_item": {}, "n_unsure": 0}
    per_participant = []
    for d in rows:
        c, t = 0, 0
        for qid in ABX_MAIN:
            if qid not in d:
                continue
            if d[qid] == "u":
                out["abx"]["n_unsure"] += 1
                continue
            t += 1
            c += int(int(d[qid]) == ABX_KEY[qid]["match"])
        if t:
            per_participant.append(c / t)

    for qid in ABX_MAIN + ABX_GLASS:
        k = t = u = 0
        for d in rows:
            if qid not in d:
                continue
            if d[qid] == "u":
                u += 1
                continue
            t += 1
            k += int(int(d[qid]) == ABX_KEY[qid]["match"])
        p = binom(k, t, "greater") if t else 1.0
        sig = p < 0.05 / BONFERRONI_M
        out["abx"]["per_item"][qid] = {
            "correct": k,
            "n": t,
            "unsure": u,
            "p": p,
            "bonferroni_sig": sig,
        }
        print(f"  {qid}: {k}/{t} (unsure {u})  p={p:.6f}  {'*' if sig else ''}(Bonferroni 0.05/8)")

    k6 = sum(out["abx"]["per_item"][q]["correct"] for q in ABX_MAIN)
    t6 = sum(out["abx"]["per_item"][q]["n"] for q in ABX_MAIN)
    p6 = binom(k6, t6, "greater")
    out["abx"]["pooled_x1_x6"] = {"correct": k6, "n": t6, "p": p6}
    print(f"  X1-X6 pooled: {k6}/{t6} = {k6 / t6:.1%}  p={p6:.3g}")

    if per_participant and len(set(per_participant)) >= 1:
        try:
            w = stats.wilcoxon(
                [a - 0.5 for a in per_participant], alternative="greater", method="exact"
            )
            p_pp = float(w.pvalue)
        except ValueError:
            p_pp = 1.0
        out["abx"]["per_participant"] = {
            "n": len(per_participant),
            "median_acc": float(sorted(per_participant)[len(per_participant) // 2]),
            "wilcoxon_p": p_pp,
        }
        print(
            f"  per-participant (X1-X6): median accuracy "
            f"{out['abx']['per_participant']['median_acc']:.1%}  "
            f"Wilcoxon vs 0.5 p={p_pp:.6f}"
        )

    (HERE / "listening_stats_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"Wrote {HERE / 'listening_stats_v2.json'}")


if __name__ == "__main__":
    main()
