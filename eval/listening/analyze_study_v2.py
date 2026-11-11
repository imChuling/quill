"""听测 v2 统计分析 —— 严格按 prereg_v2.md 执行。

输入: raw_codes_v2.txt(答案码,一行一人)+ study_v2_manifest.json
输出: listening_stats_v2.json + 终端对账表(论文数字以此为准)

预注册规则(prereg_v2.md, 2026-08-09 定稿):
- M1/M2/M4/M6 单侧二项(方向: M1→v0, M2→labeled, M4→diff, M6→diff);M3 双侧。
- M5 配对 Wilcoxon(Quill v0 = 刺激0 vs MIDI-DDSP = 刺激7)。
- ABX: unsure 剔除出正确率但报告数量;X1–X6 pooled 单侧二项;
  逐题二项 + Bonferroni(0.05/8);per-participant 准确率 + Wilcoxon vs 0.5。
  X7/X8 单独报告,不并入 X1–X6 pooled。
- 重复编号取第一份;无法解码整人剔除并报告。

跑: python3 eval/listening/analyze_study_v2.py
"""

import json
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "study_v2_manifest.json").read_text())
ABX_KEY = {t["qid"]: t for t in MANIFEST["abx"]}

# 预注册的方向假设: qid -> (正确/假设方向的取值, 备择方向)
MAIN_HYPOTH = {
    "M1": ("0", "greater"),  # v0 检索行为(刺激0)
    "M2": ("2", "greater"),  # labeled(刺激2)
    "M3": ("4", "two-sided"),  # v2 参数化(刺激4),探索、双侧
    "M4": ("diff", "greater"),
    "M6": ("diff", "greater"),
}
ABX_MAIN = ["X1", "X2", "X3", "X4", "X5", "X6"]  # 跨乐器主张
ABX_GLASS = ["X7", "X8"]  # 单独报告
BONFERRONI_M = 8  # 0.05/8, 预注册


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
    print(f"共 {n_lines} 行 -> N={n}(去重后),无法解码 {bad}")

    # ---- 主听测二元题 ----
    out["main"] = {}
    for qid, (target, alt) in MAIN_HYPOTH.items():
        votes = [d[qid] for d in rows if qid in d]
        k = sum(1 for v in votes if v == target)
        p = binom(k, len(votes), alt)
        out["main"][qid] = {"k": k, "n": len(votes), "alternative": alt, "p": p}
        print(f"  {qid}: {k}/{len(votes)}  ({alt}) p={p:.6f}")

    # ---- M5 配对评分 ----
    q = [int(d["M5r0"]) for d in rows if "M5r0" in d and "M5r7" in d]
    m = [int(d["M5r7"]) for d in rows if "M5r0" in d and "M5r7" in d]
    if q:
        try:
            w = stats.wilcoxon(q, m, alternative="two-sided", method="exact")
            p_w = float(w.pvalue)
        except ValueError:  # 全部差值为0时
            p_w = 1.0
        out["main"]["M5"] = {
            "n": len(q),
            "quill_median": float(sorted(q)[len(q) // 2]),
            "mididdsp_median": float(sorted(m)[len(m) // 2]),
            "wilcoxon_p": p_w,
        }
        print(
            f"  M5: Quill 中位 {out['main']['M5']['quill_median']:.0f} vs "
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
            f"  per-participant(X1-X6): 中位准确率 "
            f"{out['abx']['per_participant']['median_acc']:.1%}  "
            f"Wilcoxon vs 0.5 p={p_pp:.6f}"
        )

    (HERE / "listening_stats_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"已写 {HERE / 'listening_stats_v2.json'}")


if __name__ == "__main__":
    main()
