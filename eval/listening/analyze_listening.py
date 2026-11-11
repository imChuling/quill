"""听测数据分析:从匿名化 CSV 重算论文中的全部人类实验数字。

输入(同目录):
  responses_main.csv  列: participant_id, musical_training(0/1),
                      q_expressive(labeled/plain), q_style_distinct(yes/no)
  responses_abx.csv   列: participant_id, trial_id(1-6), response(match/nonmatch/unsure)

跑: conda run -n mt python eval/listening/analyze_listening.py
产出: eval/listening/listening_stats.json + 终端对账表

注意:2026-08 的两次听测逐人数据未留存(见 surviving_records.md),
本脚本用于**未来重测**——表单逐人导出后即插即用,可产出逐题 Bonferroni
与 per-participant Wilcoxon 等旧数据无法支撑的统计。
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent


def main():
    # ---- 主听测 ----
    with open(HERE / "responses_main.csv") as fh:
        main_rows = list(csv.DictReader(fh))
    n = len(main_rows)
    pref = sum(1 for r in main_rows if r["q_expressive"].strip() == "labeled")
    distinct = sum(1 for r in main_rows if r["q_style_distinct"].strip() == "yes")
    p_pref = float(stats.binomtest(pref, n, 0.5, alternative="greater").pvalue)
    p_dist = float(stats.binomtest(distinct, n, 0.5, alternative="greater").pvalue)
    trained = sum(1 for r in main_rows if r["musical_training"].strip() == "1")
    print(f"主听测 N={n}(受训 {trained})")
    print(f"  偏好 labeled: {pref}/{n} ({pref / n:.0%})  one-sided binomial p={p_pref:.4f}")
    print(f"  风格可辨:    {distinct}/{n}  p={p_dist:.5f}")

    # ---- ABX ----
    with open(HERE / "responses_abx.csv") as fh:
        abx_rows = list(csv.DictReader(fh))
    valid = [r for r in abx_rows if r["response"].strip() != "unsure"]
    excluded = len(abx_rows) - len(valid)
    correct = sum(1 for r in valid if r["response"].strip() == "match")
    p_pool = float(stats.binomtest(correct, len(valid), 0.5, alternative="greater").pvalue)
    print(f"\nABX 总回答 {len(abx_rows)},剔除 unsure {excluded},有效 {len(valid)}")
    print(f"  pooled: {correct}/{len(valid)} ({correct / len(valid):.1%})  p={p_pool:.2e}")

    # 逐题 + Bonferroni
    trials = sorted({r["trial_id"] for r in abx_rows})
    per_trial = {}
    thresh = 0.05 / len(trials)
    all_sig = True
    for t in trials:
        tv = [r for r in valid if r["trial_id"] == t]
        tc = sum(1 for r in tv if r["response"].strip() == "match")
        pt = float(stats.binomtest(tc, len(tv), 0.5, alternative="greater").pvalue)
        sig = pt < thresh
        all_sig &= sig
        per_trial[t] = {"correct": tc, "n": len(tv), "p": round(pt, 5), "bonferroni_sig": bool(sig)}
        print(f"  trial {t}: {tc}/{len(tv)}  p={pt:.4f}  {'✓' if sig else '✗'} (阈值 {thresh:.4f})")

    # per-participant + Wilcoxon
    pids = sorted({r["participant_id"] for r in abx_rows})
    accs = []
    for p in pids:
        pv = [r for r in valid if r["participant_id"] == p]
        if pv:
            accs.append(sum(1 for r in pv if r["response"].strip() == "match") / len(pv))
    accs = np.array(accs)
    w = stats.wilcoxon(accs - 0.5, alternative="greater")
    print(f"  per-participant 平均 {accs.mean():.1%}  Wilcoxon p={float(w.pvalue):.2e}")

    out = {
        "main": {
            "n": n,
            "musically_trained": trained,
            "prefer_labeled": pref,
            "p_one_sided": round(p_pref, 4),
            "style_distinct": distinct,
            "p_distinct": round(p_dist, 5),
        },
        "abx": {
            "total": len(abx_rows),
            "excluded_unsure": excluded,
            "valid": len(valid),
            "correct": correct,
            "pooled_acc": round(correct / len(valid), 3),
            "p_pooled": float(p_pool),
            "per_trial": per_trial,
            "all_trials_bonferroni_sig": bool(all_sig),
            "per_participant_mean_acc": round(float(accs.mean()), 3),
            "wilcoxon_p": float(w.pvalue),
        },
    }
    with open(HERE / "listening_stats.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已存 {HERE / 'listening_stats.json'}")


if __name__ == "__main__":
    main()
