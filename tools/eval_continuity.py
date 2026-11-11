"""序列连续性:相邻音符轨迹端点不连续度 + concatenation cost 改善。

Plan D 确认的真瓶颈:逐音符独立检索导致相邻行为不连贯。
本脚本量化基线(λ=0)和 concatenation cost reranking 后的改善。

指标 continuity cost(per note boundary):
  pitch_d:  |traj_i pitch 末 4 帧均值 − traj_{i+1} pitch 首 4 帧均值|   (cents)
  energy_d: |traj_i energy 末 4 帧均值 − traj_{i+1} energy 首 4 帧均值|  (dB-like)

--sweep: λ ∈ {0, 0.5, 1.0, 2.0} 全跑,输出改善比例
--eartest: 生成 A/B wav(λ=0 vs λ=best)

跑: conda run -n mt python tools/eval_continuity.py --sweep --eartest
产出: checkpoints/eval/continuity_results.json + checkpoints/eval_continuity/*.wav
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from neural.neural_take import NeuralTake
from tools.eval_quill import MELODIES

RESULTS = (
    Path(__file__).resolve().parent.parent / "checkpoints" / "eval" / "continuity_results.json"
)
WAV_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "eval_continuity"
EDGE = 4
LAMBDAS = [0.0, 0.5, 1.0, 2.0]


def _build_feat(note, i, n, seq_onsets, score):
    midi, dur = float(note["midi"]), note["dur"]
    prev_gap = max(0.0, seq_onsets[i] - (seq_onsets[i - 1] + score[i - 1]["dur"])) if i > 0 else 1.0
    next_gap = max(0.0, seq_onsets[i + 1] - (seq_onsets[i] + dur)) if i < n - 1 else 1.0
    base = [midi, dur, prev_gap, next_gap, 1.0 if i == 0 else 0.0, 1.0 if i == n - 1 else 0.0]
    base.append(float(note.get("art", 0)))
    return np.array(base, np.float32)


def continuity_costs(score, nn, concat_lambda=0.0):
    """score [{midi, dur, art}] → list of (pitch_cost, energy_cost) per boundary."""
    n = len(score)
    onsets = []
    t = 0.0
    for s in score:
        onsets.append(t)
        t += s["dur"]

    trajs = []
    prev_traj = None
    for i, note in enumerate(score):
        feat = _build_feat(note, i, n, onsets, score)
        traj = nn.predict(feat, prev_traj=prev_traj, concat_lambda=concat_lambda)
        trajs.append(traj)
        prev_traj = traj

    costs = []
    for i in range(n - 1):
        p_end = float(np.mean(trajs[i][0, -EDGE:]))
        p_start = float(np.mean(trajs[i + 1][0, :EDGE]))
        e_end = float(np.mean(trajs[i][1, -EDGE:]))
        e_start = float(np.mean(trajs[i + 1][1, :EDGE]))
        costs.append((abs(p_end - p_start), abs(e_end - e_start)))
    return costs


def unique_source_fraction(score, nn, concat_lambda=0.0):
    """检索多样性:不同源轨迹占比。λ 过大会导致同几条轨迹反复选中。"""
    n = len(score)
    onsets = []
    t = 0.0
    for s in score:
        onsets.append(t)
        t += s["dur"]

    indices = []
    prev_traj = None
    for i, note in enumerate(score):
        feat = _build_feat(note, i, n, onsets, score)
        pool, dist = nn._pool_and_dist(feat)
        if prev_traj is None or concat_lambda <= 0:
            idx = int(pool[int(np.argmin(dist))])
        else:
            k = min(8, len(pool))
            top_k = np.argpartition(dist, k)[:k]
            e = nn._CONCAT_EDGE
            prev_p = float(np.mean(prev_traj[0, -e:]))
            prev_e = float(np.mean(prev_traj[1, -e:]))
            best_j, best_c = top_k[0], float("inf")
            for j in top_k:
                cand = nn.trajectories[pool[j]]
                cc = abs(prev_p - float(np.mean(cand[0, :e]))) + abs(
                    prev_e - float(np.mean(cand[1, :e]))
                )
                total = float(dist[j]) + concat_lambda * cc
                if total < best_c:
                    best_c = total
                    best_j = j
            idx = int(pool[best_j])
        indices.append(idx)
        prev_traj = nn.trajectories[idx]
    return len(set(indices)) / max(len(indices), 1)


def melody_score(raw):
    return [{"midi": m, "dur": d, "art": a} for m, d, a in raw]


def _summarize(all_costs):
    all_p = [c[0] for cs in all_costs for c in cs]
    all_e = [c[1] for cs in all_costs for c in cs]
    return {
        "pitch_mean": round(float(np.mean(all_p)), 2),
        "pitch_median": round(float(np.median(all_p)), 2),
        "pitch_p90": round(float(np.percentile(all_p, 90)), 2),
        "energy_mean": round(float(np.mean(all_e)), 2),
        "energy_median": round(float(np.median(all_e)), 2),
        "energy_p90": round(float(np.percentile(all_e, 90)), 2),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="sweep concat_lambda values")
    ap.add_argument("--eartest", action="store_true", help="generate A/B ear-test wavs")
    args = ap.parse_args()

    nt = NeuralTake(device="cpu")
    behaviors = ["violin", "guitar"]

    lambdas = LAMBDAS if args.sweep else [0.0]
    results = {}

    for lam in lambdas:
        lam_key = f"lambda_{lam:.1f}"
        results[lam_key] = {}
        print(f"\n=== concat_lambda={lam} ===")
        for beh in behaviors:
            nn = nt.voice(beh).nn
            all_costs = []
            usf_list = []
            for mname, raw in MELODIES.items():
                score = melody_score(raw)
                costs = continuity_costs(score, nn, concat_lambda=lam)
                all_costs.append(costs)
                usf = unique_source_fraction(score, nn, concat_lambda=lam)
                usf_list.append(usf)
                pitch_costs = [c[0] for c in costs]
                print(
                    f"  {beh}/{mname}: pitch mean={np.mean(pitch_costs):.1f}c  "
                    f"p90={np.percentile(pitch_costs, 90):.1f}c  usf={usf:.2f}"
                )
            s = _summarize(all_costs)
            s["unique_source_fraction"] = round(float(np.mean(usf_list)), 3)
            results[lam_key][beh] = s
            print(
                f"  {beh} 全局: pitch mean={s['pitch_mean']}c  "
                f"p90={s['pitch_p90']}c  usf={s['unique_source_fraction']}"
            )

    if args.sweep:
        print("\n=== improvement vs baseline (λ=0) ===")
        for beh in behaviors:
            base = results["lambda_0.0"][beh]["pitch_mean"]
            for lam in lambdas[1:]:
                cur = results[f"lambda_{lam:.1f}"][beh]["pitch_mean"]
                pct = (1 - cur / base) * 100 if base > 0 else 0
                usf = results[f"lambda_{lam:.1f}"][beh]["unique_source_fraction"]
                print(f"  {beh} λ={lam}: pitch -{pct:.0f}%  usf={usf:.3f}")

    if args.eartest:
        import soundfile as sf

        WAV_DIR.mkdir(parents=True, exist_ok=True)
        wavs = []
        for mname in ["lyrical", "folk"]:
            for beh in behaviors:
                for lam_str, lam in [("l0", 0.0), ("l1", 1.0)]:
                    from tools.eval_timing import melody_notes

                    notes = melody_notes(MELODIES[mname])
                    y, sr = nt.render(notes, instrument="violin", behavior=beh, continuity=lam)
                    name = f"{mname}_viT_{beh[:2]}B_{lam_str}.wav"
                    sf.write(str(WAV_DIR / name), y, sr)
                    wavs.append(name)
                    print(f"  eartest: {name}")
        results["ear_test_wavs"] = wavs
        print(f"耳测 {len(wavs)} 条 -> {WAV_DIR}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\nresults -> {RESULTS}")


if __name__ == "__main__":
    main()
