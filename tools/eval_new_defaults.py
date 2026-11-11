"""E4: 新默认参数 vs 论文默认参数对比。

论文默认: concat_lambda=0, contiguity=0
新默认:   concat_lambda=1, contiguity=1 (UI_RENDER_DEFAULTS)

跑四条证据链的客观部分(控制保真度、配对反事实),确认不劣化。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
OUT = CKPT / "eval"

SR = 16000
FPS = 250

MELODIES = {
    "lyrical": [
        (62, 0.8, 1), (65, 0.4, 0), (67, 1.2, 1), (65, 0.4, 0),
        (69, 0.4, 2), (72, 1.6, 1), (67, 0.8, 0), (62, 1.6, 1),
    ],
    "scale_run": [
        (62, 0.2, 0), (64, 0.2, 0), (65, 0.2, 0), (67, 0.2, 0),
        (69, 0.2, 0), (71, 0.2, 0), (72, 0.2, 0), (74, 0.2, 0),
        (76, 0.4, 0), (74, 0.4, 0), (72, 1.6, 1),
    ],
    "leaps": [
        (57, 0.4, 0), (64, 0.4, 0), (62, 0.4, 2), (69, 0.8, 1),
        (65, 0.4, 0), (72, 0.4, 2), (71, 0.8, 0), (64, 1.2, 1),
    ],
    "folk": [
        (69, 0.4, 0), (71, 0.2, 0), (72, 0.4, 0), (69, 0.4, 0),
        (67, 0.8, 1), (64, 0.4, 0), (67, 0.4, 2), (69, 1.2, 1),
        (67, 0.4, 0), (64, 0.4, 0), (62, 1.6, 1),
    ],
}


def analyze_control(f0_np, score):
    FPS_CTRL = 250
    CF = 5
    out = []
    t_frame = 0
    prev_T = None
    for _m, dur, art in score:
        T = max(4, int(dur * FPS_CTRL))
        if prev_T is not None:
            t_frame -= min(CF, prev_T // 2, T // 2)
        margin = max(12, T // 5)
        i0 = t_frame + (6 if art == 2 else margin)
        i1 = t_frame + T - margin
        t_frame += T
        prev_T = T
        if i1 <= i0 or i1 > len(f0_np) or dur < 0.3:
            out.append({"vib_energy": np.nan, "total_rms": np.nan, "drift": 0.0, "dur": dur, "art": art})
            continue
        seg = f0_np[i0:i1]
        cents = 1200 * np.log2(seg / (np.median(seg) + 1e-9) + 1e-12)
        win = np.hanning(len(cents))
        spec = np.abs(np.fft.rfft((cents - cents.mean()) * win))
        freqs = np.fft.rfftfreq(len(cents), d=1.0 / FPS_CTRL)
        vib_band = (freqs >= 3.5) & (freqs <= 9.0)
        Nc = max(len(cents), 1)
        vib_e = float(np.sqrt((spec[vib_band] ** 2).sum() / Nc ** 2))
        total_rms = float(np.sqrt((cents ** 2).mean()))
        q = max(2, len(cents) // 4)
        qh = max(2, min(4, len(cents) // 8))
        drift = float(np.median(cents[-q:]) - np.median(cents[:qh]))
        out.append({"vib_energy": vib_e, "total_rms": total_rms, "drift": drift, "dur": dur, "art": art})
    return out


def _bootstrap_median_ci(arr, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(42)
    medians = np.array(
        [float(np.median(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    )
    return float(np.percentile(medians, 100 * alpha / 2)), float(np.percentile(medians, 100 * (1 - alpha / 2)))


def run_paired_counterfactual(nn, concat_lambda, contiguity, label):
    from neural.perform import control_from_behavior

    EPS = 1e-4
    ctrl_logr = []
    ctrl_gt1 = 0
    by_mel = {}
    for mname, score in MELODIES.items():
        score_vib = [(m, d, 1) for (m, d, _a) in score]
        score_pln = [(m, d, 0) for (m, d, _a) in score]
        notes_vib = [{"midi": m, "dur": d, "art": a} for m, d, a in score_vib]
        notes_pln = [{"midi": m, "dur": d, "art": a} for m, d, a in score_pln]

        f0_v, _ = control_from_behavior(
            notes_vib, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0,
            concat_lambda=concat_lambda, contiguity=contiguity,
        )
        f0_p, _ = control_from_behavior(
            notes_pln, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0,
            concat_lambda=concat_lambda, contiguity=contiguity,
        )
        ana_v = analyze_control(f0_v.numpy()[0, :, 0], score_vib)
        ana_p = analyze_control(f0_p.numpy()[0, :, 0], score_pln)
        mel_lrs = []
        for av, ap in zip(ana_v, ana_p):
            if np.isnan(av["vib_energy"]) or np.isnan(ap["vib_energy"]):
                continue
            lr = np.log2((av["vib_energy"] + EPS) / (ap["vib_energy"] + EPS))
            ctrl_logr.append(lr)
            mel_lrs.append(lr)
            if lr > 0:
                ctrl_gt1 += 1
        by_mel[mname] = mel_lrs

    arr = np.array(ctrl_logr)
    med = float(2 ** np.median(arr))
    ci = tuple(float(2 ** x) for x in _bootstrap_median_ci(arr))
    per_mel = {m: round(float(2 ** np.median(v)), 2) for m, v in by_mel.items() if v}
    return {
        "paired_median_ratio": round(med, 2),
        "ci95": [round(ci[0], 2), round(ci[1], 2)],
        "n": len(ctrl_logr),
        "gt1": ctrl_gt1,
        "per_melody": per_mel,
    }


def run_controllability(nn, concat_lambda, contiguity, label):
    from neural.perform import control_from_behavior

    vib_es, plain_es = [], []
    for _mname, score in MELODIES.items():
        notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
        f0, _ = control_from_behavior(
            notes, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0,
            concat_lambda=concat_lambda, contiguity=contiguity,
        )
        for a in analyze_control(f0.numpy()[0, :, 0], score):
            if np.isnan(a["vib_energy"]):
                continue
            if a["art"] == 1:
                vib_es.append(a["vib_energy"])
            elif a["art"] == 0:
                plain_es.append(a["vib_energy"])
    vib_m = float(np.median(vib_es)) if vib_es else 0
    plain_m = float(np.median(plain_es)) if plain_es else 0
    return {
        "vib_median": round(vib_m, 2),
        "plain_median": round(plain_m, 2),
        "contrast": round(vib_m / max(plain_m, 0.1), 2),
    }


def main():
    from neural.neural_take import ConditionedNN

    np.random.seed(42)
    torch.manual_seed(42)

    d = np.load(str(CKPT / "urmp_vn_trajectories.npz"), allow_pickle=False)
    nn = ConditionedNN(d["features"], d["trajectories"])

    configs = {
        "paper_default": {"concat_lambda": 0.0, "contiguity": 0.0},
        "new_default": {"concat_lambda": 1.0, "contiguity": 1.0},
    }

    results = {}
    print("=== E4: New default vs paper default ===\n")
    for label, cfg in configs.items():
        print(f"--- {label} (λ={cfg['concat_lambda']}, contig={cfg['contiguity']}) ---")
        ctrl = run_controllability(nn, **cfg, label=label)
        pc = run_paired_counterfactual(nn, **cfg, label=label)
        results[label] = {"controllability": ctrl, "paired_counterfactual": pc}
        print(f"  Controllability: contrast {ctrl['contrast']}×")
        print(f"  Paired counterfactual: median ratio {pc['paired_median_ratio']}× "
              f"CI [{pc['ci95'][0]}, {pc['ci95'][1]}]  {pc['gt1']}/{pc['n']} >1")
        print(f"  Per melody: {pc['per_melody']}")
        print()

    p = results["paper_default"]["paired_counterfactual"]["paired_median_ratio"]
    n = results["new_default"]["paired_counterfactual"]["paired_median_ratio"]
    delta = n - p
    print(f"Delta (new - paper): {delta:+.2f}×")
    if n >= p * 0.8:
        print("PASS: new defaults do not degrade (within 20%)")
    else:
        print("WARN: new defaults may degrade performance")

    out_path = OUT / "new_defaults_comparison.json"
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
