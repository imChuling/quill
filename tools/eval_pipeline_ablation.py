"""B2: 渲染管线各步的损耗 — 四个条件逐步加。

条件:
  ①raw:检索轨迹原样,不拉伸不锚定
  ②+stretch:时间拉伸到音符时值
  ③+anchor:+音准锚定(减均值)
  ④+label_proc:+标签后处理(=full)

同 29 对配对音(与 processing_ablation 同协议),每步算配对中位比。

产出: checkpoints/eval/pipeline_ablation.json + 终端表
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
OUT = CKPT / "eval"
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


def vib_energy(pitch_cents, sr):
    """3–9 Hz band energy from a pitch trajectory (cents), given sample rate."""
    N = len(pitch_cents)
    if N < 8:
        return 0.0
    win = np.hanning(N)
    spec = np.abs(np.fft.rfft((pitch_cents - pitch_cents.mean()) * win))
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    band = (freqs >= 3.5) & (freqs <= 9.0)
    if not band.any():
        return 0.0
    return float(np.sqrt((spec[band] ** 2).sum() / N**2))


def retrieve_trajectories(nn, score):
    """Retrieve raw trajectories for a score, mimicking control_from_behavior's retrieval."""
    trajs = []
    nn.last_idx = None
    for i, (midi, dur, art) in enumerate(score):
        prev_gap = 1.0 if i == 0 else 0.0
        next_gap = 1.0 if i == len(score) - 1 else 0.0
        feat = np.array([
            float(midi), dur, prev_gap, next_gap,
            1.0 if i == 0 else 0.0,
            1.0 if i == len(score) - 1 else 0.0,
            float(art),
        ], np.float32)
        traj = nn.predict(feat)
        trajs.append(traj.copy())
    return trajs


def process_condition(trajs, score, condition):
    """Apply processing up to `condition` level, return per-note vib_energy."""
    energies = []
    for _i, ((_midi, dur, art), traj) in enumerate(zip(score, trajs)):
        pitch_raw = traj[0]  # 48 frames
        n = max(4, int(dur * FPS))

        if condition == "raw":
            effective_sr = 48.0 / dur
            cents = pitch_raw - np.median(pitch_raw)
            energies.append(vib_energy(cents, effective_sr))
            continue

        # stretch
        pitch_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), pitch_raw)

        if condition == "stretch":
            cents = pitch_d - np.median(pitch_d)
            energies.append(vib_energy(cents, FPS))
            continue

        # anchor
        anchor = pitch_d.mean()
        pitch_d = pitch_d - anchor

        if condition == "anchor":
            energies.append(vib_energy(pitch_d, FPS))
            continue

        # label_proc (full)
        if art == 0 and n >= 8:
            from scipy.signal import butter, sosfiltfilt
            cutoff = 3.0
            nyq = FPS / 2.0
            if cutoff < nyq:
                sos = butter(2, cutoff / nyq, btype="low", output="sos")
                pitch_d = sosfiltfilt(sos, pitch_d)
            pitch_d = np.clip(pitch_d, -50.0, 50.0)
        elif art == 1:
            if n >= 8:
                from scipy.signal import butter, sosfiltfilt
                nyq = FPS / 2.0
                sos_d = butter(2, 2.0 / nyq, btype="low", output="sos")
                pitch_d = pitch_d - sosfiltfilt(sos_d, pitch_d)
            pitch_d = np.clip(pitch_d, -35.0, 35.0)

        energies.append(vib_energy(pitch_d, FPS))

    return energies


def _bootstrap_median_ci(arr, n_boot=5000, alpha=0.05):
    rng = np.random.default_rng(42)
    meds = [float(np.median(rng.choice(arr, len(arr)))) for _ in range(n_boot)]
    lo = np.percentile(meds, 100 * alpha / 2)
    hi = np.percentile(meds, 100 * (1 - alpha / 2))
    return lo, hi


def main():
    from neural.neural_take import ConditionedNN

    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(str(CKPT / "urmp_vn_trajectories.npz"))
    nn = ConditionedNN(d["features"], d["trajectories"])

    conditions = ["raw", "stretch", "anchor", "label_proc"]
    results = {}

    print("=== B2: 渲染管线各步损耗(配对反事实) ===\n")

    for cond in conditions:
        logr_all = []
        by_melody = {}
        for mname, score in MELODIES.items():
            score_vib = [(m, d, 1) for (m, d, _a) in score]
            score_pln = [(m, d, 0) for (m, d, _a) in score]

            trajs_v = retrieve_trajectories(nn, score_vib)
            trajs_p = retrieve_trajectories(nn, score_pln)

            ve_v = process_condition(trajs_v, score_vib, cond)
            ve_p = process_condition(trajs_p, score_pln, cond)

            EPS = 1e-4
            mel_lrs = []
            for (_, dur, _), ev, ep in zip(score, ve_v, ve_p):
                if dur < 0.3:
                    continue
                if ev < EPS and ep < EPS:
                    continue
                lr = np.log2((ev + EPS) / (ep + EPS))
                logr_all.append(lr)
                mel_lrs.append(lr)
            by_melody[mname] = mel_lrs

        arr = np.array(logr_all)
        med = float(2 ** np.median(arr))
        ci = tuple(float(2**x) for x in _bootstrap_median_ci(arr))
        per_mel = {m: round(float(2 ** np.median(v)), 2)
                   for m, v in by_melody.items() if v}

        results[cond] = {
            "paired_median_ratio": round(med, 2),
            "item_ci95": [round(ci[0], 2), round(ci[1], 2)],
            "n": len(logr_all),
            "per_melody_median": per_mel,
        }
        pm = " ".join(f"{m}:{v:.1f}×" for m, v in per_mel.items())
        print(f"  {cond:>12s}: {med:.2f}×  CI [{ci[0]:.1f}, {ci[1]:.1f}]  n={len(logr_all)}  | {pm}")

    out_path = OUT / "pipeline_ablation.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
