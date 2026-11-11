"""B1: 库本身的对比度 — vibrato 池 vs plain 池的 pitch 3–9 Hz 调制能量。

产出: checkpoints/eval/library_contrast.json + 终端表
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
OUT = CKPT / "eval"

LIBRARIES = {
    "violin": "urmp_vn_trajectories.npz",
    "trumpet": "urmp_tpt_trajectories.npz",
    "guitar": "guitarset_trajectories.npz",
    "glass": "glass_style.npz",
}

DUR_BINS = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 999)]
DUR_LABELS = ["<0.3s", "0.3–0.6s", "0.6–1.0s", ">1.0s"]


def vib_energy_48(traj_pitch, dur):
    """从 48 帧时间归一化 pitch 轨迹提取 3–9 Hz 调制能量。"""
    N = len(traj_pitch)
    if N < 8 or dur < 0.1:
        return 0.0
    effective_sr = N / dur
    cents = traj_pitch - np.median(traj_pitch)
    win = np.hanning(N)
    spec = np.abs(np.fft.rfft(cents * win))
    freqs = np.fft.rfftfreq(N, d=1.0 / effective_sr)
    vib_band = (freqs >= 3.0) & (freqs <= 9.0)
    if not vib_band.any():
        return 0.0
    return float(np.sqrt((spec[vib_band] ** 2).sum() / N**2))


def analyze_library(name, path):
    d = np.load(path, allow_pickle=False)
    trajs = d["trajectories"]  # (N, 4, 48)
    meta = json.loads(d["meta"].item())

    notes = []
    for i, m in enumerate(meta):
        dur = m.get("dur", float(d["features"][i, 1]))
        art = m.get("art", -1)
        ve = vib_energy_48(trajs[i, 0, :], dur)
        notes.append({"dur": dur, "art": art, "vib_energy": ve})
    return notes


def compute_contrast(notes, label):
    vib = [n["vib_energy"] for n in notes if n["art"] == 1]
    pln = [n["vib_energy"] for n in notes if n["art"] == 0]
    if not vib or not pln:
        return None
    med_v = float(np.median(vib))
    med_p = float(np.median(pln))
    EPS = 1e-6
    ratio = med_v / (med_p + EPS)
    return {
        "label": label,
        "n_vibrato": len(vib),
        "n_plain": len(pln),
        "median_vib": round(med_v, 4),
        "median_plain": round(med_p, 4),
        "contrast_ratio": round(ratio, 2),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    for lib_name, fname in LIBRARIES.items():
        path = CKPT / fname
        if not path.exists():
            print(f"SKIP {lib_name}: {fname} not found")
            continue

        notes = analyze_library(lib_name, path)
        has_labels = any(n["art"] in (0, 1) for n in notes)

        print(f"\n=== {lib_name} ({len(notes)} notes) ===")

        if not has_labels:
            print("  no articulation labels — skipping vibrato/plain split")
            all_ve = [n["vib_energy"] for n in notes]
            results[lib_name] = {
                "has_labels": False,
                "n": len(notes),
                "median_vib_energy": round(float(np.median(all_ve)), 4),
                "p90_vib_energy": round(float(np.percentile(all_ve, 90)), 4),
            }
            continue

        overall = compute_contrast(notes, "all")
        print(
            f"  overall: vibrato {overall['n_vibrato']} notes "
            f"(med {overall['median_vib']:.3f}) vs "
            f"plain {overall['n_plain']} notes "
            f"(med {overall['median_plain']:.3f}) "
            f"→ contrast {overall['contrast_ratio']:.1f}×"
        )

        by_dur = []
        for (lo, hi), dlabel in zip(DUR_BINS, DUR_LABELS):
            subset = [n for n in notes if lo <= n["dur"] < hi]
            c = compute_contrast(subset, dlabel)
            if c:
                by_dur.append(c)
                print(
                    f"  {dlabel:>10s}: vib {c['n_vibrato']:4d}  "
                    f"plain {c['n_plain']:4d}  "
                    f"contrast {c['contrast_ratio']:.1f}×"
                )
            else:
                print(f"  {dlabel:>10s}: insufficient data")

        results[lib_name] = {
            "has_labels": True,
            "overall": overall,
            "by_duration": by_dur,
        }

    out_path = OUT / "library_contrast.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out_path}")
    return results


if __name__ == "__main__":
    main()
