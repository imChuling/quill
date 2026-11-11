"""标注器真实数据验证:VocalSet 歌唱技法标签 vs classify_articulation。

VocalSet(Wilkins et al., ISMIR 2018)的 vibrato / straight 技法子集:
整个文件用同一技法演唱 → 文件级真值。走完整生产管线(extract_style 的
segment_notes 自动切分 + event_trajectories + classify_articulation),统计:

  - 音符级:vibrato 文件中被标 vibrato 的音符占比 vs straight 文件中的占比
  - 文件级:多数票分类(文件内 vibrato 音符占比 > 阈值 → vibrato)的准确率

歌声是跨域测试(库为器乐设计):起音上挑(scoop)可能触发 slide,
按论文口径 slide 单独统计,不并入 plain。

    conda run -n mt python tools/validate_labeler_vocalset.py \
        --vocalset $QUILL_DATASETS/vocalset/FULL

产出: checkpoints/eval/labeler_validation_vocalset.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

QUILL = Path(__file__).resolve().parent.parent
ART_NAMES = {0: "plain", 1: "vibrato", 2: "slide"}
MIN_DUR = 0.25  # 与合成验证同口径:短于一个揉弦周期的音不计


def _segment_stable_notes(f0, f0_times, min_dur=0.35, tol_semi=0.5, smooth_s=0.31):
    """验证侧切分:0.3s 中值滤波抹平揉弦(5-6Hz 周期 0.17-0.2s)后取稳定音高段。

    production 的 segment_notes 是给器乐调的(静音地板在呼吸声/气声上误触发),
    这里的目标是隔离标注器——切分基于平滑轨迹,打标仍走生产管线原始音频。
    """
    semi = 12.0 * np.log2(np.maximum(f0, 1e-6) / 440.0) + 69.0
    n = len(semi)
    if n < 10:
        return []
    dt = float(np.median(np.diff(f0_times))) if n > 1 else 0.006
    k = max(3, int(round(smooth_s / dt)) | 1)
    sm = np.full(n, np.nan)
    half = k // 2
    for j in range(n):
        win = semi[max(0, j - half) : j + half + 1]
        if np.isfinite(win).sum() >= half:
            sm[j] = np.nanmedian(win)
    notes, start = [], None
    ref = np.nan
    for j in range(n + 1):
        v = sm[j] if j < n else np.nan
        if start is not None and (not np.isfinite(v) or abs(v - ref) > tol_semi):
            if f0_times[j - 1] - f0_times[start] >= min_dur:
                notes.append((float(f0_times[start]), float(f0_times[j - 1])))
            start = None
        if start is None and np.isfinite(v):
            start, ref = j, v
        elif start is not None:
            ref = 0.95 * ref + 0.05 * v
    # 掐头去尾 50ms(过渡帧),再过滤过短段
    out = []
    for a, b in notes:
        a, b = a + 0.05, b - 0.05
        if b - a >= MIN_DUR:
            out.append((a, b))
    return out


def label_file(wav_path):
    import soundfile as sf

    from analysis.tplib import event_trajectories
    from tools.extract_style import classify_articulation, track_f0

    y, sr = sf.read(str(wav_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    f0, f0_times = track_f0(y, sr)
    notes = _segment_stable_notes(f0, f0_times)
    out = []
    for t_on, t_off in notes:
        traj = event_trajectories(y, sr, t_on, t_off)
        p = traj["pitch"]
        pts = np.asarray(p["pts"])
        pred = classify_articulation(pts, float(p["confidence"]))
        rng = float(pts.max() - pts.min())
        d = np.diff(pts)
        sc = int(np.sum(np.abs(np.diff(np.sign(d[np.abs(d) > 1e-3]))) > 0))
        out.append(
            {
                "t_on": round(t_on, 3),
                "dur": round(t_off - t_on, 3),
                "pred": pred,
                "rng": round(rng, 1),
                "sign_changes": sc,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocalset", required=True)
    ap.add_argument("--subset", default="scales", choices=["scales", "arpeggios", "all"])
    ap.add_argument("--limit-per-class", type=int, default=0, help="0=全部")
    ap.add_argument("--file-vote-threshold", type=float, default=0.5)
    ap.add_argument(
        "--out", default=str(QUILL / "checkpoints/eval/labeler_validation_vocalset.json")
    )
    args = ap.parse_args()
    root = Path(args.vocalset)

    def collect(tech):
        if args.subset == "all":
            pats = [f"*/{s}/{tech}/*.wav" for s in ("scales", "arpeggios")]
        else:
            pats = [f"*/{args.subset}/{tech}/*.wav"]
        files = sorted(f for p in pats for f in root.glob(p))
        if args.limit_per_class:
            files = files[: args.limit_per_class]
        return files

    report = {
        "schema": "quill.labeler_validation_vocalset.v1",
        "subset": args.subset,
        "min_dur": MIN_DUR,
        "classes": {},
    }
    file_rows = []
    for tech, expect in [("straight", "plain"), ("vibrato", "vibrato")]:
        files = collect(tech)
        print(f"\n=== {tech}: {len(files)} files ===")
        note_preds = []
        per_file = []
        for i, f in enumerate(files):
            try:
                labels = label_file(f)
            except Exception as e:
                print(f"  [skip] {f.name}: {e}")
                continue
            if not labels:
                continue
            preds = [r["pred"] for r in labels]
            vib_frac = float(np.mean([p == 1 for p in preds]))
            per_file.append(
                {
                    "file": str(f.relative_to(root)),
                    "n_notes": len(preds),
                    "vib_frac": vib_frac,
                    "preds": preds,
                    "preds_detail": labels,
                }
            )
            note_preds.extend(preds)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(files)} files...", flush=True)
        n = max(len(note_preds), 1)
        dist = {name: sum(1 for p in note_preds if p == k) / n for k, name in ART_NAMES.items()}
        votes = [pf["vib_frac"] > args.file_vote_threshold for pf in per_file]
        file_acc = (
            float(np.mean([v == (tech == "vibrato") for v in votes])) if votes else float("nan")
        )
        report["classes"][tech] = {
            "expected": expect,
            "n_files": len(per_file),
            "n_notes": len(note_preds),
            "note_dist": dist,
            "file_vote_acc": file_acc,
        }
        file_rows.extend({**pf, "tech": tech} for pf in per_file)
        print(
            f"  音符级 (n={len(note_preds)}): plain {dist['plain']:.2f} "
            f"vibrato {dist['vibrato']:.2f} slide {dist['slide']:.2f}"
        )
        print(f"  文件级多数票准确率: {file_acc:.2f} ({len(per_file)} files)")

    # ---- 跨域阈值分析:判别特征(pitch range,揉弦周期性门控)的可分性 ----
    # 生产阈值 30c 为器乐标定;人声直音天然抖 50-90c → 报 AUC 与最优阈值。
    y_true, y_score = [], []
    for pf in file_rows:
        is_vib = pf["tech"] == "vibrato"
        for r in pf["preds_detail"]:
            y_true.append(is_vib)
            y_score.append(r["rng"] if r["sign_changes"] >= 4 else 0.0)
    if y_true:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        order = np.argsort(y_score)
        pos = y_true.sum()
        neg = len(y_true) - pos
        ranks = np.empty(len(y_score))
        ranks[order] = np.arange(1, len(y_score) + 1)
        auc = float((ranks[y_true].sum() - pos * (pos + 1) / 2) / max(pos * neg, 1))
        best_acc, best_thr = 0.0, 0.0
        for thr in np.unique(y_score):
            acc = float(np.mean((y_score >= thr) == y_true))
            if acc > best_acc:
                best_acc, best_thr = acc, float(thr)
        acc_at_30 = float(np.mean((y_score >= 30.0) == y_true))
        report["threshold_analysis"] = {
            "note_auc": auc,
            "acc_at_violin_thr_30c": acc_at_30,
            "best_thr_cents": best_thr,
            "acc_at_best_thr": best_acc,
            "n_notes": int(len(y_true)),
        }
        print(f"\n=== 跨域阈值分析 (n={len(y_true)}) ===")
        print(f"  AUC(range, 周期门控) = {auc:.3f}")
        print(f"  器乐阈值 30c 准确率 = {acc_at_30:.2f}")
        print(f"  最优阈值 {best_thr:.0f}c 准确率 = {best_acc:.2f}")

    report["files"] = file_rows
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\n结果已存 {out}")


if __name__ == "__main__":
    main()
