"""Timbre fidelity 交叉验证:4-fold leave-one-take-out + leave-one-glass-out。

- LOTO:三只多 take 杯子(Amaltheus/Auburn/C#6-073 各 4 takes),第 k 折留出
  每杯的 take k(k=1..4),训练 6k 步,对留出 take 测重建 → n=12,
  take 级泛化(同杯其他 take 在训练集)。
- LOGO:整杯留出(训练只见其他杯+Fishbowl),测未见杯/未见音高的重建
  → 乐器级泛化基线,回应"测试音频来自训练中出现过的同一只杯子"的质疑。
- 错配参照:violin/trumpet 预训练模型重建同一素材。

跑: conda run -n mt python tools/eval_timbre_cv.py
产出: checkpoints/eval/timbre_fidelity_cv.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
SR = 16000
TRAIN_DIR = QUILL / "assets/glass/timbre_train"

GLASSES = {
    "amaltheus": [f"Amaltheus Cellars Glass Note {i}.wav" for i in (1, 2, 3, 4)],
    "auburn": [f"Auburn Road Wine Glass Note {i}.wav" for i in (1, 2, 3, 4)],
    "c5": [f"C#6-073-0{i}.wav" for i in (1, 2, 3, 4)],
}
SINGLETONS = ["Fishbowl-1.wav"]  # 单 take,永远在训练集


def load_ref_model(ck):
    from neural.ddsp import DDSP

    m = DDSP(dropout=0.0)
    m.load_state_dict(torch.load(str(ck), map_location="cpu", weights_only=True))
    m.eval()
    return m


def mfcc_dist(a, b):
    import librosa

    ma = librosa.feature.mfcc(y=a, sr=SR, n_mfcc=13)[1:].mean(axis=1)
    mb = librosa.feature.mfcc(y=b, sr=SR, n_mfcc=13)[1:].mean(axis=1)
    return float(np.linalg.norm(ma - mb))


def recon_metrics(model, f0, loud, orig):
    from neural.ddsp import multiscale_spectral_loss

    with torch.no_grad():
        yhat = model(f0, loud).numpy()[0]
    m = min(len(yhat), len(orig))
    msl = float(
        multiscale_spectral_loss(torch.tensor(yhat[None, :m]), torch.tensor(orig[None, :m])).item()
    )
    return round(msl, 2), round(mfcc_dist(yhat[:m], orig[:m]), 1)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    from analysis.io import load
    from neural.ddsp import extract_features
    from neural.train_ddsp import _trim_silence, build_dataset, train

    # 错配参照模型:trumpet checkpoint 已随代码清理删除;论文引用的完整
    # 双参照结果存档于 checkpoints/eval/timbre_fidelity_cv.json(已入 git)
    foreign = {}
    for fname, ck in [("violin", "ddsp_urmp_vn_12k.pt"), ("trumpet", "ddsp_urmp_tpt_12k.pt")]:
        if (CKPT / ck).exists():
            foreign[fname] = load_ref_model(CKPT / ck)
        else:
            print(f"[warn] {ck} 缺失,跳过 {fname} 错配参照")

    def eval_files(model, names, fold_tag, rows):
        for name in names:
            y, _ = load(str(TRAIN_DIR / name), sr=44100)
            y = _trim_silence(y, 44100)
            f0, loud, au = extract_features(y, 44100)
            orig = au.numpy()[0]
            row = {"fold": fold_tag, "file": name}
            msl, mf = recon_metrics(model, f0, loud, orig)
            row["msl_own"], row["mfcc_own"] = msl, mf
            for fname, fmodel in foreign.items():
                msl, mf = recon_metrics(fmodel, f0, loud, orig)
                row[f"msl_{fname}"], row[f"mfcc_{fname}"] = msl, mf
            rows.append(row)
            extras = "  ".join(
                f"{fn} {row[f'msl_{fn}']:6.2f}/{row[f'mfcc_{fn}']:5.1f}" for fn in foreign
            )
            print(f"  {name[:32]:32s} own {row['msl_own']:6.2f}/{row['mfcc_own']:5.1f}  {extras}")

    # ---- LOTO: 4-fold leave-one-take-out ----
    loto_rows = []
    for fold in range(4):
        held = [names[fold] for names in GLASSES.values()]
        train_files = [
            str(TRAIN_DIR / n) for names in GLASSES.values() for n in names if n not in held
        ] + [str(TRAIN_DIR / s) for s in SINGLETONS]
        print(f"\n=== LOTO fold {fold}: 留出 {held} | 训练 {len(train_files)} 段 ===")
        data = build_dataset(train_files, cache=None, verbose=False)
        model, _losses, best_val, _vi = train(data, steps=6000, verbose=False)
        model = model.cpu().eval()
        print(f"  训练完成 best_val={best_val:.2f}")
        eval_files(model, held, f"loto{fold}", loto_rows)

    # ---- LOGO: leave-one-glass-out(乐器级泛化基线) ----
    logo_rows = []
    for gname, names in GLASSES.items():
        train_files = [
            str(TRAIN_DIR / n) for g2, ns in GLASSES.items() if g2 != gname for n in ns
        ] + [str(TRAIN_DIR / s) for s in SINGLETONS]
        print(f"\n=== LOGO: 整杯留出 {gname} | 训练 {len(train_files)} 段 ===")
        data = build_dataset(train_files, cache=None, verbose=False)
        model, _losses, best_val, _vi = train(data, steps=6000, verbose=False)
        model = model.cpu().eval()
        print(f"  训练完成 best_val={best_val:.2f}")
        eval_files(model, names, f"logo_{gname}", logo_rows)

    def med(rows, k):
        return float(np.median([r[k] for r in rows]))

    out_res = {}
    ref_keys = ["own"] + list(foreign)
    for tag, rows in [("loto", loto_rows), ("logo", logo_rows)]:
        print(f"\n=== {tag.upper()} 汇总 (n={len(rows)}) ===")
        for k in ref_keys:
            print(
                f"  {k:>8s}: 谱损失中位 {med(rows, f'msl_{k}'):.2f}  "
                f"MFCC 中位 {med(rows, f'mfcc_{k}'):.1f}"
            )
        wins = (
            sum(1 for r in rows if r["msl_own"] < min(r[f"msl_{fn}"] for fn in foreign))
            if foreign
            else 0
        )
        print(f"  own 谱损失最优: {wins}/{len(rows)}")
        out_res[tag] = {
            "rows": rows,
            "median": {
                k: {"msl": med(rows, f"msl_{k}"), "mfcc": med(rows, f"mfcc_{k}")} for k in ref_keys
            },
            "own_wins_msl": f"{wins}/{len(rows)}",
        }

    out = CKPT / "eval" / "timbre_fidelity_cv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w") as f:
        json.dump(out_res, f, indent=2, ensure_ascii=False)
    print(f"结果已存 {out}")


if __name__ == "__main__":
    main()
