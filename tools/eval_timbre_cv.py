"""Timbre fidelity cross-validation: 4-fold leave-one-take-out + leave-one-glass-out.

- LOTO: three glasses with several takes (Amaltheus/Auburn/C#6-073, 4 takes each); fold k holds out
  take k of each glass (k=1..4), trains 6k steps, tests reconstruction on the held-out take -> n=12,
  take-level generalisation (the glass's other takes are in the training set).
- LOGO: hold out a whole glass (training sees only the other glasses + Fishbowl), test reconstruction of an unseen glass/unseen pitch
  -> instrument-level generalisation baseline, answering the objection that "the test audio comes from the same glass seen in training".
- Mismatched reference: violin/trumpet pretrained models reconstruct the same material.

Run: conda run -n mt python tools/eval_timbre_cv.py
Outputs: checkpoints/eval/timbre_fidelity_cv.json
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
SINGLETONS = ["Fishbowl-1.wav"]  # single take, always in the training set


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

    # mismatched reference models: the trumpet checkpoint was deleted in a code cleanup; the complete
    # two-reference results cited in the paper are archived in checkpoints/eval/timbre_fidelity_cv.json (tracked in git)
    foreign = {}
    for fname, ck in [("violin", "ddsp_urmp_vn_12k.pt"), ("trumpet", "ddsp_urmp_tpt_12k.pt")]:
        if (CKPT / ck).exists():
            foreign[fname] = load_ref_model(CKPT / ck)
        else:
            print(f"[warn] {ck} missing, skipping the {fname} mismatched reference")

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
        print(f"\n=== LOTO fold {fold}: held out {held} | training on {len(train_files)} clips ===")
        data = build_dataset(train_files, cache=None, verbose=False)
        model, _losses, best_val, _vi = train(data, steps=6000, verbose=False)
        model = model.cpu().eval()
        print(f"  training done best_val={best_val:.2f}")
        eval_files(model, held, f"loto{fold}", loto_rows)

    # ---- LOGO: leave-one-glass-out (instrument-level generalisation baseline) ----
    logo_rows = []
    for gname, names in GLASSES.items():
        train_files = [
            str(TRAIN_DIR / n) for g2, ns in GLASSES.items() if g2 != gname for n in ns
        ] + [str(TRAIN_DIR / s) for s in SINGLETONS]
        print(f"\n=== LOGO: glass held out {gname} | training on {len(train_files)} clips ===")
        data = build_dataset(train_files, cache=None, verbose=False)
        model, _losses, best_val, _vi = train(data, steps=6000, verbose=False)
        model = model.cpu().eval()
        print(f"  training done best_val={best_val:.2f}")
        eval_files(model, names, f"logo_{gname}", logo_rows)

    def med(rows, k):
        return float(np.median([r[k] for r in rows]))

    out_res = {}
    ref_keys = ["own"] + list(foreign)
    for tag, rows in [("loto", loto_rows), ("logo", logo_rows)]:
        print(f"\n=== {tag.upper()} summary (n={len(rows)}) ===")
        for k in ref_keys:
            print(
                f"  {k:>8s}: median spectral loss {med(rows, f'msl_{k}'):.2f}  "
                f"median MFCC {med(rows, f'mfcc_{k}'):.1f}"
            )
        wins = (
            sum(1 for r in rows if r["msl_own"] < min(r[f"msl_{fn}"] for fn in foreign))
            if foreign
            else 0
        )
        print(f"  own has the best spectral loss: {wins}/{len(rows)}")
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
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
