"""Train the vibrato model on existing trajectory libraries.

Usage (local, CPU is fine — tiny model):
    python tools/train_vibrato.py

Training data:
  - All vibrato-labeled notes: extract true vibrato params as target
  - All plain-labeled notes: target depth=0 (model must suppress vibrato)
  - Slide notes excluded (different phenomenon)

Saves: checkpoints/vibrato_model.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural.vibrato_model import (
    VibratoMLP,
    extract_vib_params,
    note_to_input,
    params_to_target,
    synth_vibrato,
    target_to_params,
)

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
LIBS = [
    ("violin", CKPT / "urmp_vn_trajectories.npz", 0),
    ("cello", CKPT / "urmp_vc_trajectories.npz", 1),
    ("guitar", CKPT / "guitarset_trajectories.npz", 2),
]
OUT_PATH = CKPT / "vibrato_model.pt"

ART_VIB = 1
ART_PLAIN = 0
EPOCHS = 300
LR = 1e-3
BATCH = 128
PLAIN_SAMPLE_RATIO = 1.0  # include this fraction of plain notes (1.0 = all)


def build_dataset():
    """Load all libraries, extract vibrato params, return (X, Y) tensors."""
    all_x, all_y = [], []

    for name, path, inst_idx in LIBS:
        if not path.exists():
            print(f"  skip {name}: {path} not found")
            continue
        d = np.load(str(path))
        feat = d["features"]
        traj = d["trajectories"]
        if feat.shape[1] < 7:
            print(f"  skip {name}: no articulation labels")
            continue

        art = feat[:, 6].astype(int)
        vib_idx = np.where(art == ART_VIB)[0]
        plain_idx = np.where(art == ART_PLAIN)[0]

        # subsample plain notes
        if PLAIN_SAMPLE_RATIO < 1.0:
            np.random.seed(42)
            plain_idx = np.random.choice(
                plain_idx, int(len(plain_idx) * PLAIN_SAMPLE_RATIO), replace=False
            )

        print(f"  {name}: vib={len(vib_idx)}  plain={len(plain_idx)}")

        for idx in vib_idx:
            f = feat[idx]
            x = note_to_input(f[0], f[1], f[2], f[3], f[4], f[5], f[6], inst_idx)
            params = extract_vib_params(traj[idx, 0], float(f[1]))
            y = params_to_target(params)
            all_x.append(x)
            all_y.append(y)

        for idx in plain_idx:
            f = feat[idx]
            x = note_to_input(f[0], f[1], f[2], f[3], f[4], f[5], f[6], inst_idx)
            y = np.array([5.5 / 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            all_x.append(x)
            all_y.append(y)

    X = torch.tensor(np.stack(all_x))
    Y = torch.tensor(np.stack(all_y))
    return X, Y


def main():
    print("Building dataset...")
    X, Y = build_dataset()
    n = len(X)
    print(f"Total samples: {n}")

    # train/val split (90/10)
    perm = torch.randperm(n)
    n_val = max(1, n // 10)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    model = VibratoMLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    # loss: MSE, but weight depth channel higher (it's the most important param)
    depth_weight = 3.0
    weights = torch.ones(8)
    weights[1] = depth_weight  # depth channel

    best_val = float("inf")
    for ep in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = ((pred - yb) ** 2 * weights).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                val_loss += ((pred - yb) ** 2 * weights).mean().item() * len(xb)
        val_loss /= n_val
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(OUT_PATH))

        if ep % 50 == 0 or ep == 1:
            print(f"  ep {ep:3d}  train={train_loss:.4f}  val={val_loss:.4f}  best={best_val:.4f}")

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Saved: {OUT_PATH}")

    # diagnostic: compare predictions on vibrato vs plain notes
    model.load_state_dict(torch.load(str(OUT_PATH), weights_only=True))
    model.eval()
    with torch.no_grad():
        all_pred = model(X).numpy()

    # find vibrato vs plain by checking if target depth > 0
    vib_mask = Y[:, 1].numpy() > 0.01
    plain_mask = ~vib_mask

    pred_depth = all_pred[:, 1] * 25.0
    pred_rate = all_pred[:, 0] * 10.0

    print("\nPredicted depth (cents):")
    print(
        f"  vibrato notes ({vib_mask.sum()}): mean={pred_depth[vib_mask].mean():.1f}  "
        f"std={pred_depth[vib_mask].std():.1f}"
    )
    print(
        f"  plain notes ({plain_mask.sum()}):   mean={pred_depth[plain_mask].mean():.1f}  "
        f"std={pred_depth[plain_mask].std():.1f}"
    )
    print(
        f"  depth ratio: {pred_depth[vib_mask].mean() / max(pred_depth[plain_mask].mean(), 0.01):.1f}x"
    )

    print("\nPredicted rate (Hz):")
    print(f"  vibrato: mean={pred_rate[vib_mask].mean():.1f}")
    print(f"  plain:   mean={pred_rate[plain_mask].mean():.1f}")

    # render a sample vibrato note for ear check
    sample_vib = np.where(vib_mask)[0][0]
    p = target_to_params(all_pred[sample_vib])
    synth_vibrato(p["rate"], p["depth"], p["onset_ratio"], p["envelope"], 1.0, 250)
    print(
        f"\nSample vibrato prediction: rate={p['rate']:.1f}Hz  depth={p['depth']:.1f}c  "
        f"onset={p['onset_ratio']:.2f}"
    )
    print(f"  envelope: {p['envelope'].round(2)}")


if __name__ == "__main__":
    main()
