#!/usr/bin/env python3
"""Train Plan D: retrieval prototype + MIDI-DDSP-taught residual."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from neural.behavior_distill import (
    CHECKPOINT_SCHEMA,
    PrototypeRetriever,
    ResidualBehaviorNet,
    condition_features,
    load_teacher_dataset,
)


def _load_many(paths):
    datasets = [load_teacher_dataset(path) for path in paths]
    features = np.concatenate([d["features"] for d in datasets])
    trajectories = np.concatenate([d["trajectories"] for d in datasets])
    confidence = np.concatenate([d["confidence"] for d in datasets])
    # Prefix group names by artifact to prevent accidental collisions.
    groups = np.concatenate(
        [
            np.asarray([f"{Path(path).stem}:{g}" for g in d["groups"]])
            for path, d in zip(paths, datasets)
        ]
    )
    return features, trajectories, confidence, groups


def _split(groups, seed, val_fraction, allow_note_split):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    if len(unique) >= 2:
        shuffled = rng.permutation(unique)
        n_val = max(1, min(len(unique) - 1, int(round(len(unique) * val_fraction))))
        val_groups = set(shuffled[:n_val].tolist())
        val = np.asarray([g in val_groups for g in groups])
        return np.flatnonzero(~val), np.flatnonzero(val), "source-group"
    if not allow_note_split:
        raise ValueError(
            "scientific training needs at least two teacher source groups. "
            "Pass multiple --teacher files, or --allow-note-split for a smoke test only."
        )
    order = rng.permutation(len(groups))
    n_val = max(1, min(len(groups) - 1, int(round(len(groups) * val_fraction))))
    return order[n_val:], order[:n_val], "note (development-only)"


def _weighted_mse(pred, target, confidence):
    weight = confidence[:, :, None]
    return float(
        (np.square(pred - target) * weight).sum() / max(float(weight.sum()) * pred.shape[2], 1e-8)
    )


def _weighted_mse_channels(pred, target, confidence):
    squared = np.square(pred - target)
    numerator = (squared * confidence[:, :, None]).sum(axis=(0, 2))
    denominator = np.maximum(confidence.sum(axis=0) * pred.shape[2], 1e-8)
    return numerator / denominator


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="distill MIDI-DDSP behavior into Quill")
    parser.add_argument(
        "--teacher",
        type=Path,
        action="append",
        required=True,
        help="converted teacher .npz; repeat for each independent MIDI source",
    )
    parser.add_argument(
        "--prototype", type=Path, required=True, help="Quill retrieval library (.npz)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-note-split", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("--epochs and --batch must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between zero and one")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    features, target, confidence, groups = _load_many(args.teacher)
    with np.load(args.prototype, allow_pickle=False) as p:
        proto_features = np.asarray(p["features"], dtype=np.float32)
        proto_trajectories = np.asarray(p["trajectories"], dtype=np.float32)
    retriever = PrototypeRetriever(proto_features, proto_trajectories)
    prototype = retriever.predict_batch(features)
    residual = target - prototype

    train_idx, val_idx, split_kind = _split(
        groups, args.seed, args.val_fraction, args.allow_note_split
    )
    feat_mean = features[train_idx, :6].mean(axis=0)
    feat_std = np.maximum(features[train_idx, :6].std(axis=0), 1e-6)
    traj_scale = np.maximum(
        prototype[train_idx].std(axis=(0, 2)), np.asarray([1.0, 0.01, 0.01, 0.01])
    )
    residual_scale = np.maximum(
        residual[train_idx].std(axis=(0, 2)), np.asarray([1.0, 0.01, 0.01, 0.01])
    )
    residual_clip = np.maximum(
        np.percentile(np.abs(residual[train_idx]), 99.5, axis=(0, 2)), residual_scale
    )

    cond = condition_features(features, feat_mean, feat_std)
    proto_norm = prototype / traj_scale[None, :, None]
    residual_norm = residual / residual_scale[None, :, None]
    cond_t = torch.from_numpy(cond).float()
    proto_t = torch.from_numpy(proto_norm).float()
    target_t = torch.from_numpy(residual_norm).float()
    conf_t = torch.from_numpy(confidence[:, :, None]).float()

    model = ResidualBehaviorNet(args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), args.batch):
            idx = order[start : start + args.batch]
            pred = model(cond_t[idx].to(device), proto_t[idx].to(device))
            weight = conf_t[idx].to(device)
            loss = ((pred - target_t[idx].to(device)).square() * weight).sum()
            loss = loss / torch.clamp(weight.sum() * pred.shape[-1], min=1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(cond_t[val_idx].to(device), proto_t[val_idx].to(device))
            weight = conf_t[val_idx].to(device)
            val_loss = ((pred - target_t[val_idx].to(device)).square() * weight).sum()
            val_loss = float(
                (val_loss / torch.clamp(weight.sum() * pred.shape[-1], min=1e-8)).item()
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
            print(f"epoch {epoch + 1:4d}  val_normalized_mse={val_loss:.6f}  best={best_loss:.6f}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_norm = model(cond_t[val_idx].to(device), proto_t[val_idx].to(device))
        pred_residual = pred_norm.cpu().numpy() * residual_scale[None, :, None]
    pred_residual = np.clip(
        pred_residual,
        -residual_clip[None, :, None],
        residual_clip[None, :, None],
    )
    student = prototype[val_idx] + pred_residual
    retrieval_mse = _weighted_mse(prototype[val_idx], target[val_idx], confidence[val_idx])
    student_mse = _weighted_mse(student, target[val_idx], confidence[val_idx])
    retrieval_channels = _weighted_mse_channels(
        prototype[val_idx], target[val_idx], confidence[val_idx]
    )
    student_channels = _weighted_mse_channels(student, target[val_idx], confidence[val_idx])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "model": best_state,
        "hidden": int(args.hidden),
        "feat_mean": torch.from_numpy(feat_mean.astype(np.float32)),
        "feat_std": torch.from_numpy(feat_std.astype(np.float32)),
        "traj_scale": torch.from_numpy(traj_scale.astype(np.float32)),
        "residual_scale": torch.from_numpy(residual_scale.astype(np.float32)),
        "residual_clip": torch.from_numpy(residual_clip.astype(np.float32)),
        "prototype_features": torch.from_numpy(proto_features),
        "prototype_trajectories": torch.from_numpy(proto_trajectories),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(args.output)
    metrics = {
        "schema": CHECKPOINT_SCHEMA,
        "teacher_files": [str(path) for path in args.teacher],
        "prototype": str(args.prototype),
        "notes": int(len(features)),
        "train_notes": int(len(train_idx)),
        "val_notes": int(len(val_idx)),
        "split": split_kind,
        "train_groups": sorted(set(groups[train_idx].tolist())),
        "val_groups": sorted(set(groups[val_idx].tolist())),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "epochs_requested": int(args.epochs),
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "teacher_sha256": {str(path): _sha256(path) for path in args.teacher},
        "prototype_sha256": _sha256(args.prototype),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "retrieval_val_mse": retrieval_mse,
        "student_val_mse": student_mse,
        "mse_improvement_ratio": retrieval_mse / max(student_mse, 1e-12),
        "retrieval_val_mse_by_channel": {
            key: float(value)
            for key, value in zip(("pitch", "energy", "brightness", "noise"), retrieval_channels)
        },
        "student_val_mse_by_channel": {
            key: float(value)
            for key, value in zip(("pitch", "energy", "brightness", "noise"), student_channels)
        },
        "mse_improvement_by_channel": {
            key: float(base / max(student, 1e-12))
            for key, base, student in zip(
                ("pitch", "energy", "brightness", "noise"), retrieval_channels, student_channels
            )
        },
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved checkpoint -> {args.output}")
    print(f"saved metrics    -> {metrics_path}")


if __name__ == "__main__":
    main()
