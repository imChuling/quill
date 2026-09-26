"""MIDI-DDSP teacher controls -> Quill trajectories -> residual student.

Plan D keeps the two systems on opposite sides of a small, versioned ``.npz``
contract.  MIDI-DDSP (TensorFlow/Python 3.8) is only needed by the exporter;
Quill training and inference remain NumPy/PyTorch.

The student predicts a correction to a real retrieved trajectory rather than a
trajectory from scratch.  This preserves Quill's non-parametric prior while
allowing MIDI-DDSP's expression generator to teach systematic refinements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


RAW_SCHEMA = "quill.mididdsp_teacher.raw.v1"
DATASET_SCHEMA = "quill.behavior_teacher.v1"
CHECKPOINT_SCHEMA = "quill.behavior_distilled.v1"

FRAME_RATE = 250.0
TRAJ_C = 4
TRAJ_T = 48
TRAJ_KEYS = ("pitch", "energy", "brightness", "noise")
EXPRESSION_KEYS = (
    "volume", "vol_fluc", "vibrato", "brightness", "attack", "vol_peak_pos",
)
NOTE_KEYS = ("pitch", "onset", "offset", "note_length") + EXPRESSION_KEYS
COND_DIM = 9  # six normalized note features + three-way articulation one-hot


def _scalar_string(value) -> str:
    """Read a string saved either as a Python scalar or a NumPy 0-d array."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError("expected one string value")
    return str(arr.reshape(()).item())


def _time_matrix(value, name: str) -> np.ndarray:
    """Canonicalize a MIDI-DDSP control to ``[time, channels]``."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"{name} must have shape [time, channels], got {arr.shape}")
    return arr


def validate_raw_teacher(path) -> dict[str, object]:
    """Load and strictly validate an exporter artifact.

    Returns canonical arrays so downstream code never needs TensorFlow or
    pandas.  Object arrays/pickles are intentionally not accepted.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        required = {
            "schema", "frame_rate", "note_table", "note_keys", "f0_hz",
            "amplitudes", "harmonic_distribution", "noise_magnitudes", "meta",
        }
        missing = sorted(required.difference(d.files))
        if missing:
            raise ValueError(f"raw teacher file is missing: {', '.join(missing)}")
        schema = _scalar_string(d["schema"])
        if schema != RAW_SCHEMA:
            raise ValueError(f"unsupported raw teacher schema: {schema!r}")
        frame_rate = float(np.asarray(d["frame_rate"]).reshape(()))
        if not np.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        note_keys = tuple(str(x) for x in np.asarray(d["note_keys"]).tolist())
        if note_keys != NOTE_KEYS:
            raise ValueError(f"note_keys must be {NOTE_KEYS}, got {note_keys}")
        note_table = np.asarray(d["note_table"], dtype=np.float32)
        if note_table.ndim != 2 or note_table.shape[1] != len(NOTE_KEYS):
            raise ValueError(
                f"note_table must have shape [notes, {len(NOTE_KEYS)}], "
                f"got {note_table.shape}")
        if len(note_table) == 0:
            raise ValueError("raw teacher file contains no notes")
        if not np.isfinite(note_table).all():
            raise ValueError("note_table contains non-finite values")
        if np.any(note_table[:, 2] <= note_table[:, 1]):
            raise ValueError("every note must have offset > onset")

        controls = {
            name: _time_matrix(d[name], name)
            for name in ("f0_hz", "amplitudes", "harmonic_distribution",
                         "noise_magnitudes")
        }
        lengths = {name: value.shape[0] for name, value in controls.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"teacher control lengths disagree: {lengths}")
        meta_text = _scalar_string(d["meta"])
        try:
            meta = json.loads(meta_text)
        except json.JSONDecodeError as exc:
            raise ValueError("meta must be valid JSON") from exc
    return {
        "path": path,
        "frame_rate": frame_rate,
        "note_table": note_table,
        "meta": meta,
        **controls,
    }


def _resample_residual(values, valid=None) -> tuple[np.ndarray, float]:
    """Median-center a time series and resample it to Quill's 48 points."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(values)
    if valid.sum() < 2:
        return np.zeros(TRAJ_T, np.float32), 0.0
    x = np.arange(len(values), dtype=np.float64)
    filled = np.interp(x, x[valid], values[valid])
    filled -= np.median(filled[valid])
    out = np.interp(np.linspace(0, len(values) - 1, TRAJ_T), x, filled)
    return out.astype(np.float32), float(valid.mean())


def raw_teacher_to_dataset(raw_path, output_path=None, vibrato_threshold=0.05,
                           phrase_gap=0.5) -> dict[str, object]:
    """Convert direct MIDI-DDSP controls to the normal Quill library schema.

    ``pitch`` and ``energy`` match Quill's residual definitions exactly.
    ``brightness`` is the log2 harmonic-centroid residual.  ``noise`` is the
    log10 RMS residual of DDSP's filtered-noise magnitudes, a documented proxy
    rather than waveform spectral flatness.  The current Quill renderer consumes
    pitch and energy; the other two channels are retained for future decoders.
    """
    raw = validate_raw_teacher(raw_path)
    table = raw["note_table"]
    frame_rate = float(raw["frame_rate"])
    f0 = raw["f0_hz"][:, 0]
    amp = np.maximum(raw["amplitudes"][:, 0], 1e-8)
    harmonic = np.maximum(raw["harmonic_distribution"], 0.0)
    noise = raw["noise_magnitudes"]
    n_frames = len(f0)

    # Conditioning rows may contain rests.  Exporter normally removes them, but
    # retaining this guard makes hand-built fixtures and older dumps safe.
    table = table[table[:, 0] > 0]
    if len(table) == 0:
        raise ValueError("raw teacher file contains no pitched notes")
    order = np.argsort(table[:, 1], kind="stable")
    table = table[order]

    features = []
    trajectories = []
    confidence = []
    note_meta = []
    for i, row in enumerate(table):
        midi, onset, offset = (float(row[0]), float(row[1]), float(row[2]))
        start = max(0, int(round(onset)))
        stop = min(n_frames, int(round(offset)))
        if stop - start < 2:
            raise ValueError(
                f"note {i} [{onset}, {offset}] has no usable teacher frames")

        base_hz = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
        f0_seg = f0[start:stop]
        f0_valid = np.isfinite(f0_seg) & (f0_seg > 0)
        pitch_cents = 1200.0 * np.log2(np.maximum(f0_seg, 1e-8) / base_hz)
        pitch, pitch_conf = _resample_residual(pitch_cents, f0_valid)

        energy, energy_conf = _resample_residual(np.log(amp[start:stop]))

        h = harmonic[start:stop]
        harmonic_number = np.arange(1, h.shape[1] + 1, dtype=np.float64)
        h_sum = h.sum(axis=1)
        h_centroid = (h * harmonic_number[None]).sum(axis=1) / np.maximum(h_sum, 1e-8)
        brightness, brightness_conf = _resample_residual(
            np.log2(np.maximum(h_centroid, 1e-8)), h_sum > 1e-8)

        noise_seg = noise[start:stop]
        noise_rms = np.sqrt(np.mean(np.square(noise_seg), axis=1) + 1e-12)
        noise_traj, noise_conf = _resample_residual(np.log10(noise_rms + 1e-8))

        prev_gap = ((onset - float(table[i - 1, 2])) / frame_rate
                    if i > 0 else 1.0)
        next_gap = ((float(table[i + 1, 1]) - offset) / frame_rate
                    if i + 1 < len(table) else 1.0)
        prev_gap = max(0.0, prev_gap)
        next_gap = max(0.0, next_gap)
        expression = row[4:].astype(np.float32)
        vibrato = float(expression[EXPRESSION_KEYS.index("vibrato")])
        art = 1 if vibrato >= vibrato_threshold else 0
        is_first = i == 0 or prev_gap > phrase_gap
        is_last = i == len(table) - 1 or next_gap > phrase_gap
        dur = (offset - onset) / frame_rate
        features.append([midi, dur, prev_gap, next_gap, is_first, is_last, art])
        trajectories.append([pitch, energy, brightness, noise_traj])
        confidence.append([pitch_conf, energy_conf, brightness_conf, noise_conf])
        note_meta.append({
            "note_idx": i,
            "onset_frame": onset,
            "offset_frame": offset,
            "teacher_expression": {
                key: float(value) for key, value in zip(EXPRESSION_KEYS, expression)
            },
        })

    source_meta = dict(raw["meta"])
    source_id = str(source_meta.get("source_midi", Path(raw_path).stem))
    result = {
        "schema": DATASET_SCHEMA,
        "features": np.asarray(features, dtype=np.float32),
        "trajectories": np.asarray(trajectories, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "groups": np.full(len(features), source_id, dtype=f"U{max(1, len(source_id))}"),
        "expression": table[:, 4:].astype(np.float32),
        "expression_keys": np.asarray(EXPRESSION_KEYS),
        "traj_keys": np.asarray(TRAJ_KEYS),
        "meta": json.dumps({
            "schema": DATASET_SCHEMA,
            "teacher": "MIDI-DDSP expression+synthesis generators",
            "source": source_meta,
            "vibrato_threshold": float(vibrato_threshold),
            "noise_definition": "log10 RMS of DDSP noise magnitudes (proxy)",
            "notes": note_meta,
        }, ensure_ascii=False),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **result)
    return result


def load_teacher_dataset(path) -> dict[str, np.ndarray]:
    """Load either a converted Plan-D dataset or a normal Quill style library."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        required = {"features", "trajectories", "confidence"}
        missing = sorted(required.difference(d.files))
        if missing:
            raise ValueError(f"teacher dataset is missing: {', '.join(missing)}")
        features = np.asarray(d["features"], dtype=np.float32)
        trajectories = np.asarray(d["trajectories"], dtype=np.float32)
        confidence = np.asarray(d["confidence"], dtype=np.float32)
        groups = (np.asarray(d["groups"]).astype(str) if "groups" in d.files
                  else np.full(len(features), path.stem))
    if features.ndim != 2 or features.shape[1] != 7:
        raise ValueError(f"features must have shape [notes, 7], got {features.shape}")
    if trajectories.shape != (len(features), TRAJ_C, TRAJ_T):
        raise ValueError(
            f"trajectories must have shape [notes, 4, 48], got {trajectories.shape}")
    if confidence.shape != (len(features), TRAJ_C):
        raise ValueError(f"confidence must have shape [notes, 4], got {confidence.shape}")
    if groups.shape != (len(features),):
        raise ValueError(f"groups must have shape [notes], got {groups.shape}")
    if not (np.isfinite(features).all() and np.isfinite(trajectories).all()
            and np.isfinite(confidence).all()):
        raise ValueError("teacher dataset contains non-finite values")
    return {
        "features": features,
        "trajectories": trajectories,
        "confidence": np.clip(confidence, 0.0, 1.0),
        "groups": groups,
    }


class PrototypeRetriever:
    """Conditioned nearest-neighbour prior used by both training and inference."""

    use_art = True

    def __init__(self, features, trajectories):
        features = np.asarray(features, dtype=np.float32)
        trajectories = np.asarray(trajectories, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] < 6:
            raise ValueError("prototype features must have at least six columns")
        if trajectories.shape != (len(features), TRAJ_C, TRAJ_T):
            raise ValueError("prototype trajectories must have shape [notes, 4, 48]")
        if len(features) == 0:
            raise ValueError("prototype library is empty")
        self.features = features
        self.trajectories = trajectories
        self.has_art = features.shape[1] >= 7
        self.base = features[:, :6]
        self.std = np.maximum(self.base.std(axis=0), 1e-6)
        self.norm = self.base / self.std
        if self.has_art:
            art = np.rint(features[:, 6]).astype(np.int64)
            self.pools = {a: np.flatnonzero(art == a) for a in np.unique(art)}
        else:
            self.pools = {}

    def predict(self, feat) -> np.ndarray:
        feat = np.asarray(feat, dtype=np.float32)
        if feat.shape[0] < 6:
            raise ValueError("query feature must have at least six values")
        if self.has_art and feat.shape[0] >= 7:
            art = int(round(float(feat[6])))
            pool = self.pools.get(art)
        else:
            pool = None
        if pool is None or len(pool) == 0:
            pool = np.arange(len(self.features))
        dist = np.linalg.norm(self.norm[pool] - feat[:6] / self.std, axis=1)
        return self.trajectories[pool[int(np.argmin(dist))]].copy()

    def predict_batch(self, features) -> np.ndarray:
        return np.stack([self.predict(feat) for feat in features])


class ResidualBehaviorNet(nn.Module):
    """Small MLP that predicts a normalized correction to a retrieved prototype."""

    def __init__(self, hidden=256):
        super().__init__()
        self.hidden = int(hidden)
        size = TRAJ_C * TRAJ_T
        self.net = nn.Sequential(
            nn.Linear(COND_DIM + size, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, size),
        )

    def forward(self, cond, prototype):
        x = torch.cat([cond, prototype.flatten(start_dim=1)], dim=1)
        return self.net(x).reshape(-1, TRAJ_C, TRAJ_T)


def condition_features(features, mean, std) -> np.ndarray:
    """Turn Quill's seven fields into six z-scores plus articulation one-hot."""
    features = np.asarray(features, dtype=np.float32)
    one = features.ndim == 1
    if one:
        features = features[None]
    if features.shape[1] < 6:
        raise ValueError("features must have at least six columns")
    base = (features[:, :6] - np.asarray(mean)) / np.asarray(std)
    art = (np.rint(features[:, 6]).astype(int) if features.shape[1] >= 7
           else np.zeros(len(features), dtype=int))
    onehot = np.eye(3, dtype=np.float32)[np.clip(art, 0, 2)]
    result = np.concatenate([base, onehot], axis=1).astype(np.float32)
    return result[0] if one else result


class DistilledBehavior:
    """Self-contained Plan-D checkpoint with the normal ``predict`` interface."""

    use_art = True

    def __init__(self, checkpoint, device="cpu", residual_strength=1.0):
        ck = torch.load(str(checkpoint), map_location=device, weights_only=True)
        if ck.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported distilled checkpoint: {ck.get('schema')!r}")
        self.device = device
        self.model = ResidualBehaviorNet(hidden=int(ck["hidden"])).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.feat_mean = ck["feat_mean"].cpu().numpy().astype(np.float32)
        self.feat_std = ck["feat_std"].cpu().numpy().astype(np.float32)
        self.traj_scale = ck["traj_scale"].cpu().numpy().astype(np.float32)
        self.residual_scale = ck["residual_scale"].cpu().numpy().astype(np.float32)
        self.residual_clip = ck["residual_clip"].cpu().numpy().astype(np.float32)
        proto_features = ck["prototype_features"].cpu().numpy()
        proto_trajectories = ck["prototype_trajectories"].cpu().numpy()
        self.retriever = PrototypeRetriever(proto_features, proto_trajectories)
        self.residual_strength = float(residual_strength)

    @torch.no_grad()
    def predict(self, feat) -> np.ndarray:
        feat = np.asarray(feat, dtype=np.float32)
        prototype = self.retriever.predict(feat)
        cond = condition_features(feat, self.feat_mean, self.feat_std)
        cond_t = torch.from_numpy(cond[None]).to(self.device)
        proto_t = torch.from_numpy(
            prototype[None] / self.traj_scale[None, :, None]).to(self.device)
        residual = self.model(cond_t, proto_t).cpu().numpy()[0]
        residual = residual * self.residual_scale[:, None]
        residual = np.clip(
            residual, -self.residual_clip[:, None], self.residual_clip[:, None])
        return (prototype + self.residual_strength * residual).astype(np.float32)

    def predict_batch(self, features) -> np.ndarray:
        return np.stack([self.predict(feat) for feat in features])

