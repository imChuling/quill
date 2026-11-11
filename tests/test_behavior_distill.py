import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from neural.behavior_distill import (
    CHECKPOINT_SCHEMA,
    EXPRESSION_KEYS,
    NOTE_KEYS,
    RAW_SCHEMA,
    DistilledBehavior,
    PrototypeRetriever,
    ResidualBehaviorNet,
    raw_teacher_to_dataset,
    validate_raw_teacher,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_mididdsp_teacher import _source_seed, _valid_raw


def _raw_fixture(path):
    frames = 100
    t = np.arange(frames) / 250.0
    f0 = 440.0 * 2.0 ** (20.0 * np.sin(2 * np.pi * 6 * t) / 1200.0)
    amp = np.exp(0.2 * np.sin(2 * np.pi * 2 * t))
    harmonic = np.tile(np.asarray([0.7, 0.2, 0.1], np.float32), (frames, 1))
    harmonic[:, 1] += np.linspace(0, 0.1, frames)
    noise = np.tile(np.asarray([0.1, 0.05], np.float32), (frames, 1))
    notes = np.zeros((2, len(NOTE_KEYS)), np.float32)
    notes[:, :4] = [[69, 0, 50, 50], [69, 50, 100, 50]]
    notes[:, 4:] = 0.1
    notes[1, 4 + EXPRESSION_KEYS.index("vibrato")] = 0.0
    np.savez_compressed(
        path,
        schema=np.asarray(RAW_SCHEMA),
        frame_rate=np.asarray(250.0, np.float32),
        note_keys=np.asarray(NOTE_KEYS),
        note_table=notes,
        f0_hz=f0[:, None],
        amplitudes=amp[:, None],
        harmonic_distribution=harmonic,
        noise_magnitudes=noise,
        meta=np.asarray(json.dumps({"source_midi": "fixture.mid"})),
    )


def test_raw_teacher_conversion_is_direct_and_finite(tmp_path):
    raw = tmp_path / "teacher_raw.npz"
    out = tmp_path / "teacher.npz"
    _raw_fixture(raw)
    validated = validate_raw_teacher(raw)
    assert validated["f0_hz"].shape == (100, 1)

    result = raw_teacher_to_dataset(raw, out)
    assert result["features"].shape == (2, 7)
    assert result["trajectories"].shape == (2, 4, 48)
    assert result["confidence"].shape == (2, 4)
    assert np.isfinite(result["trajectories"]).all()
    assert result["features"][:, 6].tolist() == [1.0, 0.0]
    # Centering happens before interpolation, matching analysis.tplib; resampling
    # can move the discrete median by a fraction of a cent.
    med = np.abs(np.median(result["trajectories"], axis=2))
    assert np.max(med[:, 0]) < 0.5
    assert np.max(med[:, 1:]) < 1e-3
    assert out.exists()


def test_raw_teacher_rejects_mismatched_control_lengths(tmp_path):
    raw = tmp_path / "teacher_raw.npz"
    _raw_fixture(raw)
    with np.load(raw, allow_pickle=False) as d:
        values = {key: d[key] for key in d.files}
    values["noise_magnitudes"] = values["noise_magnitudes"][:-1]
    np.savez_compressed(raw, **values)
    with pytest.raises(ValueError, match="lengths disagree"):
        validate_raw_teacher(raw)


def test_teacher_source_seed_is_content_stable(tmp_path):
    midi = tmp_path / "score.mid"
    midi.write_bytes(b"first score bytes")
    seed = _source_seed(midi, 42)
    assert seed == _source_seed(midi, 42)
    assert seed != _source_seed(midi, 43)
    midi.write_bytes(b"changed score bytes")
    assert seed != _source_seed(midi, 42)


def test_batch_resume_accepts_only_complete_raw_contract(tmp_path):
    raw = tmp_path / "teacher_raw.npz"
    _raw_fixture(raw)
    assert _valid_raw(raw)
    with np.load(raw, allow_pickle=False) as d:
        values = {key: d[key] for key in d.files if key != "noise_magnitudes"}
    np.savez_compressed(raw, **values)
    assert not _valid_raw(raw)


def _zero_checkpoint(path, features, trajectories, hidden=16):
    model = ResidualBehaviorNet(hidden=hidden)
    for param in model.parameters():
        torch.nn.init.zeros_(param)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": model.state_dict(),
            "hidden": hidden,
            "feat_mean": torch.zeros(6),
            "feat_std": torch.ones(6),
            "traj_scale": torch.ones(4),
            "residual_scale": torch.ones(4),
            "residual_clip": torch.ones(4) * 100,
            "prototype_features": torch.from_numpy(features),
            "prototype_trajectories": torch.from_numpy(trajectories),
            "seed": 42,
        },
        path,
    )


def test_distilled_behavior_zero_residual_preserves_conditioned_retrieval(tmp_path):
    features = np.asarray(
        [
            [60, 1, 0, 0, 1, 0, 0],
            [60, 1, 0, 0, 1, 0, 1],
        ],
        np.float32,
    )
    trajectories = np.zeros((2, 4, 48), np.float32)
    trajectories[0, 0] = 3.0
    trajectories[1, 0] = 7.0
    ckpt = tmp_path / "student.pt"
    _zero_checkpoint(ckpt, features, trajectories)
    model = DistilledBehavior(ckpt)
    query = np.asarray([60, 1, 0, 0, 1, 0, 1], np.float32)
    assert np.array_equal(model.predict(query), trajectories[1])


def test_distilled_behavior_uses_existing_renderer_contract(tmp_path):
    from neural.perform import control_from_behavior

    features = np.asarray(
        [
            [60, 0.5, 1, 0, 1, 0, 0],
            [62, 0.5, 0, 1, 0, 1, 0],
        ],
        np.float32,
    )
    trajectories = np.zeros((2, 4, 48), np.float32)
    trajectories[:, 1] = np.linspace(-0.1, 0.1, 48)
    ckpt = tmp_path / "student.pt"
    _zero_checkpoint(ckpt, features, trajectories)
    model = DistilledBehavior(ckpt)
    f0, loud = control_from_behavior(
        [{"midi": 60, "dur": 0.5, "art": 0}, {"midi": 62, "dur": 0.5, "art": 0}],
        model,
        pitch_glide=False,
        envelope="sustain",
    )
    assert f0.shape == loud.shape
    assert f0.shape[1] > 100
    assert torch.isfinite(f0).all()
    assert torch.isfinite(loud).all()


def test_prototype_retriever_falls_back_when_art_pool_absent():
    features = np.asarray([[60, 1, 0, 0, 1, 1, 0]], np.float32)
    trajectories = np.zeros((1, 4, 48), np.float32)
    trajectories[0, 1] = 2.0
    model = PrototypeRetriever(features, trajectories)
    query = np.asarray([61, 1, 0, 0, 1, 1, 2], np.float32)
    assert np.array_equal(model.predict(query), trajectories[0])


def test_training_cli_smoke_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    n = 12
    features = np.zeros((n, 7), np.float32)
    features[:, 0] = np.linspace(60, 71, n)
    features[:, 1] = 0.5
    features[:, 6] = np.arange(n) % 2
    prototype_traj = rng.normal(size=(n, 4, 48)).astype(np.float32) * 0.05
    target_traj = prototype_traj + 0.01
    confidence = np.ones((n, 4), np.float32)
    teacher = tmp_path / "teacher.npz"
    prototype = tmp_path / "prototype.npz"
    output = tmp_path / "student.pt"
    np.savez_compressed(
        teacher,
        features=features,
        trajectories=target_traj,
        confidence=confidence,
        groups=np.full(n, "one_piece"),
    )
    np.savez_compressed(prototype, features=features, trajectories=prototype_traj)
    script = Path(__file__).resolve().parents[1] / "tools" / "train_distilled_behavior.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--teacher",
            str(teacher),
            "--prototype",
            str(prototype),
            "--output",
            str(output),
            "--epochs",
            "2",
            "--batch",
            "4",
            "--hidden",
            "16",
            "--allow-note-split",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert output.with_suffix(".json").exists()
    model = DistilledBehavior(output)
    assert model.predict(features[0]).shape == (4, 48)
