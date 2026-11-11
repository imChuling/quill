"""C3 验收测试：behavior_player 与 control_from_behavior 逐帧一致。

同一个库、同一条乐谱（含 plain / vibrato / slide），两边产出的
f0_hz、loudness_z、bright、noise 最大偏差 < 0.01 cent / 0.001。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUILL = Path(__file__).resolve().parent.parent
LIB_PATH = QUILL / "checkpoints" / "urmp_vn_trajectories.npz"

pytestmark = pytest.mark.skipif(
    not LIB_PATH.exists(), reason="urmp_vn_trajectories.npz not available"
)

SCORES = {
    "plain_only": [
        (62, 0.8, 0), (65, 0.4, 0), (67, 1.2, 0), (62, 0.6, 0),
    ],
    "vibrato_only": [
        (62, 0.8, 1), (65, 0.4, 1), (67, 1.2, 1), (62, 0.6, 1),
    ],
    "mixed_with_slide": [
        (62, 0.8, 1), (65, 0.4, 0), (67, 1.2, 1), (65, 0.4, 0),
        (69, 0.4, 2), (72, 1.6, 1), (67, 0.8, 0), (62, 1.6, 1),
    ],
    "slide_sequence": [
        (60, 0.6, 0), (65, 0.8, 2), (69, 0.6, 2), (72, 1.0, 1),
    ],
}


def _engine_control(score_tuples, lib_path, envelope="sustain"):
    """Run engine's control_from_behavior (returns numpy)."""
    from neural.neural_take import ConditionedNN
    from neural.perform import control_from_behavior

    d = np.load(str(lib_path), allow_pickle=True)
    nn = ConditionedNN(d["features"], d["trajectories"])
    notes = [{"midi": m, "dur": d_, "art": a} for m, d_, a in score_tuples]
    f0_t, loud_t, bright, noise = control_from_behavior(
        notes, nn, pitch_glide=True, envelope=envelope, traj_gain=1.0,
        concat_lambda=0.0, contiguity=0.0, label_processing="full",
        extra_dims=True,
    )
    return f0_t.numpy()[0, :, 0], loud_t.numpy()[0, :, 0], bright, noise


def _player_control(score_tuples, lib_path, envelope="sustain"):
    """Run player's play function (returns numpy)."""
    from tools.behavior_player import BehaviorLibrary, play

    lib = BehaviorLibrary(lib_path)
    f0_hz, loud_z, bright, noise = play(
        list(score_tuples), lib, pitch_glide=True, envelope=envelope,
        traj_gain=1.0, label_processing="full",
    )
    return f0_hz, loud_z, bright, noise


class TestPlayerEngineParity:
    """Player and engine must produce identical control signals."""

    @pytest.mark.parametrize("name", list(SCORES.keys()))
    def test_f0_parity_sustain(self, name):
        score = SCORES[name]
        f0_eng, _, _, _ = _engine_control(score, LIB_PATH, "sustain")
        f0_ply, _, _, _ = _player_control(score, LIB_PATH, "sustain")
        n = min(len(f0_eng), len(f0_ply))
        assert abs(len(f0_eng) - len(f0_ply)) <= 1, (
            f"length mismatch: engine {len(f0_eng)} vs player {len(f0_ply)}"
        )
        cents_err = 1200.0 * np.abs(np.log2(f0_ply[:n] / (f0_eng[:n] + 1e-12) + 1e-12))
        max_err = float(np.nanmax(cents_err))
        assert max_err < 0.01, f"f0 max error {max_err:.4f} cents in '{name}'"

    @pytest.mark.parametrize("name", list(SCORES.keys()))
    def test_loud_parity_sustain(self, name):
        score = SCORES[name]
        _, loud_eng, _, _ = _engine_control(score, LIB_PATH, "sustain")
        _, loud_ply, _, _ = _player_control(score, LIB_PATH, "sustain")
        n = min(len(loud_eng), len(loud_ply))
        assert abs(len(loud_eng) - len(loud_ply)) <= 1
        max_err = float(np.max(np.abs(loud_ply[:n] - loud_eng[:n])))
        assert max_err < 0.001, f"loud max error {max_err:.6f} in '{name}'"

    @pytest.mark.parametrize("name", list(SCORES.keys()))
    def test_bright_parity_sustain(self, name):
        score = SCORES[name]
        _, _, br_eng, _ = _engine_control(score, LIB_PATH, "sustain")
        _, _, br_ply, _ = _player_control(score, LIB_PATH, "sustain")
        n = min(len(br_eng), len(br_ply))
        assert abs(len(br_eng) - len(br_ply)) <= 1
        max_err = float(np.max(np.abs(br_ply[:n] - br_eng[:n])))
        assert max_err < 0.001, f"bright max error {max_err:.6f} in '{name}'"

    @pytest.mark.parametrize("name", list(SCORES.keys()))
    def test_noise_parity_sustain(self, name):
        score = SCORES[name]
        _, _, _, nz_eng = _engine_control(score, LIB_PATH, "sustain")
        _, _, _, nz_ply = _player_control(score, LIB_PATH, "sustain")
        n = min(len(nz_eng), len(nz_ply))
        assert abs(len(nz_eng) - len(nz_ply)) <= 1
        max_err = float(np.max(np.abs(nz_ply[:n] - nz_eng[:n])))
        assert max_err < 0.001, f"noise max error {max_err:.6f} in '{name}'"

    @pytest.mark.parametrize("name", list(SCORES.keys()))
    def test_f0_parity_pluck(self, name):
        score = SCORES[name]
        f0_eng, _, _, _ = _engine_control(score, LIB_PATH, "pluck")
        f0_ply, _, _, _ = _player_control(score, LIB_PATH, "pluck")
        n = min(len(f0_eng), len(f0_ply))
        assert abs(len(f0_eng) - len(f0_ply)) <= 1
        cents_err = 1200.0 * np.abs(np.log2(f0_ply[:n] / (f0_eng[:n] + 1e-12) + 1e-12))
        max_err = float(np.nanmax(cents_err))
        assert max_err < 0.01, f"f0 max error {max_err:.4f} cents in '{name}' (pluck)"
