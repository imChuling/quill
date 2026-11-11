"""神经层快测(10.15 回归账):连续检索、额外维度、后处理、单声部化、L0 控制驱动接口。

不训练、不开声卡。需要 checkpoint 的用例在文件缺失时跳过。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural.neural_take import (  # noqa: E402
    INSTRUMENTS,
    ConditionedNN,
    NeuralTake,
    available_instruments,
)
from neural.perform import FPS, control_from_behavior  # noqa: E402
from neural.postfx import apply_reverb, apply_tilt  # noqa: E402

HAVE_CKPT = all(s["ckpt"].exists() and s["traj"].exists() for s in INSTRUMENTS.values())
needs_ckpt = pytest.mark.skipif(not HAVE_CKPT, reason="DDSP checkpoints / trajectory banks missing")


def _synthetic_bank(n_tracks=3, notes_per_track=6, seed=0):
    """合成库:features [midi, dur, prev_gap, next_gap, first, last, art],meta 带 track/note_idx。
    轨迹设计成同一 track 相邻音端点连续(pitch 末=下一音首),不同 track 之间不连续。"""
    rng = np.random.default_rng(seed)
    feats, trajs, meta = [], [], []
    for t in range(n_tracks):
        level = 40.0 * t  # 每个 track 自己的 pitch 偏移平台
        for i in range(notes_per_track):
            feats.append([60 + i, 0.5, 0.0 if i else 1.0, 0.0 if i < notes_per_track - 1 else 1.0,
                          1.0 if i == 0 else 0.0, 1.0 if i == notes_per_track - 1 else 0.0, 0.0])
            tr = np.zeros((4, 48), np.float32)
            tr[0] = level + rng.normal(0, 0.5, 48)  # pitch 平台(cents)
            tr[1] = 0.1 * t
            tr[2] = 0.05 * t
            tr[3] = 0.02 * t
            trajs.append(tr)
            meta.append({"track": f"trk{t}", "note_idx": i, "midi": 60 + i, "dur": 0.5, "art": 0})
    return np.array(feats, np.float32), np.stack(trajs), meta


def test_contiguity_prefers_next_note_of_same_track():
    f, tr, meta = _synthetic_bank()
    nn = ConditionedNN(f, tr, meta=meta)
    q0 = np.array([60, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0], np.float32)  # 句首 midi 60:三个 track 都有,距离相同
    nn.last_idx = None
    nn.predict(q0)
    first = nn.last_idx
    t_first = meta[first]["track"]
    q1 = np.array([61, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    # 无连续奖励:只看上下文距离,三个 track 的 61 平手,argmin 取最早的
    prev = nn.trajectories[first]
    nn.predict(q1, prev_traj=prev, concat_lambda=0.0, contiguity=0.0)
    # 有连续奖励:必须选同一 track 的 note_idx+1
    nn.last_idx = first
    nn.predict(q1, prev_traj=prev, concat_lambda=0.0, contiguity=1.0)
    chosen = meta[nn.last_idx]
    assert chosen["track"] == t_first and chosen["note_idx"] == meta[first]["note_idx"] + 1


def test_clone_shares_arrays_but_not_memory():
    f, tr, meta = _synthetic_bank()
    nn = ConditionedNN(f, tr, meta=meta)
    nn.predict(np.array([60, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0], np.float32))
    c = nn.clone()
    assert c.trajectories is nn.trajectories and c._contig_index is nn._contig_index
    assert nn.last_idx is not None and c.last_idx is None
    before = nn.last_idx
    c.predict(np.array([65, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32))
    assert c.last_idx is not None and nn.last_idx == before  # 副本检索不改原件的记忆


def test_concat_lambda_reduces_boundary_cost():
    f, tr, meta = _synthetic_bank()
    nn = ConditionedNN(f, tr, meta=meta)
    # 上一音来自 trk2(pitch 平台 80c);候选 61 三个 track 平手;λ>0 应选端点最接近的 trk2
    prev = nn.trajectories[[i for i, m in enumerate(meta) if m["track"] == "trk2" and m["note_idx"] == 0][0]]
    q1 = np.array([61, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    nn.last_idx = None
    nn.predict(q1, prev_traj=prev, concat_lambda=2.0)
    assert meta[nn.last_idx]["track"] == "trk2"


class _StubBehavior:
    """最小行为模型:predict → 固定 (4,48) 轨迹,供 control_from_behavior 接口测试。"""

    use_art = False

    def predict(self, feat, **kw):
        tr = np.zeros((4, 48), np.float32)
        tr[0] = 10.0 * np.sin(np.linspace(0, 6 * np.pi, 48))  # 3 周期 ±10c
        tr[1] = 0.0
        tr[2] = 0.3  # brightness residual (oct)
        tr[3] = -0.2  # noise residual (log10)
        return tr


def test_control_from_behavior_extra_dims_shapes():
    score = [{"midi": 60, "dur": 0.4, "art": 0}, {"midi": 62, "dur": 0.4, "art": 0}]
    f0, loud = control_from_behavior(score, _StubBehavior(), pitch_glide=False, envelope="sustain")
    f0b, loudb, br, nz = control_from_behavior(
        score, _StubBehavior(), pitch_glide=False, envelope="sustain", extra_dims=True
    )
    T = f0.shape[1]
    assert f0b.shape == f0.shape and loudb.shape == loud.shape
    assert br.shape == (T,) and nz.shape == (T,)
    assert abs(float(np.median(br)) - 0.3) < 0.05 and abs(float(np.median(nz)) + 0.2) < 0.05
    assert pytest.approx(0.8 * FPS, abs=6) == T


def test_postfx_identity_at_neutral():
    y = np.random.default_rng(0).normal(size=4000)
    assert np.array_equal(apply_tilt(y, 16000, 0.5), y)
    assert np.array_equal(apply_reverb(y, 16000, 0.0), y)
    assert not np.array_equal(apply_tilt(y, 16000, 0.9), y)


def test_monophonize_keeps_highest_of_chord():
    notes = [
        {"midi": 60, "start": 0.0, "len": 0.5, "vel": 100, "art": "plain"},
        {"midi": 64, "start": 0.5, "len": 0.5, "vel": 100, "art": "plain"},
        {"midi": 55, "start": 0.5, "len": 0.5, "vel": 100, "art": "plain"},  # 同 onset 低音
        {"midi": 67, "start": 1.0, "len": 1.0, "vel": 100, "art": "plain"},
    ]
    out = NeuralTake._monophonize(notes)
    assert [n["midi"] for n in out] == [60, 64, 67]


@needs_ckpt
def test_available_instruments_are_the_paper_four():
    assert available_instruments() == ["violin", "trumpet", "guitar", "glass"]


@needs_ckpt
def test_render_l0_engine_with_behavior_library():
    """取色 L0 音色 × 行为库:走 NeuralTake.render 同一路径,sr 取引擎的,长度随音符。"""
    from runtime.engine import Engine
    from synth.instrument import build_instrument
    from tests.helpers import make_test_snapshot

    eng = Engine(build_instrument(make_test_snapshot(), noise_level=0.02), eval_mode=True)
    nt = NeuralTake(device="cpu")
    notes = [
        {"midi": 62, "start": 0.0, "len": 0.5, "vel": 100, "art": "vibrato"},
        {"midi": 65, "start": 0.5, "len": 0.4, "vel": 100, "art": "plain"},
    ]
    y, sr = nt.render(notes, instrument=eng, behavior="violin", tail_s=0.1)
    assert sr == eng.sr
    assert abs(len(y) / sr - (0.9 + 0.1 + 0.1)) < 0.15  # 音符 + 乐句尾 0.1 + tail
    assert np.max(np.abs(y)) > 0.5 and np.isfinite(y).all()
    with pytest.raises(ValueError):
        nt.render(notes, instrument=eng, behavior=None)  # L0 音色必须显式给行为库
