"""阶段4 单测：FR-1 截断 / events.json / 幽灵八度过滤 / 召回 / 特征。"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from analysis.features import (
    _PUBLIC_KEYS,
    analyze_events,
    hnr_db,
    load_events,
    polyphony_at,
    save_events,
)
from analysis.segment import ghost_octave_filter, union_events

SR = 44100


# --------------------------------------------------------------------------- #
# FR-1：导入 60s 截断
# --------------------------------------------------------------------------- #
def test_fr1_load_length_cap(tmp_path):
    from analysis.io import load

    y = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(SR * 3)) / SR)).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(p, y, SR)
    y2, sr2 = load(str(p), max_seconds=1.0)
    assert sr2 == SR
    assert abs(len(y2) - SR) < SR * 0.05, f"应截到 ~1s，实得 {len(y2) / SR:.2f}s"
    # 峰值归一化生效
    assert abs(np.max(np.abs(y2)) - 0.97) < 0.02


# --------------------------------------------------------------------------- #
# events.json 序列化（Part B 契约 schema）
# --------------------------------------------------------------------------- #
def test_events_json_roundtrip(tmp_path):
    events = [
        {
            "id": 0,
            "t_on": 0.1,
            "t_off": 0.5,
            "pitch_midi": 60.0,
            "velocity_est": 0.8,
            "f0_track": [[0.1, 261.6], [0.3, 262.0]],
            "features": {
                "log_attack": -2.0,
                "centroid_mean": 1500.0,
                "centroid_slope": 10.0,
                "f0_slope_cents_s": 2.0,
                "polyphony": 1,
                "hnr": 25.0,
            },
            "label": "",
            "vibrato": None,
            "_agree": True,
        }
    ]
    p = save_events(tmp_path / "events.json", events)
    loaded = load_events(p)
    assert len(loaded) == 1
    for k in _PUBLIC_KEYS:
        assert k in loaded[0], f"缺字段 {k}"
    assert "_agree" not in loaded[0], "内部诊断字段不应写入 events.json"
    assert loaded[0]["pitch_midi"] == 60.0
    assert loaded[0]["features"]["polyphony"] == 1


# --------------------------------------------------------------------------- #
# 幽灵八度过滤（坑位）
# --------------------------------------------------------------------------- #
def test_ghost_octave_filter_high_and_low():
    # PESTO f0 恒为 440Hz（midi 69 = A4）
    f0 = np.full(200, 440.0)
    ftimes = np.linspace(0, 2.0, 200)
    events = [
        {"t_on": 0.1, "t_off": 0.5, "pitch_midi": 69.0, "amp": 0.8},  # 真音
        {"t_on": 0.1, "t_off": 0.5, "pitch_midi": 81.0, "amp": 0.3},  # 高八度幽灵
        {"t_on": 0.1, "t_off": 0.5, "pitch_midi": 57.0, "amp": 0.3},  # 低八度幽灵
        {"t_on": 0.1, "t_off": 0.5, "pitch_midi": 69.0, "amp": 0.05},  # 弱事件
    ]
    kept, dropped = ghost_octave_filter(events, f0, ftimes, amp_thresh=0.1)
    kept_pitches = [e["pitch_midi"] for e in kept]
    assert 69.0 in kept_pitches
    assert 81.0 not in kept_pitches and 57.0 not in kept_pitches
    reasons = {d["drop_reason"] for d in dropped}
    assert "ghost_octave" in reasons and "low_amp" in reasons


def test_union_marks_agreement():
    bp = [{"t_on": 0.10, "t_off": 0.5, "pitch_midi": 60.0, "amp": 0.8}]
    on = [{"t_on": 0.11, "t_off": 0.5, "pitch_midi": 60.0, "amp": float("nan")}]
    merged, dis = union_events(bp, on, onset_tol_s=0.05, pitch_tol=1.0)
    assert len(merged) == 1 and merged[0]["agree"] is True
    assert len(dis) == 0


# --------------------------------------------------------------------------- #
# 特征健全性
# --------------------------------------------------------------------------- #
def test_polyphony_counts_overlap():
    evs = [
        {"t_on": 0.0, "t_off": 1.0},
        {"t_on": 0.1, "t_off": 0.9},
        {"t_on": 0.2, "t_off": 0.8},
        {"t_on": 2.0, "t_off": 2.5},
    ]
    assert polyphony_at(evs[0], evs) == 3
    assert polyphony_at(evs[3], evs) == 1


def test_hnr_harmonic_vs_noise():
    t = np.arange(int(SR * 0.5)) / SR
    tone = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    assert hnr_db(tone, SR, 220.0) > hnr_db(noise, SR, 220.0)


# --------------------------------------------------------------------------- #
# 召回闭环（用引擎渲染已知曲 -> 分析 -> 召回 ≥90%）—— 需 Basic Pitch，较慢
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_recall_on_rendered_phrase():
    from eval.label_tool import compute_recall
    from runtime.engine import Engine
    from snapshot import load_snapshot
    from synth.instrument import build_instrument

    snap_path = Path(__file__).resolve().parents[1] / "assets" / "snapshots" / "guitar.npz"
    if not snap_path.exists():
        pytest.skip("需要 assets/snapshots/guitar.npz（先跑 stage2 验收）")
    inst = build_instrument(load_snapshot(snap_path), noise_level=0.0)
    eng = Engine(inst, eval_mode=True)
    notes = [(0.2, 60), (0.7, 62), (1.2, 64), (1.7, 65), (2.2, 67), (2.7, 69)]
    evs, gt = [], []
    for t, p in notes:
        evs += [(t, "on", p, 100, 0.0), (t + 0.4, "off", p, 0, 0.0)]
        gt.append({"t_on": t, "pitch_midi": float(p)})
    y = eng.render_offline(evs, 3.4)
    det = analyze_events(y)
    r = compute_recall(det, gt, onset_tol_s=0.06, pitch_tol=1.0)
    assert r["recall"] >= 0.90, f"召回 {r['recall'] * 100:.1f}% < 90%"


def test_analyze_events_onset_only_no_basic_pitch():
    """不依赖 Basic Pitch 也能出事件（自建前端单独可用）。"""
    from runtime.engine import Engine
    from synth.instrument import Instrument

    K = 30
    inst = Instrument(
        1.0 / np.arange(1, K + 1),
        np.arange(1, K + 1, dtype=float),
        np.zeros(0),
        np.zeros(0),
        220.0,
        SR,
        noise_level=0.0,
    )
    eng = Engine(inst, eval_mode=True)
    y = eng.render_offline(
        [
            (0.2, "on", 60, 100, 0.0),
            (0.6, "off", 60, 0, 0.0),
            (0.9, "on", 67, 100, 0.0),
            (1.3, "off", 67, 0, 0.0),
        ],
        1.6,
    )
    det = analyze_events(y, use_basic_pitch=False)
    assert len(det) >= 2
    assert all("features" in e for e in det)
