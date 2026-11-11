"""Stage 5 unit tests: rule-tree classification / vibrato-vs-slide / .tplib round trip / forest / strum window / IDMT."""

import numpy as np
import pytest

from analysis.articulation import (
    _strum_windows,
    classify_events,
    forest_predict,
    train_forest,
)
from analysis.datasets import load_idmt_smt_guitar
from analysis.timbre import detect_vibrato
from analysis.tplib import build_library, load_tplib, save_tplib
from snapshot import make_snapshot

SR = 44100


def _ev(
    eid,
    t_on,
    t_off,
    pitch=60.0,
    vel=0.7,
    log_attack=-1.0,
    centroid=1500.0,
    slope=0.0,
    poly=1,
    vib=None,
    label="",
    f0_track=None,
):
    return {
        "id": eid,
        "t_on": t_on,
        "t_off": t_off,
        "pitch_midi": pitch,
        "velocity_est": vel,
        "f0_track": f0_track or [],
        "label": label,
        "vibrato": vib,
        "features": {
            "log_attack": log_attack,
            "centroid_mean": centroid,
            "centroid_slope": 0.0,
            "f0_slope_cents_s": slope,
            "polyphony": poly,
            "hnr": 20.0,
        },
    }


def _mono_track(t0, hz0, hz1, n=10):
    """Monotonic f0 track (slide needs the monotonicity criterion)."""
    return [[t0 + i * 0.04, hz0 + (hz1 - hz0) * i / (n - 1)] for i in range(n)]


# --------------------------------------------------------------------------- #
# each class of the rule tree (constructed features, testing the rule logic directly)
# --------------------------------------------------------------------------- #
def test_rule_pluck():
    e = _ev(0, 0.0, 0.4, log_attack=-2.0, centroid=2500)  # fast attack + bright
    classify_events([e])
    assert e["label"] == "pluck"


def test_rule_sustain():
    e = _ev(0, 0.0, 1.0, log_attack=-1.2, centroid=900, slope=50)  # slow attack + dark + no slope
    classify_events([e])
    assert e["label"] == "sustain"


def test_rule_slide():
    # net slope>500, duration>60ms, f0 monotonically rising
    e = _ev(0, 0.0, 0.4, slope=900, log_attack=-1.2, f0_track=_mono_track(0.0, 196, 247))
    classify_events([e])
    assert e["label"] == "slide"


def test_slide_needs_monotonic_track():
    """A large slope with a non-monotonic f0 (octave-jump type) should not be judged a slide (pitfall: monotonicity check)."""
    jumpy = [[0.0, 196], [0.04, 392], [0.08, 196], [0.12, 392], [0.16, 196]]
    e = _ev(0, 0.0, 0.4, slope=900, log_attack=-1.2, f0_track=jumpy)
    classify_events([e])
    assert e["label"] != "slide"


def test_rule_strum():
    evs = [
        _ev(0, 1.00, 1.5, pitch=48),
        _ev(1, 1.02, 1.5, pitch=52),
        _ev(2, 1.04, 1.5, pitch=55),
    ]  # 3 onsets in the same window <80ms
    classify_events(evs)
    assert all(e["label"] == "strum" for e in evs)


def test_rule_legato():
    a = _ev(0, 0.0, 0.5, pitch=60, log_attack=-1.2)
    b = _ev(1, 0.51, 1.0, pitch=62, log_attack=-1.2, vel=0.7)  # joined + small interval + soft attack
    classify_events([a, b])
    assert b["label"] == "legato"


def test_strum_window_needs_three():
    evs = [_ev(0, 1.0, 1.5), _ev(1, 1.03, 1.5)]  # only 2 -> not a strum
    assert len(_strum_windows(evs)) == 0
    evs3 = [_ev(0, 1.0, 1.5), _ev(1, 1.03, 1.5), _ev(2, 1.06, 1.5)]
    assert len(_strum_windows(evs3)) == 3


# --------------------------------------------------------------------------- #
# pitfall: a slide must not be misjudged as vibrato (after linear detrending)
# --------------------------------------------------------------------------- #
def test_slide_not_flagged_vibrato():
    # f0 track of a monotonic slide (linear cents ramp)
    t = np.arange(0, 0.45, 0.01)
    hz = 165 * 2 ** (4 * (t / 0.45) / 12.0)
    assert detect_vibrato(hz, t) is None, "a monotonic slide should not be judged as vibrato"
    # real vibrato (periodicity remains after detrending) should be detected
    hz_vib = 220 * 2 ** ((30 / 1200.0) * np.sin(2 * np.pi * 5.5 * t))
    assert detect_vibrato(hz_vib, t) is not None


def test_vibrato_then_slide_priority():
    """Both a net slope and a vibrato field: the vibrato guard -> not a slide (periodicity extracted first)."""
    e = _ev(0, 0.0, 0.4, slope=900, vib={"rate_hz": 5.5, "depth_cents": 30})
    classify_events([e])
    assert e["label"] != "slide"  # not slide when vibrato is present


# --------------------------------------------------------------------------- #
# lossless .tplib round trip (FR-7 hard exit)
# --------------------------------------------------------------------------- #
def test_tplib_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.standard_normal(SR).astype(np.float32) * 0.1
    snap = make_snapshot(
        196.0,
        np.ones((60, 20), np.float32),
        np.ones((40, 20), np.float32),
        np.zeros(2646, np.float32),
        name="t",
    )
    events = [
        _ev(0, 0.10, 0.5, vel=0.3, label="pluck"),
        _ev(1, 0.60, 1.0, vel=0.9, label="pluck"),
        _ev(
            2,
            1.10,
            1.5,
            slope=900,
            label="slide",
        ),
        _ev(3, 2.00, 2.5, pitch=60, label="strum"),
        _ev(4, 2.02, 2.5, pitch=64, label="strum"),
        _ev(5, 2.04, 2.5, pitch=67, label="strum"),
    ]
    events[2]["f0_track"] = [[1.10, 196.0], [1.30, 220.0], [1.50, 247.0]]
    lib = build_library(events, y, SR, sustain_snapshot=snap)
    p = save_tplib(tmp_path / "x.tplib", lib)
    lib2 = load_tplib(p)

    assert len(lib2["pluck"]) == 2 and len(lib2["strum"]) == 1
    assert np.array_equal(lib["pluck"][0]["grain"], lib2["pluck"][0]["grain"])
    assert lib["pluck"][0]["rms_bucket"] == lib2["pluck"][0]["rms_bucket"]
    assert np.array_equal(lib["slide"][0]["curve_ctrl"], lib2["slide"][0]["curve_ctrl"])
    assert lib2["strum"][0]["direction"] in ("up", "down")
    assert np.array_equal(lib["sustain"]["H_env"], lib2["sustain"]["H_env"])
    assert lib2["meta"]["instrument_guess"] == lib["meta"]["instrument_guess"]


def test_tplib_pluck_rms_buckets():
    """pluck grains are bucketed by velocity_est into RMS buckets (0/1/2)."""
    y = np.zeros(SR, np.float32)
    events = [
        _ev(0, 0.1, 0.3, vel=0.1, label="pluck"),
        _ev(1, 0.4, 0.6, vel=0.5, label="pluck"),
        _ev(2, 0.7, 0.9, vel=0.95, label="pluck"),
    ]
    lib = build_library(events, y, SR)
    buckets = sorted(p["rms_bucket"] for p in lib["pluck"])
    assert buckets == [0, 1, 2]


# --------------------------------------------------------------------------- #
# forest fallback + graceful handling of a missing IDMT loader
# --------------------------------------------------------------------------- #
def test_forest_trains_and_predicts():
    events, labels = [], []
    for _ in range(20):
        events.append(_ev(0, 0.0, 0.4, log_attack=-2.0, centroid=2500))
        labels.append("pluck")
        events.append(_ev(0, 0.0, 1.0, log_attack=-1.0, centroid=800))
        labels.append("sustain")
    clf = train_forest(events, labels)
    pred = forest_predict(clf, events[:4])
    assert set(pred) <= {"pluck", "sustain"}


def test_idmt_loader_graceful_missing():
    assert load_idmt_smt_guitar("/nonexistent/path/idmt") == []


# --------------------------------------------------------------------------- #
# end-to-end classification accuracy (slow: runs PESTO per clip)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_classification_accuracy_on_synth_set():
    from eval.stage5_accept import build_labeled_set

    events, true, y, sr = build_labeled_set()
    classify_events(events)
    pred = [e["label"] for e in events]
    acc = float(np.mean([t == p for t, p in zip(true, pred)]))
    assert acc >= 0.80, f"rule tree accuracy {acc * 100:.1f}% < 80%"
