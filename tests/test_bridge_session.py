"""Session persistence: roundtrip, validation, corrupt-file tolerance, atomic write."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

from bridge_session import BridgeSessionMixin, clean_session  # noqa: E402


class _B(BridgeSessionMixin):
    def __init__(self, path):
        self.SESSION_PATH = path


FULL = {
    "params": {"brightness": 0.7, "noise": 0.1, "master": 0.5, "bogus": 3},
    "adsr": {"a": 0.1, "d": 0.2, "s": 0.8, "r": 0.4},
    "poly": True,
    "live_behavior": "violin",
    "tone": "glass studio",
    "tempo": 96,
    "piano": {
        "bpm": 96,
        "notes": [
            {"midi": 62, "start": 0.0, "len": 0.5, "vel": 100, "art": "pluck"},
            {"midi": 200, "start": 0.0, "len": 0.5},  # out of range → dropped
            {"midi": 65, "start": 0.5, "len": 0.4},  # vel/art default
        ],
        "neural": "glass",
        "behavior": "violin",
        "traj_gain": 5.0,  # clamped
        "vib_source": "model",
        "vib_random": 0.4,
    },
    "junk": {"x": 1},
}


def test_roundtrip_and_validation(tmp_path):
    b = _B(tmp_path / "s.json")
    r = b.save_session(FULL)
    assert r["ok"] and Path(r["path"]).exists()
    s = b.load_session()
    assert "junk" not in s and "bogus" not in s["params"]
    assert s["params"]["brightness"] == 0.7 and s["adsr"]["s"] == 0.8 and s["poly"] is True
    assert s["live_behavior"] == "violin" and s["tone"] == "glass studio" and s["tempo"] == 96.0
    pr = s["piano"]
    assert [n["midi"] for n in pr["notes"]] == [62, 65]
    assert pr["notes"][1]["vel"] == 100 and pr["notes"][1]["art"] == "pluck"
    assert pr["traj_gain"] == 2.0 and pr["vib_source"] == "model" and pr["neural"] == "glass"


def test_missing_corrupt_and_future_version(tmp_path):
    p = tmp_path / "s.json"
    b = _B(p)
    assert b.load_session() == {}
    p.write_text("{not json", encoding="utf-8")
    assert b.load_session() == {}
    p.write_text(json.dumps({"version": 99, "params": {"master": 0.3}}), encoding="utf-8")
    assert b.load_session() == {}
    p.write_text("[]", encoding="utf-8")
    assert b.load_session() == {}


def test_clean_drops_bad_adsr_and_clamps():
    s = clean_session({"adsr": {"a": 0.1, "d": "x", "s": 0.5, "r": 0.2}, "params": {"master": 7}, "tempo": 999})
    assert "adsr" not in s and s["params"]["master"] == 1.0 and s["tempo"] == 240.0


def test_atomic_write_leaves_no_temp_files(tmp_path):
    b = _B(tmp_path / "s.json")
    for i in range(5):
        assert b.save_session({"params": {"master": i / 10}})["ok"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.json"]
    assert b.load_session()["params"]["master"] == 0.4


def test_pair_timbre_survives_roundtrip(tmp_path):
    """The timbre × behavior pair is session state: a DDSP or baked pair must come back
    as saved, not fall back to the captured tone (the page restores piano.neural from the
    same file, so dropping pair_timbre would leave the label and the roll disagreeing)."""
    b = _B(tmp_path / "s.json")
    b.save_session(dict(FULL, pair_timbre="glass"))
    assert b.load_session()["pair_timbre"] == "glass"
    assert "pair_timbre" not in clean_session(dict(FULL, pair_timbre=""))
