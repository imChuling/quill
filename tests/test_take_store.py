"""TakeStore 单测:幂等 stage、attach/fail 生命周期(ui/take_store.py,render_service 用)。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

from take_store import TakeStore  # noqa: E402

NOTES = [
    {"midi": 60, "start": 0.0, "len": 0.4, "vel": 100, "art": "pluck"},
    {"midi": 64, "start": 0.5, "len": 0.4, "vel": 90, "art": "pluck"},
]


@pytest.fixture()
def store(tmp_path):
    return TakeStore(tmp_path / "takes")


# ---------- TakeStore ---------- #
def test_stage_is_idempotent(store):
    t1 = store.stage(
        title="t",
        notes=NOTES,
        bpm=120,
        timbre={"name": "voice"},
        duration_s=2.4,
        sr=44100,
        provenance={},
    )
    t2 = store.stage(
        title="t",
        notes=NOTES,
        bpm=120,
        timbre={"name": "voice"},
        duration_s=2.4,
        sr=44100,
        provenance={},
    )
    assert t1["take_id"] == t2["take_id"]  # 重复点击保护
    t3 = store.stage(
        title="t2",
        notes=NOTES,
        bpm=120,
        timbre={"name": "voice"},
        duration_s=2.4,
        sr=44100,
        provenance={},
    )
    assert t3["take_id"] != t1["take_id"]  # 内容不同 → 新 take


def test_attach_and_fail_lifecycle(store):
    t = store.stage(
        title="x",
        notes=NOTES,
        bpm=100,
        timbre={"name": "v"},
        duration_s=1.0,
        sr=44100,
        provenance={},
    )
    assert t["audio"]["state"] == "pending"
    store.attach_audio(t["take_id"], b"RIFFfake", -6.0)
    got = store.get(t["take_id"])
    assert got["audio"]["state"] == "ready" and got["audio"]["wav_bytes"] == 8
    store.fail(t["take_id"], "boom")
    assert store.get(t["take_id"])["audio"]["state"] == "failed"
