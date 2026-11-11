"""云端渲染服务契约快测(10.15 回归账):/status 能力字段、内置行为库互换、
captured tone × behavior、拒收未知音色/行为。TestClient 内存跑,目录指向 tmp。"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural.neural_take import INSTRUMENTS  # noqa: E402

HAVE_CKPT = all(s["ckpt"].exists() and s["traj"].exists() for s in INSTRUMENTS.values())
pytestmark = pytest.mark.skipif(not HAVE_CKPT, reason="DDSP checkpoints / trajectory banks missing")

NOTES = [
    {"midi": 62, "start": 0.0, "len": 0.4, "vel": 100, "art": "vibrato"},
    {"midi": 65, "start": 0.4, "len": 0.3, "vel": 100, "art": "plain"},
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    base = tmp_path_factory.mktemp("rs")
    import os

    os.environ["QUILL_TAKES_DIR"] = str(base / "takes")
    os.environ["QUILL_ASSETS_DIR"] = str(base / "assets")
    import ui.render_service as rs

    rs = importlib.reload(rs)
    # 放一个 captured tone 进资产目录
    from snapshot import save_snapshot
    from tests.helpers import make_test_snapshot

    rs.TONES_DIR.mkdir(parents=True, exist_ok=True)
    save_snapshot(str(rs.TONES_DIR / "t1.npz"), make_test_snapshot())
    return TestClient(rs.app)


def _render(client, **body):
    r = client.post("/render", json={"notes": NOTES, "bpm": 120, "title": "t", **body})
    return r


def test_status_lists_timbres_and_behaviors(client):
    s = client.get("/status").json()
    assert s["timbres"] == ["violin", "trumpet", "guitar", "glass"]
    assert set(s["timbres"]) <= set(s["behaviors"])


def test_builtin_cross_behavior_renders(client):
    r = _render(client, timbre="violin", behavior="guitar")
    assert r.status_code == 200 and r.json()["state"] == "ready"
    tid = r.json()["take_id"]
    meta = client.get(f"/takes/{tid}").json()
    assert "violin" in meta["timbre"]["name"] and "guitar" in meta["timbre"]["name"]
    wav = client.get(f"/takes/{tid}/audio.wav").content
    assert len(wav) > 1000


def test_tone_with_and_without_behavior(client):
    r1 = _render(client, timbre="tone:t1", behavior="violin")  # L0 控制驱动
    r2 = _render(client, timbre="tone:t1")  # 旧事件路径
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["take_id"] != r2.json()["take_id"]


def test_rejects_unknown(client):
    assert _render(client, timbre="cello").status_code == 400
    assert _render(client, timbre="violin", behavior="flute").status_code == 400
    assert _render(client, timbre="tone:nope").status_code == 400


def test_synth_params_change_take_identity(client):
    a = _render(client, timbre="violin", synth_params={"vibmodel": 0}).json()["take_id"]
    b = _render(client, timbre="violin", synth_params={"vibmodel": 1, "randomness": 0.5}).json()["take_id"]
    assert a != b
    # 同参数再渲染:幂等命中
    assert _render(client, timbre="violin", synth_params={"vibmodel": 0}).json()["take_id"] == a


def test_idempotent_audio_is_finite(client):
    tid = _render(client, timbre="glass", behavior="trumpet").json()["take_id"]
    import io

    import soundfile as sf

    y, sr = sf.read(io.BytesIO(client.get(f"/takes/{tid}/audio.wav").content))
    assert sr == 16000 and np.isfinite(y).all() and np.max(np.abs(y)) > 0.3
