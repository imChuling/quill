"""音色 / 行为库名字不能逃出各自目录:load_tone / delete_tone / rename_tone / delete_style。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

import bridge as bridge_mod  # noqa: E402
from bridge import QuillBridge  # noqa: E402


@pytest.fixture
def b(tmp_path, monkeypatch):
    # `bridge` (tests import it top-level) and `ui.bridge` (what the mixins import via
    # `from .bridge import USER_TONES`) are two module objects for the same file: patch both.
    import ui.bridge as pkg_bridge

    monkeypatch.setattr(bridge_mod, "USER_TONES", tmp_path / "tones")
    monkeypatch.setattr(pkg_bridge, "USER_TONES", tmp_path / "tones")
    (tmp_path / "tones").mkdir()
    return QuillBridge()


@pytest.mark.parametrize("bad", ["../escape", "../../etc/passwd", "/abs/path", "a/b", "..", "", "nul\x00l"])
def test_escaping_names_are_rejected(b, bad):
    assert QuillBridge._tone_path(bad) is None
    assert b.load_tone(bad) == {"ok": False, "msg": "Tone not found"}
    assert b.delete_tone(bad) == {"ok": False, "msg": "Invalid tone name"}
    assert b.rename_tone(bad, "x")["ok"] is False


def test_delete_never_touches_outside_folder(b, tmp_path):
    outside = tmp_path / "victim.npz"
    np.savez(outside, a=np.zeros(3))
    b.delete_tone("../victim")
    assert outside.exists()


def test_plain_names_resolve_inside_folder(b, tmp_path):
    p = QuillBridge._tone_path("my tone-1")
    assert p is not None and p.parent == (tmp_path / "tones").resolve() and p.name == "my tone-1.npz"


def test_delete_style_ignores_escaping_names(tmp_path, monkeypatch):
    import neural.neural_take as nt

    monkeypatch.setattr(nt, "USER_STYLES_DIR", tmp_path / "styles")
    (tmp_path / "styles").mkdir()
    victim = tmp_path / "victim.npz"
    np.savez(victim, a=np.zeros(3))

    obj = nt.NeuralTake.__new__(nt.NeuralTake)
    import threading

    obj._style_lock = threading.Lock()
    obj._custom_nns = {}
    nt.NeuralTake.delete_style(obj, "../victim")
    assert victim.exists()
