"""_render_neural 静音守门:有音符却渲染出静音 → 丢缓存 voice 重载一次再渲(只重试一次)。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

from bridge_neural import BridgeNeuralMixin  # noqa: E402


class _NT:
    def __init__(self):
        self._voices = {"violin": object(), "glass": object()}


class _B(BridgeNeuralMixin):
    def __init__(self, outputs):
        self._neural_take = _NT()
        self._l0_engines = {"x": 1}
        self._outputs = list(outputs)
        self.calls = []

    def _render_neural_once(self, notes, instrument, opts=None):
        self.calls.append(instrument)
        return self._outputs.pop(0), 16000


NOTES = [{"midi": 60, "start": 0.0, "len": 0.5, "vel": 100, "art": "pluck"}]


def test_silent_render_reloads_voice_and_retries_once():
    b = _B([np.zeros(8000, np.float32), np.full(8000, 0.3, np.float32)])
    y, sr = b._render_neural(NOTES, "violin")
    assert b.calls == ["violin", "violin"] and sr == 16000 and float(np.max(np.abs(y))) > 0.1
    assert "violin" not in b._neural_take._voices and "glass" in b._neural_take._voices


def test_still_silent_after_retry_is_returned_not_looped():
    b = _B([np.zeros(8000, np.float32), np.zeros(8000, np.float32)])
    y, _ = b._render_neural(NOTES, "violin")
    assert b.calls == ["violin", "violin"] and float(np.max(np.abs(y))) == 0.0


def test_good_render_does_not_touch_cache():
    b = _B([np.full(8000, 0.2, np.float32)])
    b._render_neural(NOTES, "glass")
    assert b.calls == ["glass"] and set(b._neural_take._voices) == {"violin", "glass"}


def test_tone_instrument_clears_l0_engine_cache():
    b = _B([np.zeros(8000, np.float32), np.full(8000, 0.2, np.float32)])
    b._render_neural(NOTES, "tone:glass studio")
    assert b._l0_engines == {}


def test_empty_notes_never_retries():
    b = _B([np.zeros(10, np.float32)])
    b._render_neural([], "violin")
    assert b.calls == ["violin"]
