"""Api (pywebview js_api) → main-window pushes: which events reach the page and in what shape.

pywebview itself is not needed; a stub module stands in for `webview` so the Api class can be
built with fake bridge and window objects.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

if "webview" not in sys.modules:
    _stub = types.ModuleType("webview")
    _stub.OPEN_DIALOG = 10
    sys.modules["webview"] = _stub

from ui.app_web import Api  # noqa: E402


class _Win:
    def __init__(self):
        self.js = []

    def evaluate_js(self, s):
        self.js.append(s)


class _Bridge:
    def load_tone(self, name):
        return {"ok": True, "name": name, "disp": "Saved Tone", "f0": 220.0}

    def capture_stop(self):
        return {"ok": True, "name": "_last_capture", "disp": "Capture", "f0": 196.0}

    def capture_from_file(self, path):
        raise OSError(f"Error opening {path!r}: Format not recognised.")

    def bake_status(self):
        return {"state": "done", "name": "mybake"}


def _api():
    a = Api.__new__(Api)
    a.bridge, a.main_window, a.capture_window = _Bridge(), _Win(), None
    return a


def test_load_tone_does_not_announce_a_capture():
    """Picking a saved tone syncs the name but must not fire quillCaptured, which would flip
    the page's pair to the unsaved capture and point the roll at _last_capture.npz."""
    a = _api()
    a.load_tone("saved")
    js = "\n".join(a.main_window.js)
    assert "quillToneName" in js and "quillCaptured" not in js


def test_capture_stop_announces_the_capture():
    a = _api()
    a.capture_stop()
    assert any("quillCaptured" in s for s in a.main_window.js)


def test_file_capture_failure_is_a_result_not_an_exception():
    """A dropped non-audio file must come back as {ok: False, msg} so both pages can show it;
    an exception would reject the JS promise and leave the window on "Analysing…"."""
    a = _api()
    for call in (a.capture_path, a.capture_stem):
        r = call("/nope/file.mid")
        assert r["ok"] is False and "Format not recognised" in r["msg"]
    assert not a.main_window.js


def test_bake_done_refreshes_both_timbre_lists():
    a = _api()
    a._bake_announced = False
    a.bake_status()
    js = "\n".join(a.main_window.js)
    assert "quillRefreshTones" in js and "quillRefreshInstruments" in js
