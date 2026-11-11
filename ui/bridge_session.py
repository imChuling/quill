"""Session persistence: knobs, ADSR, voice mode, live behavior, piano-roll contents.

One JSON file (`assets/session.json`). The UI owns the state and decides when to save
(debounced on change, and on close); the bridge only validates and stores it, so a
corrupt or stale file can never break startup: `load_session()` returns {} on any
problem and the UI falls back to its defaults.
"""

import json
import os
import tempfile
from pathlib import Path

SESSION_VERSION = 1

# knob keys accepted in session["params"]; each value is clamped to [0, 1]
_KNOBS = ("brightness", "noise", "vibdepth", "vibrate", "master", "artint", "strum", "gsens", "pick")


def _unit(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, f)) if f == f else None


def _clean_params(d):
    if not isinstance(d, dict):
        return None
    out = {}
    for k in _KNOBS:
        if k in d:
            v = _unit(d[k])
            if v is not None:
                out[k] = v
    return out or None


def _clean_adsr(d):
    if not isinstance(d, dict):
        return None
    out = {}
    for k in ("a", "d", "s", "r"):
        v = _unit(d.get(k))
        if v is None:
            return None
        out[k] = v
    return out


def _clean_notes(lst, limit=4000):
    if not isinstance(lst, list):
        return None
    out = []
    for n in lst[:limit]:
        if not isinstance(n, dict):
            continue
        try:
            midi = int(n["midi"])
            start = float(n["start"])
            ln = float(n["len"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= midi <= 127) or start < 0 or ln <= 0:
            continue
        vel = int(n.get("vel", 100))
        art = str(n.get("art", "pluck"))[:16]
        out.append({"midi": midi, "start": round(start, 4), "len": round(ln, 4), "vel": min(127, max(1, vel)), "art": art})
    return out


def _clean_piano(d):
    if not isinstance(d, dict):
        return None
    out = {}
    try:
        bpm = float(d.get("bpm", 120.0))
        out["bpm"] = min(240.0, max(40.0, bpm))
    except (TypeError, ValueError):
        out["bpm"] = 120.0
    out["notes"] = _clean_notes(d.get("notes")) or []
    out["neural"] = str(d.get("neural") or "")[:80]
    out["behavior"] = str(d.get("behavior") or "")[:80]
    tg = d.get("traj_gain", 1.0)
    try:
        out["traj_gain"] = min(2.0, max(0.3, float(tg)))
    except (TypeError, ValueError):
        out["traj_gain"] = 1.0
    out["vib_source"] = "model" if d.get("vib_source") == "model" else "retrieval"
    out["vib_random"] = _unit(d.get("vib_random", 0.3)) or 0.3
    return out


def clean_session(raw):
    """Validate a session dict from disk or from the UI. Unknown keys are dropped."""
    if not isinstance(raw, dict):
        return {}
    out = {"version": SESSION_VERSION}
    p = _clean_params(raw.get("params"))
    if p:
        out["params"] = p
    a = _clean_adsr(raw.get("adsr"))
    if a:
        out["adsr"] = a
    if "poly" in raw:
        out["poly"] = bool(raw["poly"])
    lb = raw.get("live_behavior")
    if lb:
        out["live_behavior"] = str(lb)[:80]
    tone = raw.get("tone")
    if tone:
        out["tone"] = str(tone)[:120]
    pair = raw.get("pair_timbre")
    if pair:
        out["pair_timbre"] = str(pair)[:120]
    try:
        tempo = float(raw.get("tempo", 120.0))
        out["tempo"] = min(240.0, max(40.0, tempo))
    except (TypeError, ValueError):
        pass
    pr = _clean_piano(raw.get("piano"))
    if pr:
        out["piano"] = pr
    return out


class BridgeSessionMixin:
    SESSION_PATH = None  # set in __init__ (needs ROOT); tests override

    def _session_path(self):
        if self.SESSION_PATH is not None:
            return Path(self.SESSION_PATH)
        from .bridge import ROOT

        return ROOT / "assets" / "session.json"

    def save_session(self, state):
        """Validate and write atomically. Returns {ok, path} or {ok: False, msg}."""
        data = clean_session(state)
        path = self._session_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".session-", suffix=".json", dir=str(path.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, str(path))
        except OSError as e:
            return {"ok": False, "msg": f"cannot save session: {str(e)[:60]}"}
        return {"ok": True, "path": str(path)}

    def load_session(self):
        """Read + validate; {} when absent, corrupt, or from a newer version."""
        path = self._session_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict) or int(raw.get("version", 0) or 0) > SESSION_VERSION:
            return {}
        return clean_session(raw)
