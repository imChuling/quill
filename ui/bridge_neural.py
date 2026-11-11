"""Neural rendering, DDSP baking and behaviour library management (Bridge mixin).

_render_neural is the entry point shared by piano-roll playback and export; instruments of the form "tone:<name>" use the offline L0
engine of the capture timbre in control-driven mode, with engines cached by file mtime. bake_ddsp fine-tunes in a background subprocess, progress is
polled through a JSON status file, and only one runs at a time.
"""

from __future__ import annotations

from pathlib import Path


class BridgeNeuralMixin:

    def _neural(self):
        """Lazily loaded singleton: torch is imported / the checkpoint loaded only on first call."""
        if not hasattr(self, "_neural_take"):
            from neural.neural_take import get_neural_take

            self._neural_take = get_neural_take()
        return self._neural_take

    def _neural_postfx(self):
        """The desktop knobs that act on neural rendering."""
        return {
            "brightness": float(self.params.get("brightness", 0.5)),
            "reverb": float(self.params.get("reverb", 0.0)),
        }

    def _l0_engine(self, name):
        """Capture timbre -> offline L0 engine (for control-driven mode), cached by file mtime."""
        from runtime.engine import engine_from_snapshot_file
        from snapshot import load_snapshot

        from .bridge import USER_TONES

        path = self._default_path if name == "__default__" else (USER_TONES / f"{name}.npz")
        if not Path(path).exists():
            raise FileNotFoundError(f"tone {name!r} not found")
        key = (str(path), Path(path).stat().st_mtime)
        cache = getattr(self, "_l0_engines", None)
        if cache is None:
            cache = self._l0_engines = {}
        if key not in cache:
            meta = load_snapshot(str(path)).get("meta", {}) or {}
            tex = bool(meta.get("texture"))
            cache.clear()
            cache[key] = engine_from_snapshot_file(
                str(path),
                eval_mode=True,
                noise_level=0.6 if tex else float(meta.get("noise_level") or 0.02),
            )
            cache[key]._texture = tex
        return cache[key]

    def _render_neural(self, notes, instrument, opts=None):
        """Piano-roll notes -> neural rendering (shared by playback and export).

        Guard: if there are notes but the render is silent (peak < 1e-4, about -80 dBFS), drop that timbre's cached voice, reload
        once and render again."""
        import numpy as np

        y, sr = self._render_neural_once(notes, instrument, opts)
        if notes and len(y) and float(np.max(np.abs(y))) < 1e-4:
            key = str(instrument)
            print(f"[bridge] neural render silent for {key!r}; reloading voice and retrying once")
            nt = self._neural()
            getattr(nt, "_voices", {}).pop(key, None)
            if key.startswith("tone:"):
                self._l0_engines = {}  # capture timbres use the L0 engine cache, rebuild it too
            y, sr = self._render_neural_once(notes, instrument, opts)
        return y, sr

    def _render_neural_once(self, notes, instrument, opts=None):
        from neural.neural_take import UI_RENDER_DEFAULTS

        o = opts or {}
        behavior = o.get("behavior") or None
        if str(instrument).startswith("tone:"):
            instrument = self._l0_engine(str(instrument)[5:])
            behavior = behavior or "violin"
            if "noise" in self.params and not getattr(instrument, "_texture", False):
                instrument.instrument.noise_level = float(self.params["noise"])
        return self._neural().render(
            notes,
            instrument=instrument,
            behavior=behavior,
            traj_gain=float(o.get("traj_gain", 1.0)),
            vibrato_source=o.get("vibrato_source", "retrieval"),
            randomness=float(o.get("randomness", 0.3)),
            **{k: float(o.get(k, v)) for k, v in UI_RENDER_DEFAULTS.items()},
            postfx=self._neural_postfx(),
        )

    def play_notes_neural(
        self,
        notes,
        bpm=120.0,
        instrument="violin",
        behavior=None,
        traj_gain=1.0,
        vibrato_source="retrieval",
        randomness=0.3,
        timing=0.0,
        continuity=0.0,
    ):
        """Piano-roll playback (neural version): render the whole thing offline -> sd.play."""
        import sounddevice as sd

        try:
            y, sr = self._render_neural(
                notes,
                instrument,
                {
                    "behavior": behavior,
                    "traj_gain": traj_gain,
                    "vibrato_source": vibrato_source,
                    "randomness": randomness,
                    "timing": timing,
                    "continuity": continuity,
                },
            )
        except Exception as e:
            return {"ok": False, "msg": str(e)[:80]}
        try:
            sd.play(y, sr, device=self._out_dev)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        return {"ok": True, "dur": round(len(y) / sr, 3)}

    def list_instruments(self):
        """Piano-roll Neural list: built-in + baked + captured."""
        from neural.neural_take import available_instruments, user_ddsp_instruments

        return (
            available_instruments()
            + user_ddsp_instruments()
            + [f"tone:{t['name']}" for t in self.list_tones()]
        )

    # ---- DDSP baking ----
    def _bake_auto_base(self):
        """Choose the base by the nature of the latest capture."""
        import numpy as np

        inst = self.engine.instrument
        ae = getattr(inst, "amp_env", None)
        if ae is not None and len(ae) > 4:
            ae = np.asarray(ae, np.float64)
            if ae[-1] / (ae.max() + 1e-9) < 0.35:
                return "guitar"
        return "glass" if float(getattr(self, "_last_centroid", 6.0)) < 1.6 else "violin"

    def bake_ddsp(self, name, base=None, steps=1000):
        """Latest capture take -> fine-tune DDSP in a background subprocess. Only one runs at a time."""
        import subprocess
        import sys as _sys

        import soundfile as sf

        from neural.bake_ddsp import MIN_TAKE_S
        from neural.neural_take import INSTRUMENTS, USER_DDSP_DIR

        from .bridge import ROOT, USER_TONES

        if self._bake_running():
            return {"ok": False, "msg": "A bake is already running"}
        wav = USER_TONES / "_last_take_gated.wav"
        if not wav.exists():
            wav = USER_TONES / "_last_take_raw.wav"
        if not wav.exists():
            return {"ok": False, "msg": "Capture a tone first"}
        take_s = float(sf.info(str(wav)).duration)
        if take_s < MIN_TAKE_S:
            return {
                "ok": False,
                "msg": f"Take is {take_s:.0f}s — DDSP needs {MIN_TAKE_S:.0f}s+ (the instant tone is fine for short takes)",
            }
        safe = self._safe_name(name, "bake")
        base = base if base in INSTRUMENTS else self._bake_auto_base()
        USER_DDSP_DIR.mkdir(parents=True, exist_ok=True)
        status = USER_DDSP_DIR / "_bake_status.json"
        if status.exists():
            status.unlink()
        log = open(USER_DDSP_DIR / "_bake.log", "w")  # noqa: SIM115
        cmd = [
            _sys.executable, "-W", "ignore", "-m", "neural.bake_ddsp",
            "--wav", str(wav), "--name", safe, "--base", base,
            "--steps", str(int(steps)), "--status", str(status),
        ]
        proc = subprocess.Popen(cmd, cwd=str(ROOT.parent), stdout=log, stderr=subprocess.STDOUT)
        self._bake = {"proc": proc, "name": safe, "base": base, "status": status, "log": log, "steps": int(steps)}
        return {"ok": True, "name": safe, "base": base, "take_s": round(take_s, 1), "steps": int(steps)}

    def _bake_running(self):
        b = getattr(self, "_bake", None)
        return b is not None and b["proc"].poll() is None

    def bake_status(self):
        """{"state": idle|preparing|training|done|failed, ...}"""
        import json

        b = getattr(self, "_bake", None)
        if b is None:
            return {"state": "idle"}
        d = {"state": "preparing", "name": b["name"], "base": b["base"], "steps": b["steps"], "step": 0}
        try:
            d.update(json.loads(b["status"].read_text()))
        except Exception:
            pass
        rc = b["proc"].poll()
        if rc is not None and d.get("state") not in ("done", "failed"):
            d["state"] = "failed"
            d["msg"] = f"bake process exited {rc} (see assets/user_ddsp/_bake.log)"
        if rc is not None and not b["log"].closed:
            b["log"].close()
        return d

    def list_behavior_sources(self):
        """List available behaviour sources (built-in + user-imported)."""
        return self._neural().list_behavior_sources()

    def import_style_from_wav(self, wav_path, name=None, min_notes=2):
        """Extract a behaviour library from a wav."""
        import subprocess
        import sys as _sys
        import tempfile
        from pathlib import Path

        import numpy as np

        from .bridge import ROOT

        p = Path(wav_path)
        if not p.exists():
            return {"ok": False, "msg": "File not found"}
        name = str(name or p.stem)
        with tempfile.TemporaryDirectory(prefix="quill_style_") as tmp:
            npz = Path(tmp) / f"{name}.npz"
            cmd = [
                _sys.executable,
                "-W",
                "ignore",
                str(ROOT / "tools" / "extract_style.py"),
                str(p),
                "-o",
                str(npz),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                return {"ok": False, "msg": (r.stderr or r.stdout).strip().splitlines()[-1][:120] if (r.stderr or r.stdout).strip() else "extract_style failed"}
            if not npz.exists():
                return {"ok": False, "msg": "extract_style produced no output"}
            n = int(np.load(str(npz))["features"].shape[0])
            if n < int(min_notes):
                return {"ok": False, "msg": f"only {n} note(s) — play several notes or a phrase for a behavior library", "n_notes": n}
            imported = self._neural().import_style(str(npz))
        return {"ok": True, "name": imported, "n_notes": n}

    def delete_behavior_source(self, name):
        """Delete a user-imported behaviour library."""
        self._neural().delete_style(name)
        return {"ok": True}
