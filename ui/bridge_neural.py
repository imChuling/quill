"""神经渲染、DDSP 烘焙与行为库管理(Bridge 的 mixin)。

_render_neural 是卷帘回放与导出共用的入口;"tone:<name>" 形式的乐器用取色音色的离线 L0
引擎做控制驱动,引擎按文件 mtime 缓存。bake_ddsp 在后台子进程里微调,进度经 JSON 状态文件
轮询,一次只跑一个。
"""

from __future__ import annotations

from pathlib import Path


class BridgeNeuralMixin:

    def _neural(self):
        """懒加载单例:首次调用才 import torch/载 checkpoint。"""
        if not hasattr(self, "_neural_take"):
            from neural.neural_take import get_neural_take

            self._neural_take = get_neural_take()
        return self._neural_take

    def _neural_postfx(self):
        """桌面旋钮里作用于神经渲染的那几个。"""
        return {
            "brightness": float(self.params.get("brightness", 0.5)),
            "reverb": float(self.params.get("reverb", 0.0)),
        }

    def _l0_engine(self, name):
        """取色音色 → 离线 L0 引擎(控制驱动用),按文件 mtime 缓存。"""
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
        """卷帘 notes → 神经渲染(回放与导出共用)。

        守门:有音符却渲染出静音(峰值 < 1e-4,约 -80 dBFS)时,丢掉该音色的缓存 voice 重载
        一次再渲。"""
        import numpy as np

        y, sr = self._render_neural_once(notes, instrument, opts)
        if notes and len(y) and float(np.max(np.abs(y))) < 1e-4:
            key = str(instrument)
            print(f"[bridge] neural render silent for {key!r}; reloading voice and retrying once")
            nt = self._neural()
            getattr(nt, "_voices", {}).pop(key, None)
            if key.startswith("tone:"):
                self._l0_engines = {}  # 取色音色走 L0 引擎缓存,一并重建
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
        """卷帘回放(神经版):离线渲染整段 → sd.play。"""
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
        """卷帘 Neural 列表:内置 + 烘焙 + 取色。"""
        from neural.neural_take import available_instruments, user_ddsp_instruments

        return (
            available_instruments()
            + user_ddsp_instruments()
            + [f"tone:{t['name']}" for t in self.list_tones()]
        )

    # ---- DDSP 烘焙 ----
    def _bake_auto_base(self):
        """按最近取色的性质选底座。"""
        import numpy as np

        inst = self.engine.instrument
        ae = getattr(inst, "amp_env", None)
        if ae is not None and len(ae) > 4:
            ae = np.asarray(ae, np.float64)
            if ae[-1] / (ae.max() + 1e-9) < 0.35:
                return "guitar"
        return "glass" if float(getattr(self, "_last_centroid", 6.0)) < 1.6 else "violin"

    def bake_ddsp(self, name, base=None, steps=1000):
        """最近一次取色的 take → 后台子进程微调 DDSP。一次只跑一个。"""
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
        """列出可用行为源(内置+用户导入)。"""
        return self._neural().list_behavior_sources()

    def import_style_from_wav(self, wav_path, name=None, min_notes=2):
        """从 wav 提取行为库。"""
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
        """删除用户导入的行为库。"""
        self._neural().delete_style(name)
        return {"ok": True}
