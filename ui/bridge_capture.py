"""Microphone capture, Demucs separation and timbre library management (Bridge mixin).

Capture flow: capture_start opens its own InputStream (44.1 kHz mono, buffers at most 65 s);
capture_stop or capture_from_file hands the audio to _capture_audio: silence trimming, spectral-gate denoising,
analyze_timbre gives the snapshot, and after build_instrument it is hot-swapped into the engine. User timbres are .npz files under USER_TONES;
_tone_path ensures names cannot escape that directory.
"""

from __future__ import annotations

from pathlib import Path


class BridgeCaptureMixin:

    def _close_cap_stream(self):
        st = getattr(self, "_cap_stream", None)
        if st is not None:
            try:
                st.stop()
                st.close()
            except Exception:
                pass
            self._cap_stream = None

    def capture_start(self):
        """Start microphone recording (its own InputStream)."""
        import sounddevice as sd

        self._close_cap_stream()
        self._cap_frames = []
        self._cap_stream = sd.InputStream(
            samplerate=44100,
            channels=1,
            device=self._in_dev,
            callback=self._cap_cb,
        )
        self._cap_stream.start()
        return True

    def _cap_cb(self, indata, frames, t, status):
        if len(self._cap_frames) * indata.shape[0] < 44100 * 65:
            self._cap_frames.append(indata[:, 0].copy())

    def capture_level(self):
        """Live level while recording (~50ms RMS -> 0..1)."""
        fr = getattr(self, "_cap_frames", None)
        if not fr:
            return 0.0
        import numpy as np

        x = np.concatenate(fr[-3:]) if len(fr) > 1 else fr[-1]
        return float(min(1.0, np.sqrt(float((x**2).mean())) * 4.0))

    def capture_active(self):
        """Whether the recording stream is running."""
        st = getattr(self, "_cap_stream", None)
        try:
            return bool(st is not None and st.active)
        except Exception:
            return False

    def capture_meter(self):
        """Get the live level + whether recording in one call."""
        return {"lvl": self.capture_level(), "active": self.capture_active()}

    def capture_stop(self):
        """Stop recording -> analyze_timbre capture -> hot-swap the current instrument."""
        import numpy as np

        try:
            self._cap_stream.stop()
            self._cap_stream.close()
        except Exception:
            pass
        frames = getattr(self, "_cap_frames", [])
        y = np.concatenate(frames).astype(np.float32) if frames else np.zeros(4410, np.float32)
        if len(y) > 22050 and float(np.abs(y).max()) < 1e-5:
            dev = "input"
            try:
                import sounddevice as sd

                dev = sd.query_devices(kind="input")["name"]
            except Exception:
                pass
            return {
                "ok": False,
                "msg": f"Silent input — '{dev}' gave no signal. "
                f"Pick your real mic: System Settings → Sound → Input.",
            }
        return self._capture_audio(y, 44100, name="capture")

    def separate_stems(self, path):
        """Separate instruments from a song: Demucs htdemucs_6s on the first 30 s on CPU; silent stems are not returned."""
        import tempfile

        import numpy as np
        import soundfile as sf

        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model

            from analysis.io import load
        except Exception as e:
            return {"ok": False, "msg": f"Demucs unavailable: {str(e)[:50]}"}
        try:
            y, sr = load(str(path), sr=44100)
            y = y[: 44100 * 30]
            if self._sep_model is None:
                m = get_model("htdemucs_6s")
                m.eval()
                self._sep_model = m
            model = self._sep_model
            wav = torch.tensor(np.asarray(y, np.float32))
            if wav.ndim == 1:
                wav = wav[None].repeat(2, 1)
            with torch.no_grad():
                out = apply_model(model, wav[None], device="cpu", progress=False)[0]
            prev = getattr(self, "_stems_dir", None)
            if prev:
                import shutil

                shutil.rmtree(prev, ignore_errors=True)
            outdir = Path(tempfile.mkdtemp(prefix="quill_stems_"))
            self._stems_dir = str(outdir)
            stems = []
            for i, name in enumerate(model.sources):
                x = out[i].mean(0).numpy()
                rms = float(np.sqrt((x**2).mean()))
                if rms < 1e-4:
                    continue
                p = outdir / f"{name}.wav"
                sf.write(str(p), x, 44100)
                stems.append({"name": name, "path": str(p), "rms": round(rms, 4)})
            stems.sort(key=lambda s: -s["rms"])
            return {"ok": True, "stems": stems}
        except Exception as e:
            return {"ok": False, "msg": str(e)[:80]}

    def capture_from_file(self, path):
        """Capture from a dropped file."""
        from analysis.io import load

        y, sr = load(str(path), sr=44100)
        return self._capture_audio(y, sr, name=Path(path).stem)

    def set_voice_mode(self, mode="solo"):
        self._voice_mode = "poly" if str(mode) == "poly" else "solo"
        return self._voice_mode

    def set_capture_mode(self, texture=False):
        self._texture_mode = bool(texture)
        return True

    @staticmethod
    def _tone_path(name):
        """User tone name → path inside USER_TONES, or None if the name would escape the
        directory (``../x``, absolute paths). Names come from the UI, but the filesystem
        boundary is enforced here so no caller can delete or load outside the tone folder."""
        from .bridge import USER_TONES

        nm = str(name or "")
        if not nm or nm in (".", "..") or "/" in nm or "\\" in nm or "\x00" in nm:
            return None
        root = USER_TONES.resolve()
        p = (USER_TONES / (nm + ".npz")).resolve()
        return p if p.parent == root else None

    @staticmethod
    def _count_onsets(y, sr):
        """Rough note count of a take (librosa onset detector). Drives the capture page's
        'behavior library needs 2+ notes' hint; None when the detector is unavailable."""
        try:
            import librosa
            import numpy as np

            yy = np.asarray(y, dtype=np.float32)
            if yy.ndim > 1:
                yy = yy.mean(axis=1)
            if len(yy) < sr // 4:
                return 0
            on = librosa.onset.onset_detect(y=yy, sr=sr, hop_length=512, backtrack=False, units="frames")
            return int(len(on))
        except Exception:
            return None

    def _capture_audio(self, y, sr, name="capture", f0_hint=None, keep_raw=True, gate=1.0):
        import warnings

        import numpy as np
        import soundfile as sf

        from analysis.io import trim_silence
        from analysis.timbre import analyze_timbre
        from synth.instrument import build_instrument

        from .bridge import USER_TONES

        tex = bool(getattr(self, "_texture_mode", False))
        USER_TONES.mkdir(parents=True, exist_ok=True)
        y = np.asarray(y, np.float32)
        if keep_raw:
            try:
                sf.write(str(USER_TONES / "_last_take_raw.wav"), y, sr)
            except Exception:
                pass
        y = trim_silence(y, sr)
        if len(y) < int(0.4 * sr):
            return {"ok": False, "msg": "Too short — hold a steady note for 1s+"}
        if not tex and gate > 0:
            from analysis.denoise import spectral_gate

            y = spectral_gate(y, sr, strength=float(gate))
        try:
            sf.write(str(USER_TONES / "_last_take_gated.wav"), y, sr)
        except Exception:
            pass
        if tex and f0_hint is None:
            seg = y[: int(4 * sr)].astype(np.float64)
            S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
            band = (fr > 60) & (fr < 1000)
            f0_hint = (
                float(fr[band][np.argmax(S[band])]) if band.any() and S[band].max() > 0 else 220.0
            )
            f0_hint = min(max(f0_hint, 60.0), 1000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snap = analyze_timbre(
                y[: int(6 * sr)] if tex else y,
                sr=sr,
                name=name,
                denoise=not tex,
                window=None if tex else "auto",
                f0_hint=f0_hint,
                cross_check=False,
            )
        if not (np.isfinite(snap.get("f0_ref", 0)) and snap["f0_ref"] > 0):
            return {
                "ok": False,
                "msg": "No stable pitch — hum or sing a held note (a moving melody is fine, just hold one note somewhere)",
            }
        meta = snap.get("meta", {}) or {}
        warn = None
        if not tex:
            vr = float(meta.get("voiced_ratio", 1.0))
            cv = float(meta.get("f0_cv", 0.0))
            win_ok = meta.get("window_found", True)
            if not win_ok and (vr < 0.5 or cv > 0.15):
                if vr >= 0.85:
                    return {
                        "ok": False,
                        "msg": "Sounds like several notes at once — use ♫ From a song to pull out one instrument",
                    }
                return {
                    "ok": False,
                    "msg": "No steady note to capture — hum or hold a single pitch for ~1 second (or turn on Texture mode)",
                }
            if cv > 0.09:
                warn = "Tip: hold a steadier note for a cleaner capture"
        try:
            from snapshot import save_snapshot

            save_snapshot(str(USER_TONES / "_last_capture.npz"), snap)
        except Exception:
            pass
        meta2 = dict(snap.get("meta", {}) or {})
        meta2["texture"] = tex
        snap["meta"] = meta2
        inst = build_instrument(
            snap, noise_level=0.6 if tex else 0.02, pick=float(self.params.get("pick", 0.5))
        )
        self._balance(inst)
        Hs = np.asarray(inst.H_steady, np.float64)
        nl = 0.6 if tex else self._auto_noise_level(Hs)
        inst.noise_level = nl
        meta2["noise_level"] = nl
        self.params["noise"] = nl
        self.engine.set_instrument(inst)
        idx = np.arange(1, len(Hs) + 1)
        centroid = float((idx * Hs).sum() / (Hs.sum() + 1e-9))
        adj = (
            "Texture"
            if tex
            else "Bright"
            if centroid > 8
            else ("Warm" if centroid > 4 else "Mellow")
        )
        self._last_snap = snap
        self._last_centroid = centroid
        take_s = round(len(y) / sr, 1)
        win_used = bool((snap.get("meta", {}) or {}).get("window_found"))
        n_bins = 96
        L = max(1, len(y) // n_bins)
        wave = [float(np.sqrt((y[i * L : (i + 1) * L] ** 2).mean())) for i in range(n_bins)]
        wmax = max(wave) or 1.0
        wave = [round(v / wmax, 3) for v in wave]
        n_onsets = self._count_onsets(y, sr)
        return {
            "ok": True,
            "name": name,
            "disp": adj + " Capture",
            "warn": warn,
            "wave": wave,
            "take_s": take_s,
            "n_onsets": n_onsets,
            "win_used": win_used,
            "f0": round(float(snap["f0_ref"]), 1),
            "n_harm": int((Hs > Hs.max() * 0.01).sum()),
            "harm": [float(x) for x in (Hs / (Hs.max() + 1e-9))[:24]],
            "noise": nl,
        }

    def play_take(self, t0=None, t1=None):
        """Preview the latest take."""
        import sounddevice as sd
        import soundfile as sf

        from .bridge import USER_TONES

        p = USER_TONES / "_last_take_gated.wav"
        if not p.exists():
            return {"ok": False, "msg": "Nothing captured yet"}
        y, sr = sf.read(str(p), dtype="float32")
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        a = int(max(0.0, float(t0 or 0.0)) * sr)
        b = int(float(t1) * sr) if t1 else len(y)
        seg = y[a:b]
        if len(seg) < 100:
            return {"ok": False, "msg": "Empty span"}
        try:
            sd.play(seg, sr, device=self._out_dev)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        return {"ok": True, "dur": round(len(seg) / sr, 3)}

    def stop_take(self):
        import sounddevice as sd

        try:
            sd.stop()
        except Exception:
            pass
        return True

    def refine_capture(
        self, f0_hint=None, lo_hz=None, hi_hz=None, t0=None, t1=None, denoise="auto"
    ):
        """Refine the latest capture."""
        import numpy as np
        import soundfile as sf

        from .bridge import USER_TONES

        raw = USER_TONES / "_last_take_raw.wav"
        if not raw.exists():
            return {"ok": False, "msg": "Nothing captured yet"}
        y, sr = sf.read(str(raw), dtype="float32")
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if lo_hz or hi_hz:
            from scipy.signal import butter, sosfiltfilt

            lo = max(20.0, float(lo_hz or 20.0))
            hi = min(sr / 2 - 200.0, float(hi_hz or sr / 2 - 200.0))
            if hi > lo + 50:
                sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
                y = sosfiltfilt(sos, np.asarray(y, np.float64)).astype(np.float32)
        if t0 is not None or t1 is not None:
            from analysis.io import trim_silence

            y = trim_silence(y, sr)
            a = int(max(0.0, float(t0 or 0.0)) * sr)
            b = int(float(t1) * sr) if t1 else len(y)
            if b - a < int(0.35 * sr):
                return {"ok": False, "msg": "Selection too short — drag a longer span"}
            y = y[a:b]
        gate = {"off": 0.0, "auto": 1.0, "strong": 1.6}.get(str(denoise), 1.0)
        return self._capture_audio(
            y,
            sr,
            name="capture",
            f0_hint=(float(f0_hint) if f0_hint else None),
            keep_raw=False,
            gate=gate,
        )

    # ---- Timbre library ----
    @staticmethod
    def _tone_css(c, name=""):
        t = max(0.0, min(1.0, (float(c) - 2) / 12.0))
        lerp = lambda a, b, u: [a[i] + (b[i] - a[i]) * u for i in range(3)]
        s_t, g_t, r_t = (
            (190, 206, 220),
            (226, 192, 110),
            (234, 168, 146),
        )
        s_b, g_b, r_b = (116, 150, 184), (196, 148, 40), (212, 106, 82)
        top, bot = (
            (lerp(s_t, g_t, t * 2), lerp(s_b, g_b, t * 2))
            if t < 0.5
            else (lerp(g_t, r_t, (t - 0.5) * 2), lerp(g_b, r_b, (t - 0.5) * 2))
        )
        hx = lambda rgb: "".join(f"{int(v):02x}" for v in rgb)
        return (
            "linear-gradient(180deg,rgba(255,255,255,.44),rgba(255,255,255,.06) 15%,transparent 34%),"
            "linear-gradient(0deg,rgba(40,30,14,.23),transparent 42%),"
            f"linear-gradient(160deg,#{hx(top)},#{hx(bot)})"
        )

    def list_tones(self):
        """Default timbre + user-saved timbres."""
        from .bridge import USER_TONES

        out = [
            {
                "name": "__default__",
                "disp": "Synth Pad",
                "builtin": True,
                "css": self._tone_css(6.0, "Synth Pad"),
            }
        ]
        try:
            from snapshot import load_snapshot

            for f in sorted(USER_TONES.glob("*.npz")):
                if f.stem.startswith("_"):
                    continue
                disp, c = f.stem, 6.0
                try:
                    m = load_snapshot(str(f)).get("meta", {}) or {}
                    disp = m.get("disp", f.stem)
                    c = float(m.get("centroid", 6.0))
                except Exception:
                    pass
                out.append(
                    {"name": f.stem, "disp": disp, "builtin": False, "css": self._tone_css(c, disp)}
                )
        except Exception:
            pass
        return out

    def save_timbre(self, name):
        """Save the most recently captured timbre under a name."""
        from snapshot import save_snapshot

        from .bridge import USER_TONES

        if self._last_snap is None:
            return {"ok": False, "msg": "Capture a tone first"}
        safe = self._safe_name(name)
        snap = dict(self._last_snap)
        meta = dict(snap.get("meta", {}) or {})
        meta["disp"] = str(name).strip() or safe
        meta["centroid"] = self._last_centroid
        snap["meta"] = meta
        USER_TONES.mkdir(parents=True, exist_ok=True)
        save_snapshot(str(USER_TONES / safe), snap)
        import shutil
        import tempfile
        import threading

        take = USER_TONES / "_last_take_raw.wav"
        if not take.exists():
            take = USER_TONES / "_last_take_gated.wav"
        job = {"name": safe, "state": "extracting", "result": None}
        if take.exists():
            tmp = Path(tempfile.mkdtemp(prefix="quill_beh_")) / f"{safe}.wav"
            shutil.copy2(str(take), str(tmp))

            def _run():
                try:
                    result = self.import_style_from_wav(tmp, name=safe)
                except Exception as e:
                    result = {"ok": False, "msg": str(e)[:80]}
                finally:
                    with self._state_lock:
                        job["result"] = result
                        job["state"] = "done"
                    shutil.rmtree(tmp.parent, ignore_errors=True)

            threading.Thread(target=_run, daemon=True, name="behavior-extract").start()
        else:
            job["state"], job["result"] = "done", {"ok": False, "msg": "no take file"}
        self._beh_job = job
        return {"ok": True, "name": safe, "disp": meta["disp"], "behavior": {"state": job["state"]}}

    def behavior_status(self):
        job = getattr(self, "_beh_job", None)
        if job is None:
            return {"state": "idle"}
        return {"state": job["state"], "name": job["name"], "result": job["result"]}

    def load_tone(self, name):
        """Load a timbre from the library -> hot swap."""
        import numpy as np

        from snapshot import load_snapshot
        from synth.instrument import build_instrument


        path = self._default_path if name == "__default__" else self._tone_path(name)
        if path is None or not Path(path).exists():
            return {"ok": False, "msg": "Tone not found"}
        snap = load_snapshot(str(path))
        meta = snap.get("meta", {}) or {}
        tex = bool(meta.get("texture"))
        inst = build_instrument(
            snap, noise_level=0.6 if tex else 0.02, pick=float(self.params.get("pick", 0.5))
        )
        self._balance(inst)
        Hs = np.asarray(inst.H_steady, np.float64)
        nl = 0.6 if tex else float(meta.get("noise_level") or self._auto_noise_level(Hs))
        inst.noise_level = nl
        self.params["noise"] = nl
        self.engine.set_instrument(inst)
        self._last_snap = snap
        disp = "Synth Pad" if name == "__default__" else meta.get("disp", name)
        idx = np.arange(1, len(Hs) + 1)
        self._last_centroid = float((idx * Hs).sum() / (Hs.sum() + 1e-9))
        return {
            "ok": True,
            "name": name,
            "disp": disp,
            "f0": round(float(snap.get("f0_ref", 0)) or 0, 1),
            "harm": [float(x) for x in (Hs / (Hs.max() + 1e-9))[:24]],
            "noise": nl,
        }

    def rename_tone(self, name, new_name):
        from snapshot import load_snapshot, save_snapshot

        from .bridge import USER_TONES

        if name == "__default__":
            return {"ok": False, "msg": "Cannot rename default"}
        src = self._tone_path(name)
        if src is None or not src.exists():
            return {"ok": False, "msg": "Tone not found"}
        safe = self._safe_name(new_name, "")
        if not safe:
            return {"ok": False, "msg": "Empty name"}
        dst = USER_TONES / (safe + ".npz")
        if dst.exists() and dst != src:
            return {"ok": False, "msg": "Name already taken"}
        try:
            snap = load_snapshot(str(src))
            meta = dict(snap.get("meta", {}) or {})
            meta["disp"] = str(new_name).strip() or safe
            snap["meta"] = meta
            save_snapshot(str(USER_TONES / safe), snap)
            if dst != src:
                src.unlink()
            try:
                from neural.neural_take import USER_STYLES_DIR
                beh_src = USER_STYLES_DIR / f"{name}.npz"
                if beh_src.exists() and safe != name:
                    beh_src.rename(USER_STYLES_DIR / f"{safe}.npz")
                    self._neural()._custom_nns.pop(name, None)
            except Exception:
                pass
            return {"ok": True, "name": safe, "disp": meta["disp"]}
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}

    def delete_tone(self, name):

        if name == "__default__":
            return {"ok": False, "msg": "Cannot delete default"}
        path = self._tone_path(name)
        if path is None:
            return {"ok": False, "msg": "Invalid tone name"}
        try:
            path.unlink()
            try:
                self._neural().delete_style(str(name))
            except Exception:
                pass
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}

    def set_adsr(self, a, d, s, r):
        self.engine.adsr = {
            "attack_ms": 5.0 + float(a) * 1000.0,
            "decay_ms": 20.0 + float(d) * 1000.0,
            "sustain": float(s),
            "release_ms": 50.0 + float(r) * 1400.0,
        }
        return True
