"""Quill desktop app (pywebview): a WebView draws the HTML UI, the real-time engine runs inside the Python process.

Api is pywebview's js_api object: the page calls its methods through window.pywebview.api.*, and pywebview invokes each method on
its own thread; Api only forwards arguments and manages windows, the engine logic lives in ui/bridge.QuillBridge,
and audio goes through sounddevice. Python pushes back to the page via main_window.evaluate_js, and the page receives it through window.quill* hooks
(quillShowArt, quillLiveBehavior, quillRefreshTones, quillTimbreSaved, etc.).

Run:  python ui/app_web.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # when run as a script, put the repo root on the path so ui/, runtime/ etc. can be imported
UI_HTML = ROOT / "web" / "quill.dc.html"  # main window page, depends on support.js in the same directory
CAPTURE_HTML = ROOT / "web" / "capture.html"  # standalone capture window


class Api:
    """js_api bridge: the UI calls these Python methods through window.pywebview.api.*."""

    def __init__(self):
        self.bridge = None
        self.err = None
        self.main_window = None  # main window reference (updates the colour card after capture)
        self.capture_window = None  # capture window reference (ensures only one is open)
        try:
            from ui.bridge import QuillBridge

            self.bridge = QuillBridge()
            self.bridge.start()  # open the real-time audio stream (sounddevice)
            self.bridge.on_art = self._push_art  # articulation decisions for external MIDI -> main window label
        except Exception as e:  # even if audio fails to start, do not crash the window
            self.err = str(e)
            print("[Api] engine start failed:", e)

    def _push_art(self, art):
        """Push external MIDI decisions back to the main window (called on the mido thread; evaluate_js is marshalled by pywebview)."""
        if self.main_window is not None:
            import json

            try:
                self.main_window.evaluate_js(
                    f"window.quillShowArt&&window.quillShowArt({json.dumps(str(art))})"
                )
            except Exception:
                pass

    def note_on(self, midi, vel=110, art="auto", prev=-1):
        """Return the decided articulation string (when auto); the UI uses it to light the label."""
        if self.bridge:
            return self.bridge.note_on(midi, vel, art, prev)
        return "pluck"

    def note_off(self, midi):
        if self.bridge:
            self.bridge.note_off(midi)
        return True

    def record_start(self):
        if self.bridge:
            self.bridge.record_start()
        return True

    def play_notes(self, notes, bpm=120.0, metro=False, bpb=4):
        """Piano-roll playback: sample-accurate engine scheduling (notes and click on the same clock)."""
        return self.bridge.play_notes(notes, bpm, metro, bpb) if self.bridge else False

    def stop_playback(self):
        return self.bridge.stop_playback() if self.bridge else False

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
        """Piano-roll playback (neural timbre): art field = articulation switch (vibrato/slide/plain);
        instrument = timbre, behavior = style source, traj_gain = trajectory intensity."""
        return (
            self.bridge.play_notes_neural(
                notes, bpm, instrument, behavior, traj_gain, vibrato_source, randomness,
                timing, continuity,
            )
            if self.bridge
            else {"ok": False, "msg": "Engine not started"}
        )

    def try_demo(self):
        """One-click preview: glass × violin behavior, prerendered first, falls back to real-time neural rendering."""
        from pathlib import Path

        demo = Path(__file__).resolve().parent.parent / "assets" / "demos" / "ab_l0ctrl" / "B2b_l0_glass_x_violin_fixed.wav"
        if demo.exists():
            try:
                import sounddevice as sd
                import soundfile as sf

                y, sr = sf.read(str(demo), dtype="float32")
                out = getattr(self.bridge, "_out_dev", None) if self.bridge else None
                sd.play(y, sr, device=out)
                return {"ok": True, "source": "demo_wav", "dur": round(len(y) / sr, 2)}
            except Exception:
                pass
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        notes = [
            {"midi": 72, "start": 0, "len": 0.4, "vel": 100},
            {"midi": 74, "start": 0.5, "len": 0.3, "vel": 90},
            {"midi": 76, "start": 0.9, "len": 0.5, "vel": 95},
            {"midi": 79, "start": 1.5, "len": 0.8, "vel": 100},
            {"midi": 76, "start": 2.5, "len": 0.4, "vel": 85},
            {"midi": 74, "start": 3.0, "len": 0.3, "vel": 80},
            {"midi": 72, "start": 3.4, "len": 1.0, "vel": 100},
        ]
        return self.bridge.play_notes_neural(notes, 120.0, "glass", "violin")

    def list_instruments(self):
        """Neural timbre names available on this machine (the UI's Neural button cycles through them)."""
        if not self.bridge:
            return []
        try:
            return self.bridge.list_instruments()
        except Exception:
            return []

    def list_behavior_sources(self):
        """List available behaviour sources (built-in instruments + user-imported custom libraries)."""
        if not self.bridge:
            return []
        try:
            return self.bridge.list_behavior_sources()
        except Exception:
            return []

    def import_style_wav(self):
        """File dialog to pick a wav -> extract_style -> import as a behaviour source."""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            import webview as wv

            win = self.main_window
            paths = win.create_file_dialog(
                wv.OPEN_DIALOG,
                allow_multiple=False,
                file_types=("Audio (*.wav;*.mp3;*.m4a;*.flac)",),
            )
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}
        if not paths:
            return {"ok": False, "msg": "cancelled"}
        p = paths[0] if isinstance(paths, (list, tuple)) else paths
        return self.bridge.import_style_from_wav(p)

    def delete_behavior_source(self, name):
        if not self.bridge:
            return {"ok": False}
        return self.bridge.delete_behavior_source(name)

    def export_take(self, notes, bpm=120.0, fmt="wav", neural_opts=None):
        """Piano-roll export: save dialog -> MIDI (standard .mid file) or WAV (offline render with the current timbre).
        neural_opts (wav_neural only): {behavior, traj_gain, vibrato_source, randomness}."""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        if not notes:
            return {"ok": False, "msg": "Nothing to export — record or draw notes first"}
        try:
            import webview as wv

            fname = "quill_take." + ("mid" if fmt == "midi" else "wav")
            p = self.main_window.create_file_dialog(wv.SAVE_DIALOG, save_filename=fname)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:50]}
        if not p:
            return {"ok": False, "msg": "cancelled"}
        path = p if isinstance(p, str) else (p[0] if p else None)
        return self.bridge.export_take(notes, bpm, fmt, path, neural_opts)

    def click_on(self, bpm, bpb=4):
        return self.bridge.click_on(bpm, bpb) if self.bridge else False

    def click_off(self):
        return self.bridge.click_off() if self.bridge else False

    def list_devices(self):
        return (
            self.bridge.list_devices()
            if self.bridge
            else {"input": [], "output": [], "midi": [], "engine": {}}
        )

    def set_input_device(self, idx):
        return self.bridge.set_input_device(idx) if self.bridge else {"ok": False}

    def set_output_device(self, idx):
        return self.bridge.set_output_device(idx) if self.bridge else {"ok": False}

    def set_midi_input(self, name):
        return self.bridge.set_midi_input(name) if self.bridge else {"ok": False}

    def probe_input(self, ms=350):
        return self.bridge.probe_input(ms) if self.bridge else {"ok": False}

    def save_session(self, state):
        """Persist UI state (knobs/ADSR/mode/behaviour library/piano roll); the UI debounces, this only validates + writes atomically."""
        return self.bridge.save_session(state) if self.bridge else {"ok": False, "msg": "Engine not started"}

    def load_session(self):
        """Last session; {} if missing/corrupt."""
        return self.bridge.load_session() if self.bridge else {}

    def get_status(self):
        """Status bar data: MIDI device name and measured output stream latency."""
        if not self.bridge:
            return {"midi": None, "latency_ms": None}
        try:
            lat = self.bridge.engine.stream_latency_ms()
            lat = round(float(lat), 1) if lat == lat else None  # NaN guard
        except Exception:
            lat = None
        return {"midi": self.bridge.midi_name(), "latency_ms": lat}

    def record_poll(self):
        """While recording, fetch the recorded notes + duration so far (polled by the piano roll for follow-scrolling)."""
        return self.bridge.record_poll() if self.bridge else {"notes": [], "t": 0.0}

    def record_stop(self):
        """Stop recording and return the recorded notes [{midi,start,len,vel}] (seconds) for the piano roll."""
        return self.bridge.record_stop() if self.bridge else []

    def set_param(self, name, value):
        if self.bridge:
            self.bridge.set_param(name, value)
        return True

    def set_adsr(self, a, d, s, r):
        if self.bridge:
            self.bridge.set_adsr(a, d, s, r)
        return True

    # ---- Capture (standalone window) ----
    def open_capture(self, mode=None):
        """Capture clicked in the main UI -> open the standalone capture window. If already open, do not create another (only one allowed).
        mode='song': go straight into extraction from a song after opening (pick file -> separate), matching FROM A SONG in the main window."""
        if self.capture_window is not None:  # capture window exists: bring it to the front, do not create a new one
            try:
                self.capture_window.restore()  # restore first if minimised
                self.capture_window.show()  # bring to front/focus
            except Exception:
                pass
            if mode == "song":
                try:
                    self.capture_window.evaluate_js("typeof fromSong==='function'&&fromSong()")
                except Exception:
                    pass
            return True
        win = webview.create_window(
            "Capture",
            str(CAPTURE_HTML),
            js_api=self,
            width=400,
            height=612,
            resizable=False,  # 612: fits the NOISE row and the stem picker
            background_color="#15130e",
        )
        self.capture_window = win

        def _closed():
            self.capture_window = None  # clear the reference on close so it can be opened again

        try:
            win.events.closed += _closed
        except Exception:
            pass
        if mode == "song":  # wait for the page to load before triggering song selection

            def _loaded():
                try:
                    win.evaluate_js("typeof fromSong==='function'&&fromSong()")
                except Exception:
                    pass

            try:
                win.events.loaded += _loaded
            except Exception:
                pass
        return True

    def capture_start(self):
        return self.bridge.capture_start() if self.bridge else False

    def set_voice_mode(self, mode="solo"):
        return self.bridge.set_voice_mode(mode) if self.bridge else "solo"

    def set_live_behavior(self, name=None):
        """Which behaviour library drives live playing (empty = off, rule path). Loading happens in the background; ready/failure is pushed to the main window
        (`window.quillLiveBehavior(name, ready, error)`), and key presses use the rule path in the meantime."""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            r = self.bridge.set_live_behavior(name)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:80]}
        if r.get("ok") and not r.get("ready"):
            import json
            import threading
            import time

            def _watch():
                for _ in range(300):  # wait at most 30s
                    st = self.bridge.live_behavior_status()
                    if st["ready"] or st["error"] or st["name"] != name:
                        break
                    time.sleep(0.1)
                if not st["ready"] and not st["error"] and st["name"] == name:
                    # the warm-up thread is stuck (checkpoint unreadable / import hangs): go back to the rule path
                    # and push the timeout back to the page as an error.
                    try:
                        self.bridge.set_live_behavior(None)
                    except Exception:
                        pass
                    st = {"name": name, "ready": False, "error": f"behavior {name!r} did not load in 30 s — back to rules"}
                self._refresh_main(
                    "window.quillLiveBehavior&&window.quillLiveBehavior("
                    f"{json.dumps(st['name'])},{json.dumps(bool(st['ready']))},{json.dumps(st['error'])})"
                )

            threading.Thread(target=_watch, daemon=True, name="live-behavior-watch").start()
        return r

    def live_behavior_status(self):
        return self.bridge.live_behavior_status() if self.bridge else {"name": None, "ready": False}

    def set_capture_mode(self, texture=False):
        """Texture mode switch (non-musical capture such as rain/wind/machines)."""
        return self.bridge.set_capture_mode(texture) if self.bridge else False

    def capture_level(self):
        """Live level while recording (polled by the capture window's level meter)."""
        return self.bridge.capture_level() if self.bridge else 0.0

    def play_take(self, t0=None, t1=None):
        """Capture window preview transport: play the take/selection (aligned to waveform coordinates)."""
        return self.bridge.play_take(t0, t1) if self.bridge else {"ok": False}

    def stop_take(self):
        return self.bridge.stop_take() if self.bridge else True

    def capture_meter(self):
        """Main window SOURCE live waveform: level + whether recording (shared bridge, so the main window follows when the capture window records)."""
        return self.bridge.capture_meter() if self.bridge else {"lvl": 0.0, "active": False}

    def _push_tone_name(self, r, captured=True):
        """Sync the main window after a successful capture/timbre change: TONE·COLOUR name + name field + top preset name + SOURCE true f0.

        captured=True means the engine is now playing an unsaved new capture and the page's pairing should follow it;
        pass False when loading a saved timbre, otherwise the pairing would be mislabelled "(unsaved)"."""
        if r.get("ok") and self.main_window is not None:
            import json
            import math

            nm = json.dumps(r.get("disp") or r.get("name") or "Capture")
            js = (
                f"var e=document.getElementById('quillToneName'); if(e) e.textContent={nm};"
                f"var i=document.getElementById('quillToneInput'); if(i) i.value={nm};"
            )
            f0 = r.get("f0")
            if f0:
                midi = int(round(69 + 12 * math.log2(float(f0) / 440.0)))
                names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                nn = f"{names[midi % 12]}{midi // 12 - 1}"
                js += (
                    f"var s=document.getElementById('quillSrcF0');"
                    f" if(s) s.textContent='f\u2080 {float(f0):.1f} Hz \u00b7 {nn}';"
                )
            wave = r.get("wave")
            if wave:
                js += f"window._quillWave={json.dumps(wave)};window._quillWaveT0=performance.now();"
            if r.get("noise") is not None:  # automatic noise floor -> the Air/Noise knob follows
                js += f"window.quillSetKnob&&window.quillSetKnob('noise',{float(r['noise']):.3f});"
            # animation: name fade-in and timbre-card light sweep, in time with the capture action
            js += (
                "var tn=document.getElementById('quillToneName');"
                " if(tn){tn.classList.remove('nameSwap'); void tn.offsetWidth; tn.classList.add('nameSwap');}"
                "var tc=document.getElementById('quillToneCard');"
                " if(tc){tc.classList.remove('sweepGo'); void tc.offsetWidth; tc.classList.add('sweepGo');}"
            )
            if captured:  # the engine now plays the (unsaved) capture: let the page's pair follow it
                js += f"window.quillCaptured&&window.quillCaptured({nm});"
            try:
                self.main_window.evaluate_js(js)
            except Exception:
                pass

    def capture_stop(self):
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            r = self.bridge.capture_stop()
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        self._push_tone_name(r)
        return r

    def open_file_capture(self):
        """OPEN FILE... button: file dialog -> capture directly (for single-note/single-instrument material; whole songs go through pick_song)."""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            import webview as wv

            win = self.capture_window or self.main_window
            paths = win.create_file_dialog(wv.OPEN_DIALOG, allow_multiple=False)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}
        if not paths:
            return {"ok": False, "msg": "No file chosen"}
        p = paths[0] if isinstance(paths, (list, tuple)) else paths
        r = self.capture_from_file(p)
        self._push_tone_name(r)
        return r

    # ---- Extract an instrument from a song (Demucs separation -> pick a stem and capture) ----
    def pick_song(self):
        """File dialog to pick a song -> separate -> return the list of selectable stems."""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            import webview as wv

            win = self.capture_window or self.main_window
            paths = win.create_file_dialog(wv.OPEN_DIALOG, allow_multiple=False)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}
        if not paths:
            return {"ok": False, "msg": "No file chosen"}
        p = paths[0] if isinstance(paths, (list, tuple)) else paths
        return self.bridge.separate_stems(p)

    def capture_stem(self, path):
        """Capture from one separated stem (the keyboard plays it immediately)."""
        r = self.capture_from_file(path)
        self._push_tone_name(r)
        return r

    def capture_from_file(self, path):
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            return self.bridge.capture_from_file(path)
        except Exception as e:  # non-audio file, decode failure: the page must get msg, not an exception
            return {"ok": False, "msg": str(e)[:70]}

    def refine_capture(
        self, f0_hint=None, lo_hz=None, hi_hz=None, t0=None, t1=None, denoise="auto"
    ):
        """Refine the latest capture: reference pitch / low cut high cut / drag-selected segment / denoise level -> reanalyse -> hot swap + push animation."""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            r = self.bridge.refine_capture(f0_hint, lo_hz, hi_hz, t0, t1, denoise)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        self._push_tone_name(r)
        return r

    def capture_path(self, path):
        """Capture from a dragged file (pywebview gives the real path): analyse -> hot swap -> push animation."""
        r = self.capture_from_file(path)
        self._push_tone_name(r)
        return r

    def capture_bytes(self, b64, name="drop"):
        """Drag fallback: the front end sends base64 bytes -> write a temp file -> capture."""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        import base64
        import os
        import tempfile

        try:
            data = base64.b64decode(b64)
            ext = os.path.splitext(name)[1] or ".wav"
            fd, tmp = tempfile.mkstemp(suffix=ext, prefix="quill_drop_")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            try:
                r = self.capture_from_file(tmp)
            finally:
                os.unlink(tmp)
            self._push_tone_name(r)
            return r
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}

    # ---- Timbre library (save/list/load/delete) ----
    def list_tones(self):
        return self.bridge.list_tones() if self.bridge else []

    def _refresh_main(self, js):
        if self.main_window is not None:
            try:
                self.main_window.evaluate_js(js)
            except Exception:
                pass

    def save_timbre(self, name):
        """Save the timbre; on success tell the main window to refresh SAVED TONES and the piano-roll Neural list (new tone: entry)."""
        r = self.bridge.save_timbre(name) if self.bridge else {"ok": False}
        if r.get("ok"):
            import json

            self._refresh_main("window.quillRefreshTones&&window.quillRefreshTones()")
            self._refresh_main("window.quillRefreshInstruments&&window.quillRefreshInstruments()")
            # saved from the capture window → the main window's pair follows the new tone
            self._refresh_main(f"window.quillTimbreSaved&&window.quillTimbreSaved({json.dumps(str(r.get('name')))},{json.dumps(bool((r.get('behavior') or {}).get('state') == 'extracting'))})")
            self._beh_announced = False
        return r

    def behavior_status(self):
        """Background extraction progress of the same-named behaviour library after saving a timbre; refreshes the piano roll's behaviour source list when done."""
        if not self.bridge:
            return {"state": "idle"}
        d = self.bridge.behavior_status()
        if d.get("state") == "done" and not getattr(self, "_beh_announced", False):
            import json

            self._beh_announced = True
            res = d.get("result") or {}
            n = int(res.get("n_notes", 0) or 0) if res.get("ok") else 0
            self._refresh_main("window.quillBehaviorsChanged&&window.quillBehaviorsChanged()")
            # one recording → two assets: tell the main window so it can offer the cross-use right away
            self._refresh_main(f"window.quillAssetsNew&&window.quillAssetsNew({json.dumps(str(d.get('name') or ''))},{n})")
        return d

    def bake_ddsp(self, name, base=None, steps=1000):
        """Bake a DDSP timbre after capture (background subprocess); progress is polled through bake_status."""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        return self.bridge.bake_ddsp(name, base or None, steps)

    def bake_status(self):
        if not self.bridge:
            return {"state": "idle"}
        d = self.bridge.bake_status()
        if d.get("state") == "done" and not getattr(self, "_bake_announced", False):
            self._bake_announced = True
            self._refresh_main("window.quillRefreshTones&&window.quillRefreshTones()")
            self._refresh_main("window.quillRefreshInstruments&&window.quillRefreshInstruments()")
        elif d.get("state") != "done":
            self._bake_announced = False
        return d

    def rename_tone(self, name, new_name):
        return self.bridge.rename_tone(name, new_name) if self.bridge else {"ok": False}

    def load_tone(self, name):
        if not self.bridge:
            return {"ok": False}
        r = self.bridge.load_tone(name)
        self._push_tone_name(r, captured=False)  # name/preset/SOURCE f0 synced in one go; not a new capture
        return r

    def delete_tone(self, name):
        return self.bridge.delete_tone(name) if self.bridge else {"ok": False}


def _set_dock_icon():
    """macOS Dock icon = brand Q seal (assets/brand); silently skipped on non-macOS / without pyobjc."""
    icon = ROOT.parent / "assets" / "brand" / "quill_icon_1024.png"
    if not icon.exists():
        return
    try:
        from AppKit import NSApplication, NSImage

        img = NSImage.alloc().initWithContentsOfFile_(str(icon))
        if img:
            NSApplication.sharedApplication().setApplicationIconImage_(img)
    except Exception:
        pass


def main():
    if not UI_HTML.exists():
        print(f"UI missing: {UI_HTML}")
        return 1
    api = Api()
    # the page is designed at a fixed 1440×900 and scaled uniformly: content does not distort, windows not at a 1.6 ratio get same-coloured dark margins.
    api.main_window = webview.create_window(
        "Quill",
        str(UI_HTML),
        js_api=api,
        width=1280,
        height=800,
        min_size=(960, 600),
        background_color="#15130e",
    )
    webview.start(_set_dock_icon)
    if api.bridge:
        api.bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
