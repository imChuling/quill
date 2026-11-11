"""Quill app bridge: UI events -> real-time engine (runtime/engine).

Key presses/knobs in the WebView call in here through the pywebview js_api, and pywebview invokes each method on its own thread;
this class pushes events into the engine's SimpleQueue (the same path as midi_in), and the engine consumes them and sounds on the
sounddevice audio callback thread. UI-side calls only enqueue, change scalars or replace whole objects, never doing work on the callback thread.
Features are split into mixins by area: devices, neural rendering, capture, export, session.
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.engine import Event  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402  single source of truth for articulation decisions
from ui.bridge_capture import BridgeCaptureMixin  # noqa: E402
from ui.bridge_device import BridgeDeviceMixin  # noqa: E402
from ui.bridge_export import BridgeExportMixin  # noqa: E402
from ui.bridge_neural import BridgeNeuralMixin  # noqa: E402
from ui.bridge_session import BridgeSessionMixin  # noqa: E402

DEFAULT_SNAPSHOT = ROOT / "assets" / "snapshots" / "synth_pad.npz"  # warm sustained pad, default timbre at startup
USER_TONES = ROOT / "assets" / "user_tones"  # user-saved capture timbres (.npz)


class _RecQueue:
    """Event queue proxy: every note source (on-screen keyboard/computer keyboard/external MIDI) enqueues through here;
    while recording, each event is timestamped into log. The engine only calls get_nowait(), forwarded transparently."""

    def __init__(self):
        self._q: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self.rec = False
        self._t0 = 0.0
        self.log = []  # [(t_sec, Event)]

    def arm(self):
        self.log = []
        self._t0 = time.perf_counter()
        self.rec = True

    def disarm(self):
        self.rec = False
        out = self.log
        self.log = []  # cleared on stop; the recording is carried away by the return value
        return out

    def put(self, ev):
        if self.rec:
            self.log.append((time.perf_counter() - self._t0, ev))
        self._q.put(ev)

    def get_nowait(self):
        return self._q.get_nowait()


class _MidiRouter:
    """External MIDI -> the same articulation decision path as the on-screen keyboard (on/off go through note_on/off, the rest pass straight to the queue)."""

    def __init__(self, bridge):
        self._b = bridge

    def put(self, ev):
        if ev.kind == "on":
            art = self._b.note_on(ev.note, ev.vel, "auto")
            hook = getattr(self._b, "on_art", None)  # MIDI path: push the decision back to the UI label
            if hook and art:
                try:
                    hook(art)
                except Exception:
                    pass
        elif ev.kind == "off":
            self._b.note_off(ev.note)
        else:
            self._b._q.put(ev)  # bend/vibrato/pedal pass straight through


class QuillBridge(
    BridgeDeviceMixin,
    BridgeNeuralMixin,
    BridgeCaptureMixin,
    BridgeExportMixin,
    BridgeSessionMixin,
):
    """Real-time engine bridge: performance events, knobs, timbre hot swap; the other feature areas live in the mixins."""

    def __init__(
        self, snapshot=DEFAULT_SNAPSHOT, noise_level=0.02
    ):  # noise layer defaults to 2%, adjustable with the Noise knob
        import threading

        import numpy as np

        from runtime.engine import Engine
        from snapshot import load_snapshot
        from synth.instrument import build_instrument

        self._state_lock = threading.Lock()
        self._q = _RecQueue()  # recording proxy queue (where all note sources meet)
        snap = load_snapshot(str(snapshot))
        inst = build_instrument(snap, noise_level=noise_level)
        # the steady energy of the default timbre is the reference for balancing capture timbre volume
        self._ref_energy = float(np.sqrt((np.asarray(inst.H_steady, np.float64) ** 2).sum())) or 0.3
        self.engine = Engine(inst, eval_mode=False)
        self.params = {}
        self._running = False
        self._midi = None
        self._default_path = Path(snapshot)  # timbre library: default timbre snapshot
        self._last_snap = None  # most recently captured/loaded snapshot (for saving)
        self._last_centroid = 6.0
        self._gi = GestureInterpreter()  # single source of thresholds/durations (legato_max_st, velocity -> slide speed...)
        self._held = []  # keys currently held live (in press order)
        self._onsets = []  # recent onset times (strum window)
        self._sep_model = None  # lazily loaded Demucs cache
        self._in_dev = None  # selected recording input device (None=system default)
        self._out_dev = None  # selected output device (None=system default)

    def _classify(self, midi):
        """Real-time decision for live playing; rules match GestureInterpreter.process and parameters come straight from it:
        ≥3 notes within an 80 ms window -> strum; with a held note, |interval| ≤ legato_max_st -> legato, larger -> slide;
        otherwise pluck."""
        now = time.perf_counter()
        self._onsets = [t for t in self._onsets if now - t < self._gi.strum_window]
        self._onsets.append(now)
        if len(self._onsets) >= self._gi.strum_min:
            return "strum", -1
        if getattr(self, "_voice_mode", "solo") == "poly":
            return "pluck", -1  # poly: overlap means a chord, no legato/slide
        frm = self._held[-1] if self._held else None
        if frm is not None and frm != midi:
            if abs(midi - frm) <= self._gi.legato_max_st:
                return "legato", frm
            return "slide", frm
        return "pluck", -1

    @staticmethod
    def _safe_name(name, default="tone"):
        """User-given asset name -> filename-safe form (one rule used by timbres, behaviour libraries and bakes, so the same name really is the same name)."""
        import re

        return re.sub(r"[^\w\- ]", "", str(name or "")).strip()[:40] or default

    @staticmethod
    def _auto_noise_level(Hs):
        """Default capture noise floor set by spectral centroid: near-sinusoidal sources (rubbed glass/whistling, centroid ≤1.2
        harmonics) have all their texture in the noise layer -> 30%; sources with harmonic structure (voice/guitar, centroid ≥1.8) -> 2%; linear in between."""
        import numpy as np

        hs = np.abs(np.asarray(Hs, np.float64))
        cen = float((np.arange(1, len(hs) + 1) * hs).sum() / (hs.sum() + 1e-12))
        u = min(1.0, max(0.0, (1.8 - cen) / 0.6))
        return round(0.02 + 0.28 * u, 3)

    def _balance(self, inst):
        """Normalise the steady harmonic loudness of the captured instrument to the reference (evens out capture volumes; the attack grain scales with it)."""
        import numpy as np

        Hs = np.asarray(inst.H_steady, np.float64)
        norm = float(np.sqrt((Hs**2).sum()))
        if norm > 1e-6:
            g = self._ref_energy / norm
            inst.H_steady = (Hs * g).astype(np.float32)
            ag = np.asarray(getattr(inst, "attack_grain", []), np.float64)
            if ag.size:
                inst.attack_grain = (ag * g).astype(np.float32)
        return inst

    # ---- Lifecycle ----
    def start(self):
        """Open the real-time audio stream (sounddevice OutputStream, its own callback thread) + try to connect external MIDI."""
        if not self._running:
            self.engine.start(queue=self._q)
            self._running = True
            self._open_midi()
        return self._running

    def stop(self):
        self._close_cap_stream()
        prev = getattr(self, "_stems_dir", None)
        if prev:
            import shutil

            shutil.rmtree(prev, ignore_errors=True)
            self._stems_dir = None
        if self._midi:
            try:
                self._midi.close()
            except Exception:
                pass
            self._midi = None
        if self._running:
            try:
                self.engine.stop()
            finally:
                self._running = False

    # ---- Performance (called from JS / MIDI) ----
    def note_on(self, midi, vel=100, art="auto", prev=-1):
        """art="auto" (live playing) -> unified decision on the engine side (gesture.py semantics); explicit art (piano-roll playback) -> used as given.
        Returns the decision string, which the UI uses to light the articulation label."""
        midi, vel, prev = int(midi), int(vel), int(prev)
        if art == "auto":
            art, frm = self._classify(midi)
            if prev < 0:
                prev = frm  # slide/legato start = the currently held note
            if midi in self._held:
                self._held.remove(midi)
            self._held.append(midi)
        if getattr(self, "_live_behavior", None) and getattr(self, "_live_ready", False):
            return self._note_on_live(midi, vel, art, prev)
        if art == "slide" and prev >= 0:
            self._q.put(Event("slide", midi, vel, self._gi._slide_dur(vel), prev))  # velocity -> slide speed
        elif art == "legato" and prev >= 0:
            self._q.put(Event("legato", midi, vel, self._gi.legato_dur, prev))
        elif art == "strum" and len(self._onsets) >= 2:
            self._later(self._strum_delay(), Event("on", midi, vel, 0.0, -1))
        else:
            self._q.put(Event("on", midi, vel, 0.0, -1))  # pluck = normal attack
        return art

    def _note_on_live(self, midi, vel, art, prev):
        """Live attack in behaviour mode: retrieve a trajectory -> ctl / ctl_legato event; falls back to the rule path if retrieval fails."""
        try:
            ctl = self._live_ctl(midi, vel, art, prev)
        except Exception as e:  # the real-time path must not go silent because of a retrieval error
            print(f"[bridge] live behavior fallback: {str(e)[:80]}")
            self._q.put(Event("on", midi, vel, 0.0, -1))
            return art
        self._live_on_t[midi] = time.perf_counter()
        if art in ("slide", "legato") and prev >= 0:
            ev = Event("ctl_legato", midi, vel, ctl, prev)  # reuse the previous note's voice, no re-attack
        else:
            ev = Event("ctl", midi, vel, ctl, -1)
        if art == "strum" and len(self._onsets) >= 2:
            self._later(self._strum_delay(), ev)
        else:
            self._q.put(ev)
        return art

    def _strum_delay(self):
        """Live strum stagger: note k of the cluster lands k×spread after the first note (STRUM knob); 0 if that time has already passed."""
        k = len(self._onsets) - 1
        return max(0.0, self._onsets[0] + k * self._gi.strum_spread - time.perf_counter())

    def note_off(self, midi):
        m = int(midi)
        if m in self._held:
            self._held.remove(m)
        self._q.put(Event("off", m, 0, 0.0, -1))
        t_on = getattr(self, "_live_on_t", {}).pop(m, None)
        if t_on is not None:  # behaviour mode: update the duration prior (EMA) with the actual key duration, so the next note's retrieval is more accurate
            d = time.perf_counter() - t_on
            if 0.05 < d < 4.0:
                self._live_dur = 0.6 * float(self._live_dur) + 0.4 * d
        return True

    # ---- Piano-roll playback: sample-accurate engine scheduling, same clock as the click ----
    def play_notes(self, notes, bpm=120.0, metro=False, bpb=4):
        """Write the whole phrase into engine._sched (absolute sample timestamps, consumed per block by the real-time callback).
        notes: [{midi,start(seconds),len,vel,art}]; with metro=True the click is aligned to the same reference sample as the first note."""
        eng = self.engine
        sr = eng.sr
        base = int(getattr(eng, "_stream_samples", 0)) + int(0.05 * sr)  # 50ms start-up margin
        plan, midis = [], set()
        for t, kind, m, v, value, note2 in self._gi.notes_to_events(notes):
            plan.append((base + int(t * sr), Event(kind, m, v, value, note2)))
            midis.add(m)
        plan.sort(key=lambda x: x[0])
        self._play_midis = midis
        eng._sched_i = 0  # reset to zero before swapping the plan: expired events of the old plan are skipped
        eng._sched = plan
        if metro:
            eng.set_click(float(bpm), accent=int(bpb), start=base)
        return True

    def stop_playback(self):
        """Cancel unsent events + release all scheduled notes + turn off the click."""
        eng = self.engine
        eng._sched = []
        eng.set_click(None)
        for m in getattr(self, "_play_midis", set()):
            self._q.put(Event("off", int(m), 0, 0.0, -1))
        import sounddevice as sd

        try:
            sd.stop()  # also stop neural render playback
        except Exception:
            pass
        return True

    def vibrato(self, depth01):
        # also sent in behaviour mode: the Voice does not add the LFO before the library trajectory ends, after that it takes over vibrato on long notes;
        # also record the latest depth so behaviour mode can decide the next note's vibrato label (CC1 and the knob have equal weight)
        self._live_vib = float(depth01)
        self._q.put(Event("vibrato", -1, 0, float(depth01), -1))
        return True

    # ---- Live playing with a behaviour library ----
    def set_live_behavior(self, name=None, wait=False):
        """Live playing driven by a behaviour library instead: key press -> retrieve a trajectory by (pitch, previous note, label) -> control stream drives the
        Voice; since the duration is unknown, stretch by the duration of the last few notes (initially the library median), hold the last frame, release on key up.
        Empty name = off, back to the rule path (GestureInterpreter).
        Loading + warm-up (~3s the first time: import, checkpoint, JIT) runs on a background thread, key presses use the rule path until ready,
        so the UI does not block; poll `live_behavior_status()`, or wait=True blocks until done (for tests)."""
        import threading

        name = str(name or "").strip()
        self._live_ready = False
        self._live_error = None
        if not name:
            self._live_behavior = None
            return {"ok": True, "name": None, "ready": True}
        self._live_behavior = name

        def _warm():
            import numpy as np

            try:
                nn, env = self._neural().resolve_behavior(name)  # same resolver as NeuralTake.render
                if hasattr(nn, "clone"):
                    nn = nn.clone()  # does not share last_idx with offline rendering (MIDI thread vs UI thread)
                feats = getattr(nn, "features", None)
                med = float(np.median(feats[:, 1])) if feats is not None and len(feats) else 0.5
                self._live_src = (nn, env)
                self._live_dur = min(2.0, max(0.25, med))  # duration prior: library median, then an EMA of playing
                self._live_on_t = {}
                self._live_ctl(60, 100, "pluck", -1)  # warm-up: imports and JIT for the first retrieval/filtering
                self._live_ctl(62, 100, "legato", 60)
                with self._state_lock:
                    if self._live_behavior == name:  # not switched away in the meantime
                        self._live_ready = True
            except Exception as e:
                with self._state_lock:
                    self._live_error = f"behavior {name!r} unusable: {str(e)[:60]}"
                    if self._live_behavior == name:
                        self._live_behavior = None

        t = threading.Thread(target=_warm, daemon=True, name="live-behavior-warmup")
        t.start()
        if wait:
            t.join()
            if self._live_error:
                return {"ok": False, "msg": self._live_error}
            return {"ok": True, "name": name, "ready": True, "dur_prior": round(self._live_dur, 3)}
        return {"ok": True, "name": name, "ready": False}

    def live_behavior_status(self):
        return {
            "name": getattr(self, "_live_behavior", None),
            "ready": bool(getattr(self, "_live_ready", False)),
            "error": getattr(self, "_live_error", None),
        }

    def _live_art_label(self, art, prev):
        """Label for a live note: slide comes from the gesture decision; vibrato if the latest vibrato depth (knob or CC1) is >0.15;
        otherwise plain."""
        if art == "slide" and prev >= 0:
            return 2
        depth = getattr(self, "_live_vib", None)
        if depth is None:
            depth = float(self.params.get("vibdepth", 0.0))
        return 1 if depth > 0.15 else 0

    def _live_ctl(self, midi, vel, art, prev):
        """One live note -> (f0_hz[T], amp[T], fps). With a previous note (legato/slide), retrieve a two-note phrase
        and take the second note's frames, so the retrieval context is a "note change within a phrase" rather than a "phrase start"; otherwise a single note (phrase-start attack)."""
        import numpy as np

        from neural.perform import FPS, control_from_behavior

        nn, env = self._live_src
        # retrieval duration: EMA of the last few notes, but not below 0.8 s. Short notes lose nothing when cut by key release, and long notes get a long enough
        # library trajectory (vibrato/fluctuation) to walk; after it ends the Voice hands vibrato to the engine LFO.
        dur = max(0.8, float(self._live_dur))
        art_i = self._live_art_label(art, prev)
        score = []
        if prev >= 0 and art in ("slide", "legato"):
            score.append({"midi": int(prev), "dur": 0.2, "art": 0})
        score.append({"midi": int(midi), "dur": dur + 0.12, "art": art_i})  # an extra 0.12s, then the phrase-end bow release is cut off
        f0, loud = control_from_behavior(
            score, nn, pitch_glide=False, envelope=env, traj_gain=float(getattr(self, "_live_gain", 1.0))
        )[:2]
        f0 = f0.numpy()[0, :, 0].astype(np.float64)
        loud = loud.numpy()[0, :, 0].astype(np.float64)
        if len(score) == 2:
            start = int(0.2 * FPS) - 5  # start of the second note in the control stream (minus 20ms crossfade overlap)
            f0, loud = f0[start:], loud[start:]
        keep = max(8, int(dur * FPS))
        f0, loud = f0[:keep], loud[:keep]
        # bring pitch back to nominal over the last 100ms: holding the last frame must not freeze at some vibrato phase (up to ±30 cents out of tune)
        f0_nom = 440.0 * 2.0 ** ((int(midi) - 69) / 12.0)
        r = min(len(f0), int(0.1 * FPS))
        if r > 1:
            w = np.linspace(0.0, 1.0, r)
            f0[-r:] = np.exp(np.log(f0[-r:]) * (1.0 - w) + np.log(f0_nom) * w)
        from runtime.engine import loud_to_amp

        return np.ascontiguousarray(f0), loud_to_amp(loud), float(FPS)

    # ---- Knobs (called from JS) ----
    def set_param(self, name, value):
        """Knobs change the sound in real time; every knob acts on the engine:
        master/noise/vibrate/vibdepth/brightness/artint change scalars that take effect immediately;
        strum/gsens change the gesture interpreter's durations; pick needs to reshape the attack grain -> debounced instrument rebuild."""
        from synth.voice import set_articulation_intensity, set_tone_tilt

        v = float(value)
        self.params[name] = v
        eng = self.engine
        if name == "master":
            eng.master_gain = max(0.0, v * 1.4)  # output multiplier (see process_block)
        elif name == "noise":
            if eng.instrument is not None:  # noise-side ratio: changed directly, the Voice reads it per block
                eng.instrument.noise_level = v
        elif name == "vibrate":
            eng.vib_rate_hz = 2.0 + v * 8.0  # 2–10 Hz
        elif name == "vibdepth":
            eng.vib_max_cents = v * 80.0
            self._live_vib = v
            # make held notes start vibrato immediately; in behaviour mode the LFO takes over after the trajectory ends
            self._q.put(Event("vibrato", -1, 0, v, -1))
        elif name == "brightness":
            set_tone_tilt((v - 0.5) * 12.0)  # ±6 dB/oct, matches the UI label
        elif name == "artint":
            set_articulation_intensity(v * 2.0)  # 0.5 is the default intensity, 1.0 doubles friction/dip
        elif name == "strum":
            self._gi.strum_spread = (8.0 + v * 70.0) / 1000.0  # string-order offset 8–78 ms (same formula as the UI)
        elif name == "gsens":
            glide = (20.0 + v * 80.0) / 1000.0  # legato glide 20–100 ms (same formula as the UI)
            self._gi.legato_dur = glide
            self._gi.slide_dur = glide * 2.2  # keep the default 130/60 ratio
            self._gi.slide_dur_min = glide * 0.9
        elif name == "pick":
            self._rebuild_instrument_later()
        return True

    def _rebuild_instrument(self):
        """Rebuild the instrument from the current snapshot + pick/noise knobs and hot swap (the attack grain shape lives in the build stage)."""
        from snapshot import load_snapshot
        from synth.instrument import build_instrument

        snap = self._last_snap or load_snapshot(str(self._default_path))
        tex = bool((snap.get("meta", {}) or {}).get("texture"))
        inst = build_instrument(
            snap, noise_level=0.6 if tex else 0.02, pick=float(self.params.get("pick", 0.5))
        )
        self._balance(inst)
        if not tex and "noise" in self.params:
            inst.noise_level = float(self.params["noise"])
        with self._state_lock:
            self.engine.set_instrument(inst)

    def _rebuild_instrument_later(self, delay_s=0.06):
        """Dragging a knob sends dozens of events per second; rebuild only once, 60ms after the hand stops (latest value wins)."""
        import threading

        t = getattr(self, "_rebuild_timer", None)
        if t is not None:
            t.cancel()
        t = threading.Timer(delay_s, self._rebuild_instrument)
        t.daemon = True
        self._rebuild_timer = t
        t.start()

    def _later(self, delay_s, ev):
        """Post an engine event with a delay (for strum staggering; Timer jitter ~1ms, far smaller than string-order offsets)."""
        import threading

        if delay_s <= 0.0:
            self._q.put(ev)
            return
        t = threading.Timer(delay_s, self._q.put, args=(ev,))
        t.daemon = True
        t.start()

