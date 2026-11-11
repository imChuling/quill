"""Real-time audio engine: block-based voice pool and event scheduling.

`process_block` is the single code path shared by the real-time callback and offline rendering: `render_offline` opens no sound card,
feeds an event list through the same function, and is the basis for unit tests, regression and listening-stimulus generation.

Real-time path constraints:
  in performance mode gc.freeze() + gc.disable(), restored on stop;
  events carry sample timestamps and fire at sample offsets within a block, so strum's 5–15 ms offsets survive block boundaries;
  loudness/f0 linear ramps within a block and envelope-tail clearing live in voice;
  bus nan_to_num + tanh soft limiting;
  no allocation, locks or IO in the callback; the UI and audio threads communicate only through queues and atomic reference swaps;
  instrument hot swap = atomic rebind of the instrument reference;
  outermost try/except in the callback, on error mute and set an error flag; numba warm-up prevents a first-note xrun.

This module depends only on numpy and numba; mido is only used in midi_in.py.
"""

from __future__ import annotations

import gc
import os
from collections import namedtuple

import numpy as np

from synth.oscillator import warmup as _osc_warmup
from synth.voice import (
    _REL,
    Voice,
    regrain_legato,
    set_humanize,
    set_liveness,
    set_nyquist_ramp_hz,
    set_vibrato_shape,
)

try:
    from quill_config import CFG
except Exception:  # pragma: no cover
    CFG = None

# Events: kind in {"on","off","bend","cc","slide","legato","vibrato","ctl","retrig"}
#   slide/legato: note=target note, note2=start note (continued from its voice), value=duration in seconds
#   vibrato: value=depth 0-1
#   ctl: attack and enter control-driven mode, value=(f0_hz[T], amp[T], fps); retrig: that voice replays its attack grain
Event = namedtuple("Event", ["kind", "note", "vel", "value", "note2"], defaults=[-1])


LOUD_GAIN = 0.4  # dynamic range of z-score loudness -> linear amplitude (±2σ ≈ ±7 dB)


def loud_to_amp(loud, gain: float = LOUD_GAIN):
    """z-score log-RMS loudness (the neural/perform.py convention) -> linear amplitude, peak-normalised to 1.

    Shared by offline render_control and the real-time behaviour mode; the dynamic range is defined only here.
    """
    loud = np.asarray(loud, np.float64)
    return np.exp(float(gain) * (loud - loud.max()))


def _runtime_cfg(cfg):
    cfg = cfg or {}
    rt = cfg.get("runtime", {})
    return {
        "blocksize": int(rt.get("blocksize", 256)),
        "max_voices": int(rt.get("max_voices", 16)),
        "channels": int(rt.get("channels", 1)),
        "noise_level": float(rt.get("noise_level", 0.5)),
        "attack_fade_ms": float(rt.get("attack_fade_ms", 20.0)),
        "steal_fade_ms": float(rt.get("steal_fade_ms", 4.0)),
        "nyquist_ramp_hz": float(rt.get("nyquist_ramp_hz", 600.0)),
    }


class Engine:
    def __init__(self, instrument, cfg=CFG, eval_mode: bool = False):
        self.instrument = instrument  # atomically rebound on hot swap
        self.sr = int(instrument.sr)
        self.eval_mode = bool(eval_mode)
        c = cfg if cfg is not None else {}
        self.rt = _runtime_cfg(c)
        self.adsr = dict(
            c.get(
                "adsr", {"attack_ms": 8.0, "decay_ms": 120.0, "sustain": 0.7, "release_ms": 180.0}
            )
        )
        self.vel_cfg = dict(c.get("velocity", {"loud_exp": 1.5, "tilt_db_oct": 6.0}))

        self.blocksize = self.rt["blocksize"]
        self.max_voices = self.rt["max_voices"]
        K = instrument.K
        self.pool = [Voice(K, self.sr) for _ in range(self.max_voices)]
        self._note2voice = {}
        self._age = 0
        # channel-level pitch-bend state (semitones): a new note_on must inherit it, otherwise while the wheel is pushed every new note
        # first chirps from the original pitch to the bent pitch
        self._bend_st = 0.0
        # channel-level vibrato depth (0–1): inherited by control-driven voices at attack, the LFO takes over after the library trajectory ends
        self._vib_st = 0.0
        self.error_flag = False

        set_nyquist_ramp_hz(self.rt["nyquist_ramp_hz"])
        _syn = c.get("synth", {}) if isinstance(c, dict) else {}
        set_humanize(_syn.get("humanize_cents", 6.0), _syn.get("humanize_amp", 0.05))
        set_liveness(
            _syn.get("liveness_spec", 0.85),
            _syn.get("liveness_dark", 0.0),
            _syn.get("liveness_breath", 0.8),
        )
        _ges = c.get("gesture", {}) if isinstance(c, dict) else {}
        set_vibrato_shape(_ges.get("vibrato_onset_ms", 110.0), _ges.get("vibrato_ramp_ms", 220.0))

        # preallocation: no allocation in the callback
        self._buf = np.zeros(self.blocksize, dtype=np.float64)

        # gesture templates: slide curve and vibrato rate, can be injected from .tplib
        self.slide_curve = None  # g(τ)∈[0,1], None means linear
        # CC1 vibrato rate prefers the source clip measurement (snapshot.vibrato.rate_hz), otherwise the default
        self.vib_rate_hz = float(getattr(instrument, "vib_rate", None) or 5.5)
        self.vib_max_cents = 50.0  # maximum vibrato depth at CC1=1
        # CC64 sustain pedal
        self._pedal_down = False
        self._pedal_sustained = set()

        # event scheduling
        self._sched = []  # offline: [(t_samp, Event)], ascending
        self._sched_i = 0
        self._queue = None  # real time: SimpleQueue fed by midi_in
        self._stream = None
        self._stream_samples = 0
        self._rng = np.random.default_rng(0)

        _osc_warmup()  # JIT warm-up, prevents a first-note xrun

    def set_instrument(self, inst):
        """Hot-swap the instrument: CPython reference assignment is atomic, so the callback thread sees either the old or the new one."""
        self.instrument = inst

    def _alloc_voice(self) -> Voice:
        """Prefer an idle voice; otherwise steal the oldest releasing voice; otherwise steal the oldest."""
        for v in self.pool:
            if not v.active:
                return v
        released = [v for v in self.pool if v.stage == _REL]
        cand = released if released else self.pool
        return min(cand, key=lambda v: v.age)

    def _noise_pos0(self):
        if self.eval_mode:
            return 0
        tlen = len(self.instrument.noise_table)
        return int(self._rng.integers(0, tlen)) if tlen > 0 else 0

    def _apply_event(self, ev: Event):
        if ev.kind in ("on", "ctl"):
            v = self._alloc_voice()
            # if the stolen voice is still sounding, unbind its note mapping first
            if v.active and v.note in self._note2voice and self._note2voice[v.note] is v:
                self._note2voice.pop(v.note, None)
            self._age += 1
            v.note_on(
                ev.note,
                ev.vel,
                self.instrument,
                self.adsr,
                self.vel_cfg,
                self.rt,
                self._age,
                self._noise_pos0(),
            )
            v.init_bend(self._bend_st)  # inherit the channel pitch bend, applied instantly
            if ev.kind == "ctl":
                v.drive(*ev.value)
                if self._vib_st > 0.0:  # the LFO takes over vibrato after the library trajectory ends; before that the Voice gate prevents stacking
                    v.set_vibrato(self._vib_st * self.vib_max_cents, self.vib_rate_hz)
            self._note2voice[ev.note] = v
        elif ev.kind == "retrig":
            v = self._note2voice.get(ev.note)
            if v is not None and v.active:
                v.retrigger_attack()
        elif ev.kind == "ctl_legato":
            # legato switch to a new control stream: reuse note2's voice (phase/envelope not reset), retrigger the grain at low level
            v = self._note2voice.get(ev.note2)
            if v is None or not v.active:
                self._apply_event(Event("ctl", ev.note, ev.vel, ev.value, -1))
                return
            v.drive(*ev.value)
            v.retrigger_attack(regrain_legato())  # same multiplier as the legato grain on the rule path
            v.note = int(ev.note)
            self._note2voice.pop(ev.note2, None)
            self._note2voice[ev.note] = v
        elif ev.kind == "off":
            v = self._note2voice.pop(ev.note, None)
            if v is not None and v.note == ev.note:
                if self._pedal_down:  # pedal down: sustain, defer release
                    self._pedal_sustained.add(v)
                else:
                    v.note_off()
        elif ev.kind == "pedal":  # CC64 sustain pedal
            down = float(ev.value) >= 0.5
            if self._pedal_down and not down:  # pedal up: release all voices held by the pedal
                for v in self._pedal_sustained:
                    if v.active:
                        v.note_off()
                self._pedal_sustained.clear()
            self._pedal_down = down
        elif ev.kind == "bend":
            self._bend_st = float(ev.value)  # channel-level state, inherited by new notes
            tbd = getattr(self, "_trace_bend", None)
            if tbd is not None:  # diagnostics: record bend events (sample time, semitones)
                tbd.append((self._stream_samples, float(ev.value)))
            for v in self.pool:
                if v.active:
                    v.set_bend(ev.value)
        elif ev.kind in ("slide", "legato"):
            self._glide_event(ev)
        elif ev.kind == "vibrato":  # CC1 -> vibrato depth
            self._vib_st = max(0.0, float(ev.value))
            depth = self._vib_st * self.vib_max_cents
            for v in self.pool:
                if v.active:
                    v.set_vibrato(depth, self.vib_rate_hz)

    def _glide_event(self, ev):
        """slide/legato: continue from note2's voice and slide to note, without a new attack."""
        v = self._note2voice.get(ev.note2)
        if v is None or not v.active:
            # no start note to continue from, fall back to a normal attack
            self._apply_event(Event("on", ev.note, max(1, int(ev.vel or 90)), 0.0))
            return
        f0_tgt = 440.0 * 2.0 ** ((ev.note - 69) / 12.0)
        dur_s = float(ev.value) if ev.value else (0.04 if ev.kind == "legato" else 0.12)
        curve = self.slide_curve if ev.kind == "slide" else None
        v.glide_to(
            f0_tgt, int(dur_s * self.sr), curve=curve, kind=ev.kind, vel=int(ev.vel or 90)
        )
        v.note = int(ev.note)
        self._note2voice.pop(ev.note2, None)
        self._note2voice[ev.note] = v

    def _render_segment(self, buf, a, b):
        if b <= a:
            return
        seg = buf[a:b]
        nlen = b - a
        for v in self.pool:
            if v.active:
                v.render_into(seg, nlen)

    def _events_for_block(self, block_start, n):
        """Return [(offset_in_block, Event)], sorted by offset ascending."""
        out = []
        if self._queue is not None:  # real time: events arriving in this block fire immediately
            while True:
                try:
                    ev = self._queue.get_nowait()
                except Exception:
                    break
                out.append((0, ev))
        # offline: take those falling in [start, start+n) from the ordered schedule
        while self._sched_i < len(self._sched):
            t, ev = self._sched[self._sched_i]
            if t < block_start:
                self._sched_i += 1
                continue
            if t >= block_start + n:
                break
            out.append((t - block_start, ev))
            self._sched_i += 1
        out.sort(key=lambda x: x[0])
        return out

    def process_block(self, out, n, block_start):
        """Core: render n samples into out[:n] (float32). Shared by real time and offline."""
        buf = self._buf
        try:
            buf[:n] = 0.0
            events = self._events_for_block(block_start, n)
            seg = 0
            for offset, ev in events:
                offset = 0 if offset < 0 else (n if offset > n else offset)
                if offset > seg:
                    self._render_segment(buf, seg, offset)
                self._apply_event(ev)  # events take effect at their in-block offset, sample-accurate
                seg = offset
            if seg < n:
                self._render_segment(buf, seg, n)
            np.nan_to_num(buf[:n], copy=False)
            buf[:n] *= getattr(self, "master_gain", 1.0)  # master volume knob, default 1.0
            ck = getattr(self, "_click", None)  # metronome, same sample clock as the notes
            if ck is not None:
                self._render_click(buf, n, block_start, ck)
            np.tanh(buf[:n], out=buf[:n])  # bus soft limiting
            out[:n] = buf[:n]
        except Exception:  # exceptions do not kill the stream: mute and set the error flag
            out[:n] = 0.0
            self.error_flag = True

    # ---- Real time (needs a sound card and sounddevice) ----
    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            self.error_flag = True
        self.process_block(outdata[:, 0], frames, self._stream_samples)
        if outdata.shape[1] > 1:
            outdata[:, 1:] = outdata[:, :1]
        tb = getattr(self, "_trace_buf", None)
        if tb is not None:  # diagnostics: capture the audio actually sent to the sound card; breaks zero allocation, debug only
            tb.append(outdata[:, 0].copy())
        self._stream_samples += frames

    def start(self, queue=None, device=None):
        """Open the output stream and enter performance mode: gc.freeze() + gc.disable(), restored on stop."""
        import sounddevice as sd

        self._queue = queue
        self._stream_samples = 0
        self.error_flag = False
        if os.environ.get("QUILL_BEND_TRACE"):  # diagnostic mode: capture device audio and bend events
            self._trace_buf = []
            self._trace_bend = []
            print("[BEND TRACE] armed — writes assets/user_tones/_bend_trace.* on window close (or Ctrl+C)")
            if not getattr(self, "_trace_atexit", False):  # also saves on abnormal shutdown
                import atexit

                atexit.register(self._dump_trace)
                self._trace_atexit = True
        else:
            self._trace_buf = None
            self._trace_bend = None
        gc.collect()
        try:
            gc.freeze()
        except Exception:
            pass
        gc.disable()
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.blocksize,
            channels=self.rt["channels"],
            dtype="float32",
            callback=self._audio_callback,
            device=device,
        )
        self._stream.start()
        return self._stream

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._dump_trace()
        gc.enable()

    def _dump_trace(self):
        tb = getattr(self, "_trace_buf", None)
        if not tb:
            return
        try:
            import json

            y = np.concatenate(tb).astype(np.float32)
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "assets", "user_tones"
            )
            os.makedirs(out_dir, exist_ok=True)
            wav_path = os.path.join(out_dir, "_bend_trace.wav")
            try:
                import soundfile as sf

                sf.write(wav_path, y, self.sr)
            except Exception:  # without soundfile fall back to the standard library, 16-bit PCM
                import wave

                pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2")
                with wave.open(wav_path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self.sr)
                    w.writeframes(pcm.tobytes())
            with open(os.path.join(out_dir, "_bend_trace.json"), "w") as f:
                json.dump({"sr": self.sr, "blocksize": self.blocksize, "bend": self._trace_bend}, f)
            print(
                f"[BEND TRACE] {len(y) / self.sr:.1f}s audio + "
                f"{len(self._trace_bend)} bend events -> assets/user_tones/_bend_trace.*"
            )
        except Exception as e:
            print(f"[BEND TRACE] dump failed: {e}")
        finally:
            self._trace_buf = None
            self._trace_bend = None

    def stream_latency_ms(self):
        if self._stream is not None:
            return float(self._stream.latency) * 1000.0
        return float("nan")

    # ---- Offline rendering (no sound card, shares process_block with real time) ----
    def reset(self):
        for v in self.pool:
            v.active = False
            v._reset_runtime()
        self._note2voice.clear()
        self._age = 0
        self._sched = []
        self._sched_i = 0
        self._stream_samples = 0
        self.error_flag = False
        self._pedal_down = False
        self._pedal_sustained = set()
        self._bend_st = 0.0
        self._vib_st = 0.0
        self._rng = np.random.default_rng(0)

    def set_gesture_templates(self, slide_curve=None, vib_rate_hz=None):
        """Inject gesture templates from .tplib: slide curve and vibrato rate."""
        if slide_curve is not None:
            self.slide_curve = np.asarray(slide_curve, dtype=np.float64)
        if vib_rate_hz:
            self.vib_rate_hz = float(vib_rate_hz)

    # ---- Metronome: sample-accurate click on the same clock as the notes ----
    def set_click(self, bpm=None, accent=4, gain=0.45, start=None):
        """Turn the metronome on/off (bpm=None turns it off).

        The tick is a decaying sine of about 5 ms, pregenerated here so the callback allocates nothing; beat 1 of every accent
        group is accented. start is the absolute start sample, None means now.
        """
        if not bpm:
            self._click = None
            return
        period = int(round(self.sr * 60.0 / float(bpm)))
        t = np.arange(int(self.sr * 0.005), dtype=np.float64) / self.sr
        burst = np.exp(-t / 0.0015)
        hi = (np.sin(2 * np.pi * 2093.0 * t) * burst * gain).astype(np.float32)
        lo = (np.sin(2 * np.pi * 1568.0 * t) * burst * gain * 0.7).astype(np.float32)
        now = int(getattr(self, "_stream_samples", 0))
        nxt = max(int(start), now) if start is not None else now
        self._click = {
            "period": period,
            "accent": max(1, int(accent)),
            "next": nxt,
            "beat": 0,
            "hi": hi,
            "lo": lo,
        }

    def _render_click(self, buf, n, block_start, ck):
        """Add the ticks that fall in this block to buf. A tick is only 5 ms, so truncation at the block end is negligible."""
        end = block_start + n
        while ck["next"] < end:
            off = ck["next"] - block_start
            wav = ck["hi"] if ck["beat"] % ck["accent"] == 0 else ck["lo"]
            if off >= 0:
                L = min(len(wav), n - off)
                if L > 0:
                    buf[off : off + L] += wav[:L]
            ck["next"] += ck["period"]
            ck["beat"] += 1

    def render_offline(self, events_sec, dur_s: float) -> np.ndarray:
        """Feed a (t_sec, kind, note, vel, value[, note2]) event list, render offline -> float32.

        Runs the same process_block as the real-time callback; with eval_mode=True the noise pointer is reset and output is reproducible.
        For slide/legato events the 6th item note2 is the start note.
        """
        self.reset()
        self._queue = None
        sched = []
        for e in events_sec:
            t_sec, kind = e[0], e[1]
            note = e[2] if len(e) > 2 else -1
            vel = e[3] if len(e) > 3 else 0
            value = e[4] if len(e) > 4 else 0.0
            note2 = e[5] if len(e) > 5 else -1
            sched.append((int(round(t_sec * self.sr)), Event(kind, note, vel, value, note2)))
        sched.sort(key=lambda x: x[0])
        self._sched = sched

        total = int(round(dur_s * self.sr))
        bs = self.blocksize
        nblocks = (total + bs - 1) // bs
        out = np.zeros(nblocks * bs, dtype=np.float32)
        for bi in range(nblocks):
            start = bi * bs
            self.process_block(out[start : start + bs], bs, start)
        return out[:total]


    def render_control(
        self,
        f0_hz,
        loud,
        fps: float = 250.0,
        onsets_s=(),
        vel: int = 100,
        loud_gain: float = LOUD_GAIN,
        tail_s: float = 0.3,
        bright=None,
        noise=None,
        motion: float = 1.0,
    ) -> np.ndarray:
        """(f0, loudness) control stream -> audio; same interface as the DDSP decoder, so the behaviour library can drive the captured timbre directly.

        loud is the articulation layer's z-score log-RMS (see neural/perform.py), restored to linear amplitude as amp = exp(loud_gain·loud)
        and normalised to peak 1; loud_gain sets the dynamic range (±2σ ≈ ±7 dB at 0.4).
        The attack grain is replayed at onsets_s. bright/noise are the behaviour library's 3rd/4th dimensions (log2 centroid residual in oct,
        log10 flatness residual): centroid residual × 6 dB/oct goes into the spectral tilt, flatness residual into the noise-layer multiplier; motion
        scales both overall, 0 turns them off and uses only f0/loudness.
        """
        f0_hz = np.asarray(f0_hz, np.float64)
        amp = loud_to_amp(loud, loud_gain)
        note = int(round(69.0 + 12.0 * np.log2(max(float(f0_hz[0]), 1e-3) / 440.0)))
        T = len(f0_hz)
        ctl = (f0_hz, amp, float(fps))
        if motion > 0 and bright is not None and noise is not None:
            tilt = np.asarray(bright, np.float64) * 6.0 * float(motion)  # oct → dB/oct
            nz = 10.0 ** (np.asarray(noise, np.float64) * float(motion))  # log10 -> multiplier
            ctl = ctl + (tilt, nz)
        evs = [(0.0, "ctl", note, int(vel), ctl, -1)]
        evs += [(float(t), "retrig", note, 0, 0.0, -1) for t in onsets_s if float(t) > 1e-3]
        evs.append((T / float(fps), "off", note, 0, 0.0, -1))
        saved = self.adsr
        # the envelope is left to the control stream: full level at attack, no decay; 80 ms tail after note_off
        self.adsr = {"attack_ms": 3.0, "decay_ms": 1.0, "sustain": 1.0, "release_ms": 80.0}
        try:
            return self.render_offline(evs, T / float(fps) + float(tail_s))
        finally:
            self.adsr = saved


def engine_from_snapshot_file(path, eval_mode=True, noise_level=None, cfg=CFG):
    """Convenience: snapshot file -> Instrument -> Engine."""
    from snapshot import load_snapshot
    from synth.instrument import build_instrument

    snap = load_snapshot(path)
    nl = (
        (CFG or {}).get("runtime", {}).get("noise_level", 0.5)
        if noise_level is None
        else noise_level
    )
    inst = build_instrument(snap, noise_level=nl)
    return Engine(inst, cfg=cfg, eval_mode=eval_mode)
