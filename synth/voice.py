"""Single Voice: phase array, ADSR state machine, attack grain.

Each Voice holds its own persistent state (phase, ADSR, control ramps, grain/noise read pointers) and accumulates its contribution into
the main mix block per block. note-on resets phase; a stolen voice first fades out quickly over 3–5 ms before reuse; loudness and f0 ramp
linearly within a block; the state is cleared when the envelope tail drops below -120 dB. This module depends only on numpy and the numba kernel.
"""

from __future__ import annotations

import numpy as np

from synth.oscillator import voice_block_add

_SILENCE = 10 ** (-120 / 20.0)  # -120 dB: envelope-tail clearing threshold, prevents denormals

# ADSR stages
_ATK, _DEC, _SUS, _REL, _DONE = 0, 1, 2, 3, 4

# immutable constants (not changed via knobs/config)
_GRAIN_WITH_ATK = 0.30
_BEND_RATE_ST_S = 250.0


class LiveParams:
    """Mutable real-time parameters of the Voice layer, updated by the engine/config/knobs through this module's setters."""
    __slots__ = (
        "ramp_hz", "hum_cents", "hum_amp",
        "vib_onset_ms", "vib_ramp_ms",
        "live_spec", "live_dark", "live_breath",
        "slide_noise", "regrain_legato", "regrain_slide",
        "art_dip_legato", "art_dip_slide",
        "tone_tilt",
    )

    def __init__(self):
        self.ramp_hz = 600.0
        self.hum_cents = 6.0
        self.hum_amp = 0.05
        self.vib_onset_ms = 110.0
        self.vib_ramp_ms = 220.0
        self.live_spec = 0.85
        self.live_dark = 0.0
        self.live_breath = 0.8
        self.slide_noise = 0.30
        self.regrain_legato = 0.16
        self.regrain_slide = 0.10
        self.art_dip_legato = 0.30  # note-change dip depth (linear amplitude, 0.30 ≈ -3 dB)
        self.art_dip_slide = 0.22
        self.tone_tilt = 0.0


_P = LiveParams()

_SLIDE_FEEL_BASE = (0.30, 0.16, 0.10, 0.30, 0.22)


class Voice:
    def __init__(self, K: int, sr: int):
        self.K = int(K)
        self.sr = int(sr)
        self.phase = np.zeros(K, dtype=np.float64)
        # Schroeder phases (minimum crest factor): note-on uses them to disperse initial phases, avoiding the mechanical buzz of all harmonics
        # aligning at phase 0 into a pulse train; initial phases are deterministic, so renders are reproducible.
        ki = np.arange(K, dtype=np.float64)
        self._init_phase = (np.pi * ki * (ki + 1.0) / max(K, 1)) % (2.0 * np.pi)
        self.gain_k = np.zeros(K, dtype=np.float64)
        self._gain_buf = np.zeros(K, dtype=np.float64)  # write buffer for spectral trajectory/tilt, no allocation on the hot path
        self._gain_prev = np.zeros(K, dtype=np.float64)  # gain of the previous block, start point of the kernel's in-block interpolation
        self.ratio = np.arange(1, K + 1, dtype=np.float64)
        self.active = False
        self.note = -1
        self.age = 0  # note-on counter, used to steal the oldest voice
        self._reset_runtime()

    def _reset_runtime(self):
        self.stage = _DONE
        self.env = 0.0
        self.vel_loud = 0.0
        self._tilt = 0.0
        self.f0 = 220.0
        self.f0_target = 220.0
        self.f0_base = 220.0  # current sounding f0, glide moves on top of this
        self.bend = 1.0  # pitch-bend ratio, after smoothing
        self.bend_prev = 1.0
        self._bend_target = 1.0  # pitch-bend target, set by set_bend, followed with per-block smoothing
        # glide (slide/legato continuation: f0 slides along a curve in the log domain without resetting phase/ADSR)
        self._glide_from = 220.0
        self._glide_to = 220.0
        self._glide_total = 1
        self._glide_pos = 0
        self._glide_curve = None
        # vibrato LFO (CC1 depth, rate from library measurement or default)
        self._vib_depth = 0.0  # cents
        self._vib_rate = 5.5
        self._vib_phase = 0.0
        self._vib_age = 0  # samples elapsed since vibrato onset, used for delay and fade-in
        self._hum_off = 0.0  # humanisation LFO phase offset, staggered by pitch to avoid phase locking in unison
        self._gain_from = None  # gain_k interpolates along the glide, removing clicks at slide/legato start
        self._gain_to = None
        self._traj_pos = 0.0  # spectral trajectory read position (frames, float), walks the source recording's sustain segment
        self._traj_dir = 1.0  # ping-pong direction (±1), avoids a wrap seam
        self._traj_rate = 1.0  # small per-voice rate difference so chord notes breathe out of sync
        self._tilt_s = 0.0  # smoothing state of the dynamic tilt (150 ms first order), removes per-block jump glitches
        self._amp_prev = 1.0  # previous block's loudness-breathing factor: a uses it, b uses this block's, no steps within the kernel block
        self._grain_scale = 1.0  # attack grain multiplier: set by velocity in note_on, low level for legato/slide retriggers
        self._atk_traj_pos = 1 << 30  # attack spectral trajectory position (samples; large value = inactive)
        self._glide_stren = 0.0  # slide strength (by interval), drives the friction-noise swell
        self._art_pos = 1 << 30  # note-change dip envelope position (samples; large value = inactive)
        self._art_len = 1
        self._art_depth = 0.0
        self.samples_since_on = 0
        self.attack_pos = 0
        self.noise_pos = 0
        self._inst = None
        # per-sample increment of each linear ADSR stage, computed from sr at note_on
        self._atk_inc = self._dec_inc = self._rel_inc = 0.0
        self._sustain = 0.7
        self._fadein_len = 1
        # control-driven mode: an external (f0 Hz, linear amplitude) frame stream directly sets each block's f0/loudness, bypassing glide,
        # envelope and humanisation. None means normal event-driven mode.
        self._ctl_fps = 250.0
        self._ctl_last_amp = 1.0
        self._ctl_last_f0 = 220.0
        self._clear_ctl()

    def _clear_ctl(self):
        """Clear control-driven state; shared by note_on (a reused voice must not carry leftovers of the previous control stream) and runtime reset."""
        self._ctl_f0 = self._ctl_amp = self._ctl_tilt = self._ctl_nz = None
        self._ctl_done = False
        self._ctl_s0 = 0
        self._ctl_xf_from = None
        self._ctl_xf_f0 = None
        self._ctl_xf_len = 0

    # ---- Control-driven: the same (f0, loudness) frame-stream interface as DDSP ----
    def drive(self, f0_hz, amp, fps: float = 250.0, tilt_db=None, noise_mul=None):
        """Enter control-driven mode: render_into then takes f0/loudness from these two frame streams (linear interpolation, clamped to the last frame).

        The read position counts from samples_since_on, so this must be called right after note_on. tilt_db (dB/oct) and
        noise_mul (noise layer multiplier) are optional and carry the behaviour library's brightness/breath motion.
        """
        switching = self._ctl_f0 is not None and self.active  # legato trajectory switch, the voice is already sounding
        self._ctl_f0 = np.ascontiguousarray(f0_hz, dtype=np.float64)
        self._ctl_amp = np.ascontiguousarray(amp, dtype=np.float64)
        self._ctl_fps = float(fps)
        self._ctl_tilt = None if tilt_db is None else np.ascontiguousarray(tilt_db, dtype=np.float64)
        self._ctl_nz = None if noise_mul is None else np.ascontiguousarray(noise_mul, dtype=np.float64)
        self._ctl_done = False  # until the trajectory ends the library's vibrato applies; the engine LFO is not added on top
        # read position counts from this drive call; samples_since_on is not reset on a legato trajectory switch
        self._ctl_s0 = int(self.samples_since_on) if switching else 0
        if switching:
            # note change: loudness and f0 transition from current values to the new stream over 20 ms (loudness linear, f0 geometric). A hard
            # cut of several semitones within one block clicks; a 20 ms geometric transition is not heard as a slide but keeps the slope continuous.
            # The slow f0 that the formant mapping follows is not reset; the 120 ms smoothing walks it over. Resetting would jump harmonic gains.
            self._ctl_xf_from = float(self._ctl_last_amp)
            self._ctl_xf_f0 = float(self._ctl_last_f0)
            self._ctl_xf_len = max(1, int(0.02 * self.sr))
        else:
            self._ctl_f0_slow = float(self._ctl_f0[0])  # slow f0 followed by the formant mapping, does not follow vibrato
            self._ctl_xf_from = None
            self._ctl_xf_f0 = None
            self._ctl_xf_len = 0

    def _ctl_at(self, s: int):
        s_rel = s - self._ctl_s0
        p = s_rel * self._ctl_fps / self.sr
        last = len(self._ctl_f0) - 1
        if p >= last:
            if not self._ctl_done:  # trajectory finished: hold the last frame, the engine LFO vibrato takes over from here
                self._ctl_done = True
                self._vib_age = 0
            f0, amp = self._ctl_f0[last], self._ctl_amp[last]
        else:
            i = max(0, int(p))
            fr = max(0.0, p - i)
            f0 = self._ctl_f0[i] * (1.0 - fr) + self._ctl_f0[i + 1] * fr
            amp = self._ctl_amp[i] * (1.0 - fr) + self._ctl_amp[i + 1] * fr
        if self._ctl_xf_from is not None and s_rel < self._ctl_xf_len:
            w = max(0.0, s_rel / self._ctl_xf_len)
            amp = self._ctl_xf_from * (1.0 - w) + amp * w
            if self._ctl_xf_f0:
                f0 = self._ctl_xf_f0 ** (1.0 - w) * f0**w
        self._ctl_last_amp = amp
        self._ctl_last_f0 = f0
        return f0, amp

    def _ctl_aux(self, arr, s: int):
        p = (s - self._ctl_s0) * self._ctl_fps / self.sr
        last = len(arr) - 1
        if p >= last:
            return float(arr[last])
        i = int(p)
        fr = p - i
        return float(arr[i] * (1.0 - fr) + arr[i + 1] * fr)

    def retrigger_attack(self, scale: float = 0.6):
        """Note boundary in control-driven mode: only replays the attack grain/attack spectral trajectory, does not touch phase, fade-in or envelope
        (the loudness arc is handled by the behaviour library)."""
        self.attack_pos = 0
        self._grain_scale = float(scale)
        if getattr(self._inst, "atk_traj", None) is not None:
            self._grain_scale *= _GRAIN_WITH_ATK
            self._atk_traj_pos = 0

    def note_on(
        self,
        note: int,
        vel: int,
        inst,
        adsr: dict,
        vel_cfg: dict,
        runtime_cfg: dict,
        age: int,
        noise_pos0: int = 0,
    ):
        """Trigger: set target f0, velocity mapping, brightness tilt and gain_k; reset phase, grain and ADSR."""
        self._inst = inst
        self.note = int(note)
        self.age = int(age)
        self.active = True

        self.f0_target = 440.0 * 2.0 ** ((note - 69) / 12.0)
        self.f0 = self.f0_base = self.f0_target
        self.bend = self.bend_prev = self._bend_target = 1.0
        # reset glide / vibrato: note_on is a plucked attack, no slide or vibrato
        self._glide_from = self._glide_to = self.f0_target
        self._glide_total, self._glide_pos, self._glide_curve = 1, 0, None
        self._vib_depth, self._vib_phase, self._vib_age = 0.0, 0.0, 0
        self._hum_off = (int(note) % 12) * 0.523  # stagger humanisation phase by pitch
        st = inst.spec_traj  # spectral trajectory start/rate staggered deterministically by note number, reproducible
        if st is not None:
            F = st.shape[1]
            self._traj_pos = float((int(note) * 7.13) % max(F - 1, 1))
            self._traj_rate = 1.0 + 0.12 * np.sin(int(note) * 1.7)
            self._traj_dir = 1.0 if (int(note) % 2 == 0) else -1.0
        self._amp_prev = 1.0
        self._tilt_s = 0.0
        self._atk_traj_pos = 0  # attack spectral trajectory read position (samples), each partial's own attack
        # new notes start event-driven; control-driven mode needs an explicit drive() after note_on. Otherwise a reused voice would carry
        # leftovers of the previous control stream and treat a normal attack as control-driven, or treat a new control note as a legato switch.
        self._clear_ctl()

        v = max(0, min(127, int(vel))) / 127.0
        self.vel_loud = v ** float(vel_cfg["loud_exp"])
        self._tilt = -(1.0 - v) * float(vel_cfg["tilt_db_oct"])  # dB/oct, softer notes are darker
        self._grain_scale = 0.6 + 0.6 * v  # velocity -> attack grain share, harder notes have a more prominent transient
        if getattr(inst, "atk_traj", None) is not None:
            self._grain_scale *= _GRAIN_WITH_ATK  # when the harmonic layer produces the pluck itself, the pasted grain becomes a light accent
        self._glide_stren = 0.0
        self.ratio[:] = inst.ratio_k[: self.K]
        # gain_k = formant-preserving redirection × brightness tilt, set by the current played pitch
        self.gain_k[:] = self._compute_gain_k(self.f0_target)
        self._gain_prev[:] = self.gain_k  # no interpolation in the first note-on block, start equals end
        self._gain_to = None

        self.phase[:] = self._init_phase  # dispersed initial phases, removes the mechanical feel
        self.attack_pos = 0
        self.noise_pos = int(noise_pos0)
        self.samples_since_on = 0

        sr = self.sr
        self._sustain = float(adsr["sustain"])
        self._atk_inc = 1.0 / max(1, adsr["attack_ms"] / 1000.0 * sr)
        self._dec_inc = (1.0 - self._sustain) / max(1, adsr["decay_ms"] / 1000.0 * sr)
        self._rel_inc = 1.0 / max(1, adsr["release_ms"] / 1000.0 * sr)
        self._fadein_len = max(1, int(runtime_cfg["attack_fade_ms"] / 1000.0 * sr))
        self.stage = _ATK
        self.env = 0.0

    def _compute_gain_k(self, f0_play):
        """Formant preservation: amplitude of component k = spectral envelope value at absolute frequency ratio_k·f0_play, times brightness tilt.

        The envelope is defined by the snapshot (ratio_k·f0_ref, H_steady). Instead of carrying over by harmonic index (which shifts formants with
        pitch and makes voices lose their vowels), it resamples by absolute frequency: formants stay pinned, timbre does not change across notes. Returns a new array.
        """
        inst = self._inst
        K = self.K
        Hs = inst.H_steady[:K]
        ratio = inst.ratio_k[:K]
        xp = ratio * inst.f0_ref  # absolute frequencies of the snapshot components, i.e. envelope sample points
        targets = ratio * f0_play  # absolute frequencies of the components of the currently played note
        g = np.interp(targets, xp, Hs, left=Hs[0], right=Hs[-1])
        hi = xp[-1]  # high frequencies outside support extrapolate with a 12 dB/oct decay
        above = targets > hi
        if np.any(above) and Hs[-1] > 0 and hi > 0:
            octv = np.log2(np.maximum(targets[above], hi) / hi)
            g[above] = Hs[-1] * 10.0 ** (-12.0 * octv / 20.0)
        # Formant preservation only holds for sources with formant structure. For near-sinusoidal sources (whistling, rubbed glass: almost all
        # energy in the fundamental) the envelope is a single spike, and resampling by absolute frequency would sample the played note's 2nd/3rd
        # harmonics onto the source fundamental peak, so blend between formant preservation and harmonic preservation by spectral centroid.
        fw = getattr(inst, "_formant_w", None)
        if fw is None:
            hs = np.abs(np.asarray(inst.H_steady, np.float64))
            cen = float((np.arange(1, len(hs) + 1) * hs).sum() / (hs.sum() + 1e-12))
            fw = inst._formant_w = float(min(1.0, max(0.0, (cen - 1.2) / 1.0)))
        if fw < 1.0:
            g = fw * g + (1.0 - fw) * Hs
        g = g * (10.0 ** (self._tilt * inst.oct_k[:K] / 20.0))
        gmax = float(np.max(np.abs(g))) if K else 0.0
        if gmax > 0:  # drop components below -60 dB, the kernel skips them
            g[np.abs(g) < 1e-3 * gmax] = 0.0
        return g

    def note_off(self):
        if self.active and self.stage != _DONE:
            if self._ctl_f0 is not None:
                # in control-driven mode the envelope did not take part, so release must start from full scale, otherwise it first drops to the sustain value and ticks
                self.env = 1.0
            self.stage = _REL

    def steal(self, steal_ms: float):
        """Voice stealing: enter fast release (3–5 ms); once faded, the engine reclaims it for reuse."""
        if self.active:
            self._rel_inc = 1.0 / max(1, steal_ms / 1000.0 * self.sr)
            self.stage = _REL

    def set_bend(self, semitones: float):
        self._bend_target = 2.0 ** (semitones / 12.0)  # render_into follows with per-block smoothing, removing sparse MIDI steps

    def init_bend(self, semitones: float):
        """note_on inherits the channel pitch bend: applied instantly, the new note starts at the bent pitch instead of sliding from centre."""
        self.bend = self.bend_prev = self._bend_target = 2.0 ** (semitones / 12.0)

    def glide_to(
        self, f0_hz: float, dur_samp: int, curve=None, kind: str | None = None, vel: int = 90
    ):
        """Slide from the current f0_base to f0_hz within dur_samp along curve (g(τ)∈[0,1]).

        Continuation does not reset phase/ADSR (fast legato slide without a hard hit, slide follows the library curve). When kind is 'legato' or
        'slide', retrigger the attack grain at low level so the note change is audible, and set the friction-noise swell strength by interval
        (applied to the noise layer at render time, a random texture without stepping glitches).
        """
        self._glide_from = self.f0_base
        self._glide_to = float(f0_hz)
        self._glide_total = max(1, int(dur_samp))
        self._glide_pos = 0
        self._glide_curve = np.asarray(curve, dtype=np.float64) if curve is not None else None
        self.f0_target = float(f0_hz)
        if kind in ("legato", "slide"):
            vv = max(0, min(127, int(vel))) / 127.0
            base = _P.regrain_legato if kind == "legato" else _P.regrain_slide
            self.attack_pos = 0  # low-level retrigger, effective for sources with strong grains
            self._grain_scale = base * (0.6 + 0.6 * vv)
            self._art_pos = 0  # note-change dip: a short 50 ms arc, like a bow change or breath
            self._art_len = max(1, int(0.05 * self.sr))
            self._art_depth = _P.art_dip_legato if kind == "legato" else _P.art_dip_slide
            if self._glide_from > 0:
                oct_span = abs(np.log2(max(self._glide_to, 1e-6) / self._glide_from))
                self._glide_stren = min(1.0, oct_span * 1.6)  # the larger the interval, the more audible the friction
            else:
                self._glide_stren = 0.0
        # formants are redirected to the target pitch, but gain_k interpolates along the glide: an abrupt change means every harmonic amplitude jumps,
        # and the start of a slide/legato would click
        if self._inst is not None:
            self._gain_from = self.gain_k.copy()
            self._gain_to = self._compute_gain_k(f0_hz)

    def set_vibrato(self, depth_cents: float, rate_hz: float = None):
        """CC1 vibrato depth (cents); rate from library measurement or default. depth=0 turns it off."""
        if depth_cents > 0.0 and self._vib_depth <= 0.0:
            self._vib_age = 0  # new vibrato onset, go through delay and fade-in again
        self._vib_depth = float(depth_cents)
        if rate_hz:
            self._vib_rate = float(rate_hz)

    def _glide_f0(self, pos):
        if self._glide_total <= 1 or pos >= self._glide_total:
            return self._glide_to
        tau = pos / self._glide_total
        if self._glide_curve is not None and len(self._glide_curve) >= 2:
            g = float(np.interp(tau, np.linspace(0, 1, len(self._glide_curve)), self._glide_curve))
        else:
            g = tau
        ratio = self._glide_to / self._glide_from if self._glide_from > 0 else 1.0
        return self._glide_from * ratio**g

    def _advance_adsr(self, n: int):
        """Advance the ADSR by n samples and return (env_start, env_end). Linear stages."""
        e0 = self.env
        e = self.env
        st = self.stage
        if st == _ATK:
            e += self._atk_inc * n
            if e >= 1.0:
                e = 1.0
                self.stage = _DEC
        elif st == _DEC:
            e -= self._dec_inc * n
            if e <= self._sustain:
                e = self._sustain
                self.stage = _SUS
        elif st == _SUS:
            e = self._sustain
        elif st == _REL:
            e -= self._rel_inc * n
            if e <= _SILENCE:
                e = 0.0
                self.stage = _DONE
                self.active = False
        self.env = e
        return e0, e

    def render_into(self, block: np.ndarray, n: int):
        """Accumulate this voice's n samples into block[:n] (block is a view of the main mix buffer)."""
        if not self.active:
            return
        inst = self._inst
        env_a, env_b = self._advance_adsr(n)

        # harmonic fade-in (overlaps the attack grain by about 20 ms): 0->1
        s0 = self.samples_since_on
        fi_a = min(1.0, s0 / self._fadein_len)
        fi_b = min(1.0, (s0 + n) / self._fadein_len)

        loud_a = self.vel_loud * env_a * fi_a
        loud_b = self.vel_loud * env_b * fi_b

        # natural amplitude envelope (plucked sources decay, sustained ≈1): sampled by time since note start; beyond the table, keep decaying at
        # the tail rate, since clamping to the last frame would freeze at the reverb level and ring forever
        ae = inst.amp_env
        ib = 0
        if ae is not None:
            hop = inst.amp_hop
            L = len(ae) - 1
            ja, jb = s0 // hop, (s0 + n) // hop
            tr = inst.amp_tail_ratio
            fa = ae[min(ja, L)] * (tr ** max(0, ja - L))
            fb = ae[min(jb, L)] * (tr ** max(0, jb - L))
            ib = min(jb, L)
            loud_a *= fa
            loud_b *= fb

        ctl = self._ctl_f0 is not None
        if ctl:  # control-driven: loudness = velocity × external amplitude frame; ADSR only acts in the release stage
            cf0_a, camp_a = self._ctl_at(s0)
            cf0_b, camp_b = self._ctl_at(s0 + n)
            # multiply by the release envelope in the release stage, including the block that just entered _DONE: otherwise the last block renders
            # at full scale and then goes silent, a click after key release. Other stages leave the envelope to the control stream.
            rel_a, rel_b = (env_a, env_b) if self.stage in (_REL, _DONE) else (1.0, 1.0)
            loud_a = self.vel_loud * camp_a * fi_a * rel_a
            loud_b = self.vel_loud * camp_b * fi_b * rel_b

        # f0 ramp within the block = glide (f0 slides along the curve) × pitch-bend × vibrato
        pos0 = self._glide_pos
        pos1 = min(self._glide_total, pos0 + n)
        base_a = self._glide_f0(pos0)
        base_b = self._glide_f0(pos1)
        self._glide_pos = pos1
        self.f0_base = base_b
        # slide progress (a 0->1->0 arc), drives the friction-noise swell
        slide_u = (
            np.sin(np.pi * pos1 / self._glide_total)
            if (self._glide_total > 1 and pos1 < self._glide_total and self._glide_stren > 0)
            else 0.0
        )
        # gain_k interpolates toward the target formant mapping along the glide, removing clicks at slide/legato start
        if self._gain_to is not None:
            frac = pos1 / self._glide_total if self._glide_total > 0 else 1.0
            if frac >= 1.0:
                self.gain_k[:] = self._gain_to
                self._gain_to = None
            else:
                self.gain_k[:] = self._gain_from + (self._gain_to - self._gain_from) * frac
        if self._vib_depth > 0.0 and (not ctl or self._ctl_done):
            # vibrato starts after vib_onset_ms delay, depth fades in with smoothstep over vib_ramp_ms
            # (in control-driven mode it only takes over after the library trajectory ends, avoiding double vibrato)
            on = _P.vib_onset_ms * self.sr / 1000.0
            rp = max(1.0, _P.vib_ramp_ms * self.sr / 1000.0)
            u = min(1.0, max(0.0, (self._vib_age - on) / rp))
            venv = u * u * (3.0 - 2.0 * u)  # smoothstep, vibrato goes from shallow to deep
            self._vib_age += n
            depth = self._vib_depth * venv
            va = 2.0 ** (depth * np.sin(self._vib_phase) / 1200.0)
            self._vib_phase += 2.0 * np.pi * self._vib_rate * n / self.sr
            vb = 2.0 ** (depth * np.sin(self._vib_phase) / 1200.0)
        else:
            va = vb = 1.0
        # pitch-bend rate limit (semitone domain): touch-strip bends can teleport (±4 st / 12 ms, 1000+ st/s), and rendering
        # that literally gurgles. The phrasing limit is about 163 st/s (13 Hz manual vibrato × ±2 st); a 250 st/s limit does not affect
        # real playing and turns a teleport into a slide of about 16 ms. With no bend, bend == target == 1.0 and the branch is skipped.
        tgt = self._bend_target
        if self.bend != tgt:
            cur_st = 12.0 * np.log2(self.bend)
            step = 12.0 * np.log2(tgt) - cur_st
            lim = _BEND_RATE_ST_S * (n / self.sr)
            if step > lim:
                step = lim
            elif step < -lim:
                step = -lim
            self.bend = tgt if abs(step) < 1e-9 else 2.0 ** ((cur_st + step) / 12.0)
        if ctl:  # control-driven: f0 comes from the external frame stream
            base_a, base_b = cf0_a, cf0_b
            self.f0_base = base_b
            # the formant mapping follows f0 after 120 ms first-order smoothing: slides/note changes still change colour, but vibrato (5–7 Hz)
            # does not sample back and forth across spectral peaks, otherwise harmonic amplitudes would wobble with the vibrato
            self._ctl_f0_slow += (base_b - self._ctl_f0_slow) * (1.0 - np.exp(-n / (0.12 * self.sr)))
            self.gain_k[:] = self._compute_gain_k(self._ctl_f0_slow)
        f0_a = base_a * self.bend_prev * va
        f0_b = base_b * self.bend * vb
        self.bend_prev = self.bend

        # humanisation: micro pitch jitter + micro amplitude tremor, two LFO sets with incommensurate frequencies, deterministic.
        # Disabled in control-driven mode: micro-dynamics come from the behaviour library, and another jitter layer would blur the retrieved vibrato
        if (_P.hum_cents > 0.0 or _P.hum_amp > 0.0) and not ctl:
            t0 = s0 / self.sr
            t1 = (s0 + n) / self.sr
            ph = self._hum_off
            jit0 = _P.hum_cents * (
                0.6 * np.sin(2 * np.pi * 4.3 * t0 + ph)
                + 0.4 * np.sin(2 * np.pi * 6.7 * t0 + 1.7 * ph)
            )
            jit1 = _P.hum_cents * (
                0.6 * np.sin(2 * np.pi * 4.3 * t1 + ph)
                + 0.4 * np.sin(2 * np.pi * 6.7 * t1 + 1.7 * ph)
            )
            f0_a *= 2.0 ** (jit0 / 1200.0)
            f0_b *= 2.0 ** (jit1 / 1200.0)
            loud_a *= 1.0 + _P.hum_amp * np.sin(2 * np.pi * 3.1 * t0 + 0.5 * ph)
            loud_b *= 1.0 + _P.hum_amp * np.sin(2 * np.pi * 3.1 * t1 + 0.5 * ph)

        # ---- Expression layer ----
        # spectral trajectory: ping-pong loop along the shape motion of the source recording's sustain, start/rate/direction staggered per voice.
        # brightness follows level: loud is bright, decay is darker (dark_db·oct_k tilt).
        gk = self.gain_k
        # attack spectral trajectory: each partial's own attack/early decay, the physical source of the plucked feel, produced by the harmonic layer itself
        atk = inst.atk_traj
        if atk is not None and self._atk_traj_pos < (1 << 29):
            Fa = atk.shape[1]
            p_a = self._atk_traj_pos * inst.atk_fps / self.sr
            if p_a >= Fa - 1:
                self._atk_traj_pos = 1 << 30  # attack finished, hand back to the steady-state path
            else:
                i0 = int(p_a)
                i1 = min(i0 + 1, Fa - 1)
                fr = p_a - i0
                Kv = min(self.K, atk.shape[0])
                avec = atk[:Kv, i0] * (1.0 - fr) + atk[:Kv, i1] * fr
                buf = self._gain_buf
                buf[: self.K] = gk[: self.K]
                buf[:Kv] *= avec
                gk = buf
                self._atk_traj_pos += n
        live_amp = 1.0  # loudness-breathing factor, level fluctuation of the source sustain
        st = inst.spec_traj
        if st is not None and _P.live_spec > 0.0:
            F = st.shape[1]
            Kv = min(self.K, st.shape[0])
            p = self._traj_pos
            i0 = int(p)
            i1 = min(i0 + 1, F - 1)
            fr = p - i0
            fvec = st[:Kv, i0] * (1.0 - fr) + st[:Kv, i1] * fr
            buf = self._gain_buf
            if gk is not buf:  # may already have been rewritten by the attack trajectory
                buf[: self.K] = gk[: self.K]
            buf[:Kv] *= 1.0 + _P.live_spec * (fvec - 1.0)  # depth shrinks the factor toward 1
            gk = buf
            at = inst.amp_traj
            if at is not None:  # loudness breathing: a uses the previous block's factor, b this block's, no boundary steps
                af = at[i0] * (1.0 - fr) + at[i1] * fr
                now = 1.0 + _P.live_spec * (af - 1.0)
                loud_a *= self._amp_prev
                loud_b *= now
                live_amp = 0.5 * (self._amp_prev + now)  # the noise layer uses the midpoint, the random texture masks steps
                self._amp_prev = now
            adv = self._traj_dir * self._traj_rate * inst.traj_fps * (n / self.sr)
            p2 = p + adv
            if p2 >= F - 1.0:  # ping-pong bounce, no wrap seam
                p2 = 2.0 * (F - 1.0) - p2
                self._traj_dir = -self._traj_dir
            if p2 <= 0.0:
                p2 = -p2
                self._traj_dir = abs(self._traj_dir)
            self._traj_pos = min(max(p2, 0.0), F - 1.0)
        if self._art_pos < self._art_len:  # note-change dip: short sine arc, a/b sampling keeps in-block interpolation continuous
            pa = self._art_pos / self._art_len
            pb = min(1.0, (self._art_pos + n) / self._art_len)
            loud_a *= 1.0 - self._art_depth * np.sin(np.pi * pa)
            loud_b *= 1.0 - self._art_depth * np.sin(np.pi * pb)
            self._art_pos += n
        tilt_dyn = _P.tone_tilt  # dynamic tilt (dB/oct): Brightness knob, darkening and brightness breathing merged into one power
        if ctl and self._ctl_tilt is not None:  # the behaviour library's brightness motion, centroid residual -> dB/oct
            tilt_dyn += self._ctl_aux(self._ctl_tilt, s0 + n // 2)
        if _P.live_dark > 0.0:
            lvl = env_b * fi_b
            if ae is not None:
                lvl *= ae[ib]
            tilt_dyn -= _P.live_dark * (1.0 - min(1.0, lvl))  # lower level is darker
        if _P.live_breath > 0.0:
            tb = s0 / self.sr  # slow brightness breathing: two incommensurate frequencies, phase offset per voice
            ph = self._hum_off * 2.1
            tilt_dyn += _P.live_breath * (
                0.6 * np.sin(2 * np.pi * 0.31 * tb + ph)
                + 0.4 * np.sin(2 * np.pi * 0.47 * tb + 1.9 * ph)
            )
        # 150 ms first-order smoothing removes tilt jumps and is applied continuously with no gate: an amplitude gate would toggle at LFO zero
        # crossings and produce residual pulses. After the knob returns to zero, _tilt_s still has to slide back to 0, otherwise the last block's tilt freezes in the voice.
        if _P.live_breath > 0.0 or _P.live_dark > 0.0 or tilt_dyn != 0.0 or self._tilt_s != 0.0:
            alpha = 1.0 - np.exp(-n / (0.15 * self.sr))
            self._tilt_s += (tilt_dyn - self._tilt_s) * alpha
            if gk is self.gain_k:  # did not go through the trajectory branch, copy into the buffer first
                buf = self._gain_buf
                buf[: self.K] = gk[: self.K]
                gk = buf
            gk[: self.K] *= 10.0 ** (self._tilt_s * inst.oct_k[: self.K] / 20.0)

        nyq = self.sr * 0.5
        voice_block_add(
            block,
            n,
            self.phase,
            self.ratio,
            self._gain_prev,
            gk,
            self.K,
            f0_a,
            f0_b,
            loud_a,
            loud_b,
            float(self.sr),
            nyq,
            _P.ramp_hz,
        )
        self._gain_prev[: self.K] = gk[: self.K]  # this block's final value is the next block's start, continuous across blocks

        # attack grain playback (velocity-scaled), overlapping the harmonic fade-in
        grain = inst.attack_grain
        gp = self.attack_pos
        if gp < len(grain):
            m = min(n, len(grain) - gp)
            block[:m] += grain[gp : gp + m] * self.vel_loud * self._grain_scale
        self.attack_pos = gp + n

        # noise texture (table lookup, ADSR-scaled); fixed table plus reset pointer keeps renders reproducible
        tbl = inst.noise_table
        nl = inst.noise_level
        if ctl and self._ctl_nz is not None:  # the behaviour library's breath/friction motion, flatness residual -> multiplier
            nl = nl * self._ctl_aux(self._ctl_nz, s0 + n // 2)
        # slide friction is injected as an absolute amount, independent of the Air/Noise floor, but still uses the source's own noise texture
        if slide_u > 0.0:
            nl = (
                nl + _P.slide_noise * slide_u * self._glide_stren
            )
        if nl > 0 and len(tbl) > 0:
            np_ = self.noise_pos
            env_mid = 0.5 * (env_a + env_b) * self.vel_loud * nl * live_amp
            if np_ + n <= len(tbl):
                block[:n] += tbl[np_ : np_ + n] * env_mid
                self.noise_pos = (np_ + n) % len(tbl)
            else:  # wrap around the table end
                m = len(tbl) - np_
                block[:m] += tbl[np_:] * env_mid
                block[m:n] += tbl[: n - m] * env_mid
                self.noise_pos = n - m

        self.samples_since_on = s0 + n


# ---- Public setters: engine/config/knobs -> _P ----


def set_nyquist_ramp_hz(hz: float):
    _P.ramp_hz = float(hz)


def set_humanize(cents: float, amp: float):
    _P.hum_cents = float(cents)
    _P.hum_amp = float(amp)


def set_vibrato_shape(onset_ms: float, ramp_ms: float):
    _P.vib_onset_ms = float(onset_ms)
    _P.vib_ramp_ms = float(ramp_ms)


def set_slide_feel(
    noise_bump: float,
    regrain_legato: float,
    regrain_slide: float,
    dip_legato: float = 0.30,
    dip_slide: float = 0.22,
):
    _P.slide_noise = float(noise_bump)
    _P.regrain_legato = float(regrain_legato)
    _P.regrain_slide = float(regrain_slide)
    _P.art_dip_legato = float(dip_legato)
    _P.art_dip_slide = float(dip_slide)


def set_tone_tilt(db_oct: float):
    _P.tone_tilt = float(db_oct)


def set_articulation_intensity(scale: float):
    s = max(0.0, float(scale))
    n, rl, rs, dl, ds = _SLIDE_FEEL_BASE
    set_slide_feel(n * s, rl * s, rs * s, min(0.6, dl * s), min(0.6, ds * s))


def regrain_legato() -> float:
    return _P.regrain_legato


def set_liveness(spec: float, dark_db_oct: float, breath_db_oct: float = 0.6):
    _P.live_spec = float(spec)
    _P.live_dark = float(dark_db_oct)
    _P.live_breath = float(breath_db_oct)
