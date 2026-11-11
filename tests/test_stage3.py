"""Stage 3 unit tests: ADSR time constants / velocity mapping / voice stealing / sample-accurate triggering / offline determinism."""

import numpy as np
import pytest

from runtime.engine import Engine, Event
from synth.instrument import Instrument
from synth.voice import _REL, _SUS, Voice

SR = 44100
VEL_CFG = {"loud_exp": 1.5, "tilt_db_oct": 6.0}
RT = {"attack_fade_ms": 20.0, "steal_fade_ms": 4.0, "nyquist_ramp_hz": 600.0}


def _inst(K=8, noise=0.0):
    H = 1.0 / np.arange(1, K + 1)
    return Instrument(
        H,
        np.arange(1, K + 1, dtype=float),
        np.zeros(0),
        np.zeros(0),
        220.0,
        SR,
        name="t",
        noise_level=noise,
    )


def _engine(K=8, noise=0.0, eval_mode=True):
    cfg = {
        "runtime": {
            "blocksize": 256,
            "max_voices": 16,
            "channels": 1,
            "noise_level": noise,
            "attack_fade_ms": 20.0,
            "steal_fade_ms": 4.0,
            "nyquist_ramp_hz": 600.0,
        },
        "adsr": {"attack_ms": 8.0, "decay_ms": 120.0, "sustain": 0.7, "release_ms": 180.0},
        "velocity": VEL_CFG,
    }
    return Engine(_inst(K, noise), cfg=cfg, eval_mode=eval_mode)


# --------------------------------------------------------------------------- #
# ADSR time constants measured vs set ±10%
# --------------------------------------------------------------------------- #
def test_adsr_attack_release_within_10pct():
    adsr = {"attack_ms": 12.0, "decay_ms": 60.0, "sustain": 0.6, "release_ms": 150.0}
    v = Voice(8, SR)
    v.note_on(69, 100, _inst(), adsr, VEL_CFG, RT, age=1)

    n = 8
    t = 0
    # measure attack time: env first reaches ~1.0
    atk_t = None
    while t < int(SR * 0.05):
        _, e = v._advance_adsr(n)
        t += n
        if atk_t is None and e >= 0.999:
            atk_t = t / SR * 1000.0
    assert atk_t is not None
    assert abs(atk_t - adsr["attack_ms"]) / adsr["attack_ms"] < 0.10, (
        f"attack {atk_t:.2f}ms vs set {adsr['attack_ms']}ms"
    )

    # push to sustain, note_off, measure from sustain to silence and convert to full scale
    while v.stage != _SUS and t < int(SR * 0.5):
        v._advance_adsr(n)
        t += n
    sus_level = v.env
    v.note_off()
    t_off = 0
    while v.active and t_off < int(SR * 0.5):
        _, e = v._advance_adsr(n)
        t_off += n
    rel_ms = (t_off / SR * 1000.0) / sus_level  # normalised to full scale
    assert abs(rel_ms - adsr["release_ms"]) / adsr["release_ms"] < 0.10, (
        f"release (full scale) {rel_ms:.1f}ms vs set {adsr['release_ms']}ms"
    )


# --------------------------------------------------------------------------- #
# velocity mapping: loud=(vel/127)^1.5; higher velocity is brighter
# --------------------------------------------------------------------------- #
def test_velocity_loudness_and_brightness():
    v = Voice(8, SR)
    inst = _inst(8)
    adsr = {"attack_ms": 8.0, "decay_ms": 60.0, "sustain": 0.7, "release_ms": 150.0}

    v.note_on(69, 127, inst, adsr, VEL_CFG, RT, age=1)
    assert v.vel_loud == pytest.approx(1.0, abs=1e-6)
    bright_hi = v.gain_k[6] / v.gain_k[0]  # high/low harmonic ratio

    v.note_on(69, 40, inst, adsr, VEL_CFG, RT, age=2)
    assert v.vel_loud == pytest.approx((40 / 127) ** 1.5, rel=1e-5)
    bright_lo = v.gain_k[6] / v.gain_k[0]

    assert bright_hi > bright_lo * 1.5, "high velocity should be clearly brighter (relatively stronger high harmonics)"


# --------------------------------------------------------------------------- #
# voice stealing: prefer the oldest released voice, otherwise the oldest
# --------------------------------------------------------------------------- #
def test_voice_steal_prefers_oldest_released():
    eng = _engine()
    # fill all 16 voices, age increasing
    for i in range(16):
        eng._apply_event(Event("on", 60 + i, 100, 0.0))
    # put two of them into release; the one with smaller age should be stolen
    eng.pool[5].stage = _REL
    eng.pool[5].age = 1
    eng.pool[9].stage = _REL
    eng.pool[9].age = 3
    victim = eng._alloc_voice()
    assert victim is eng.pool[5], "should steal the released voice with the smallest age"


def test_alloc_prefers_free_voice():
    eng = _engine()
    for i in range(3):
        eng._apply_event(Event("on", 60 + i, 100, 0.0))
    v = eng._alloc_voice()
    assert not v.active, "an idle voice should be preferred when available"


# --------------------------------------------------------------------------- #
# sample-accurate triggering within a block: silent before note-on, sounding after
# --------------------------------------------------------------------------- #
def test_sample_accurate_onset():
    eng = _engine()
    y = eng.render_offline([(0.05, "on", 69, 110, 0.0)], 0.2)
    on = int(np.argmax(np.abs(y) > 0.01))
    expect = int(0.05 * SR)
    assert abs(on - expect) < 512, f"onset {on}, expected ~{expect}"
    assert np.max(np.abs(y[: expect - 300])) < 1e-4, "should be silent before note-on"


def test_pitch_matches_note():
    eng = _engine(K=20)
    y = eng.render_offline([(0.02, "on", 57, 100, 0.0), (0.45, "off", 57, 0, 0.0)], 0.5)
    seg = y[int(0.1 * SR) : int(0.35 * SR)]
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    f0 = 440.0 * 2 ** ((57 - 69) / 12.0)  # A3 = 220Hz
    assert abs(f[np.argmax(S)] - f0) < 5.0, f"f0 {f[np.argmax(S)]:.1f} vs {f0:.1f}"


# --------------------------------------------------------------------------- #
# offline rendering: determinism + bounded, finite bus
# --------------------------------------------------------------------------- #
def test_offline_determinism():
    evs = [
        (0.0, "on", 60, 90, 0.0),
        (0.1, "on", 64, 90, 0.0),
        (0.3, "off", 60, 0, 0.0),
        (0.3, "off", 64, 0, 0.0),
    ]
    y1 = _engine(noise=0.5).render_offline(evs, 0.5)
    y2 = _engine(noise=0.5).render_offline(evs, 0.5)
    assert np.array_equal(y1, y2), "eval_mode with a fixed seed should be identical sample by sample"


def test_master_bus_bounded_finite():
    eng = _engine(K=40, noise=0.5)
    # 16 notes at once, full load
    evs = [(0.0, "on", 48 + i, 120, 0.0) for i in range(16)]
    y = eng.render_offline(evs, 0.3)
    assert np.all(np.isfinite(y))
    # tanh approaches ±1 asymptotically (float32 may round to 1.0); require bounded, no overflow
    assert np.max(np.abs(y)) <= 1.0, "tanh soft limiting should keep the bus within ±1"
    assert not eng.error_flag, "rendering should not set the error flag"


def test_callback_never_crashes_on_bad_state():
    """Outermost try/except in the callback: even with a broken instrument, output silence and do not raise."""
    eng = _engine()
    eng.instrument = None  # deliberately broken
    out = np.zeros(256, dtype=np.float32)
    eng._apply_event_safe = None
    eng.process_block(out, 256, 0)  # should not raise
    # note: rendering needs an active voice to trigger; this only checks that nothing is raised
    assert out.dtype == np.float32
