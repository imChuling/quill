"""Stage 6 unit tests: gesture interpreter (pluck/legato/slide/strum/vibrato) + glide rendering."""

import numpy as np

from runtime.engine import Engine
from runtime.gesture import GestureInterpreter, canonical_slide_curve
from synth.instrument import Instrument

SR = 44100


def _inst(K=30):
    return Instrument(
        1.0 / np.arange(1, K + 1),
        np.arange(1, K + 1, dtype=float),
        np.zeros(0),
        np.zeros(0),
        220.0,
        SR,
        noise_level=0.0,
    )


def _dom(y, sr=SR):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(S)])


def _kinds(cooked):
    return [c[1] for c in cooked]


# --------------------------------------------------------------------------- #
# gesture decisions
# --------------------------------------------------------------------------- #
def test_single_note_is_pluck():
    gi = GestureInterpreter()
    cooked = gi.process([(0.1, "on", 60, 100), (0.5, "off", 60, 0)])
    assert _kinds(cooked) == ["on", "off"]


def test_close_interval_is_legato():
    gi = GestureInterpreter()
    raw = [
        (0.1, "on", 60, 100),
        (0.3, "on", 62, 90),  # interval 2 semitones -> legato
        (0.6, "off", 62, 0),
        (0.6, "off", 60, 0),
    ]
    cooked = gi.process(raw)
    leg = [c for c in cooked if c[1] == "legato"]
    assert len(leg) == 1 and leg[0][5] == 60 and leg[0][2] == 62


def test_far_interval_is_slide():
    gi = GestureInterpreter()
    raw = [
        (0.1, "on", 60, 100),
        (0.3, "on", 67, 90),  # interval 7 semitones -> slide
        (0.6, "off", 67, 0),
        (0.6, "off", 60, 0),
    ]
    cooked = gi.process(raw)
    sl = [c for c in cooked if c[1] == "slide"]
    assert len(sl) == 1 and sl[0][5] == 60 and sl[0][2] == 67


def test_chord_burst_is_strum():
    gi = GestureInterpreter()
    raw = [
        (1.0, "on", 48, 100),
        (1.015, "on", 52, 100),
        (1.03, "on", 55, 100),
        (1.5, "off", 48, 0),
        (1.5, "off", 52, 0),
        (1.5, "off", 55, 0),
    ]
    cooked = gi.process(raw)
    ons = [c for c in cooked if c[1] == "on"]
    assert len(ons) == 3, "a chord burst should expand into 3 staggered attacks"
    # staggered: the attack times all differ
    ts = sorted(c[0] for c in ons)
    assert ts[1] > ts[0] and ts[2] > ts[1], "strum should have string-order offsets"


def test_two_notes_not_strum():
    gi = GestureInterpreter()
    raw = [(1.0, "on", 48, 100), (1.02, "on", 55, 100)]  # only 2 notes -> not a strum
    cooked = gi.process(raw)
    assert "slide" in _kinds(cooked) or "legato" in _kinds(cooked)
    assert _kinds(cooked).count("on") == 1


def test_velocity_affects_slide_speed():
    gi = GestureInterpreter()
    soft = gi.process([(0.0, "on", 60, 30), (0.2, "on", 67, 30)])
    hard = gi.process([(0.0, "on", 60, 120), (0.2, "on", 67, 120)])
    ds = next(c[4] for c in soft if c[1] == "slide")
    dh = next(c[4] for c in hard if c[1] == "slide")
    assert dh < ds, "higher velocity slides faster (shorter duration)"


# --------------------------------------------------------------------------- #
# rendering: glide / vibrato really happen
# --------------------------------------------------------------------------- #
def test_slide_renders_f0_glide():
    gi = GestureInterpreter()
    cooked = gi.process([(0.1, "on", 60, 100), (0.5, "on", 72, 100), (1.2, "off", 72, 0)])
    eng = Engine(_inst(), eval_mode=True)
    gi.apply_to_engine(eng)
    y = eng.render_offline(cooked, 1.4)
    f_before = _dom(y[int(0.25 * SR) : int(0.45 * SR)])
    f_after = _dom(y[int(0.9 * SR) : int(1.1 * SR)])
    assert abs(f_before - 261.6) < 8, f"before the slide should be ~262Hz, got {f_before:.0f}"
    assert abs(f_after - 523.3) < 12, f"after the slide should be ~523Hz (note72), got {f_after:.0f}"


def test_cc_renders_vibrato():
    from analysis.harmonic import track_f0

    gi = GestureInterpreter()
    cooked = gi.process([(0.1, "on", 69, 100), (0.3, "cc", -1, 110), (2.0, "off", 69, 0)])
    eng = Engine(_inst(), eval_mode=True)
    gi.apply_to_engine(eng)
    y = eng.render_offline(cooked, 2.2)
    f0, _, _ = track_f0(y[int(0.5 * SR) : int(1.8 * SR)])
    m = np.isfinite(f0) & (f0 > 0)
    assert np.std(f0[m]) > 2.0, f"CC1 should trigger vibrato (f0 std>2Hz), got {np.std(f0[m]):.2f}"


def test_full_phrase_renders_clean():
    gi = GestureInterpreter()
    raw = [
        (0.1, "on", 60, 100),
        (0.5, "on", 62, 90),
        (0.9, "on", 67, 95),
        (1.4, "on", 48, 100),
        (1.41, "on", 52, 100),
        (1.42, "on", 55, 100),
        (2.0, "off", 55, 0),
        (2.0, "off", 52, 0),
        (2.0, "off", 48, 0),
        (2.0, "off", 67, 0),
    ]
    eng = Engine(_inst(), eval_mode=True)
    gi.apply_to_engine(eng)
    y = eng.render_offline(gi.process(raw), 2.4)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0
    assert not eng.error_flag


# --------------------------------------------------------------------------- #
# library slide curve
# --------------------------------------------------------------------------- #
def test_fr9_recombination_decoupled():
    """FR-9: same articulation library + two different timbres -> same technique (f0 track), different timbre (harmonic structure)."""
    lib = {
        "slide": [
            {
                "interval_st": 5.0,
                "curve_ctrl": np.array([0, 0.3, 0.55, 0.72, 0.84, 0.92, 0.97, 1.0]),
            }
        ],
        "strum": [],
        "pluck": [],
        "legato": [],
        "sustain": None,
        "meta": {},
    }
    K = 8
    # two timbres: both have a strong fundamental (FFT peak = f0), but different harmonic rolloff
    Hx = np.array([1, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03])
    Hy = np.array([1, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12])
    instX = Instrument(
        Hx, np.arange(1, K + 1, dtype=float), np.zeros(0), np.zeros(0), 220.0, SR, noise_level=0.0
    )
    instY = Instrument(
        Hy, np.arange(1, K + 1, dtype=float), np.zeros(0), np.zeros(0), 220.0, SR, noise_level=0.0
    )
    gi = GestureInterpreter(library=lib)
    cooked = gi.process([(0.1, "on", 60, 100), (0.5, "on", 67, 100), (1.2, "off", 67, 0)])

    def _render(inst):
        eng = Engine(inst, eval_mode=True)
        gi.apply_to_engine(eng)
        return eng.render_offline(cooked, 1.4)

    yx, yy = _render(instX), _render(instY)

    # technique decoupling: the slide f0 end point is the same for both timbres (both ~523Hz = note67)
    fx = _dom(yx[int(0.9 * SR) : int(1.1 * SR)])
    fy = _dom(yy[int(0.9 * SR) : int(1.1 * SR)])
    assert abs(fx - fy) < 6, f"same articulation, different timbres should share the f0 track: {fx:.0f} vs {fy:.0f}"
    assert abs(fx - 392.0) < 12, f"slide to note67 (~392Hz=G4), got {fx:.0f}"

    # different timbres: different steady harmonic 2/harmonic 1 energy ratio
    def _h2h1(y):
        seg = y[int(0.25 * SR) : int(0.45 * SR)]
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        b1 = S[(f > 240) & (f < 285)].max()
        b2 = S[(f > 500) & (f < 545)].max()
        return b2 / (b1 + 1e-9)

    assert _h2h1(yx) != _h2h1(yy), "different timbres should have different harmonic structures"


def test_sustain_pedal_defers_release():
    """CC64 sustain pedal: note-off is sustained while the pedal is down, released only when it comes up."""
    eng = Engine(_inst(), eval_mode=True)
    # on@0.1, pedal down@0.3, off@0.5 (sustained), pedal up@1.2
    y = eng.render_offline(
        [
            (0.1, "on", 60, 100, 0.0),
            (0.3, "pedal", -1, 0, 1.0),
            (0.5, "off", 60, 0, 0.0),
            (1.2, "pedal", -1, 0, 0.0),
        ],
        1.8,
    )

    def rms(t0, t1):
        return float(np.sqrt(np.mean(y[int(t0 * SR) : int(t1 * SR)] ** 2)))

    assert rms(0.7, 0.9) > 0.05, "should still sound after off while the pedal is down"
    assert rms(1.5, 1.7) < 0.02, "should release after the pedal comes up"


def test_vibrato_rate_from_snapshot():
    """CC1 vibrato rate taken from the source clip measurement (snapshot.vibrato.rate_hz)."""
    from snapshot import make_snapshot
    from synth.instrument import build_instrument

    snap = make_snapshot(
        220.0,
        np.ones((60, 8), np.float32),
        np.ones((40, 8), np.float32),
        np.zeros(2646, np.float32),
        vibrato={"rate_hz": 6.8, "depth_cents": 25.0},
    )
    eng = Engine(build_instrument(snap, noise_level=0.0), eval_mode=True)
    assert abs(eng.vib_rate_hz - 6.8) < 1e-6


def test_canonical_slide_curve_monotonic():
    lib = {
        "slide": [
            {"interval_st": 2.0, "curve_ctrl": np.array([0, 0.3, 0.5, 0.7, 0.85, 0.92, 0.97, 1.0])},
            {
                "interval_st": 3.0,
                "curve_ctrl": np.array([0, 0.25, 0.48, 0.66, 0.8, 0.9, 0.96, 1.0]),
            },
            {
                "interval_st": 17.0,
                "curve_ctrl": np.array([0, 1, 0, 1, 0, 1, 0, 1.0]),
            },  # large noisy interval, should be filtered
        ]
    }
    g = canonical_slide_curve(lib)
    assert g is not None
    assert np.all(np.diff(g) >= -1e-9), "the canonical slide curve should be monotonic"
    assert abs(g[0]) < 1e-6 and abs(g[-1] - 1.0) < 1e-6
