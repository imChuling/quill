"""True Envelope (analysis/envelope.py) regression tests.

Note: measured True Envelope is **not better** than direct harmonic amplitude interpolation (see the negative result in docs/design_notes.md),
so it is not used in the synthesis path; this only verifies that the implementation itself is correct: the envelope covers the spectrum, passes through formants, valleys are below peaks.
"""

import numpy as np
import pytest

from analysis.envelope import envelope_from_audio, sample_envelope_db, true_envelope

SR = 44100


def _formant_tone(f0=174.6, dur=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t)
    for k in range(1, 60):
        fk = k * f0
        if fk >= sr / 2:
            break
        # two formants 700 / 1800 Hz + floor
        a = np.exp(-(((fk - 700) / 350) ** 2)) + 0.6 * np.exp(-(((fk - 1800) / 600) ** 2)) + 0.03
        y += a * np.cos(2 * np.pi * fk * t)
    return (y / np.max(np.abs(y))).astype(np.float32)


def test_envelope_covers_spectrum():
    """The True Envelope should lie (approximately) above the log magnitude spectrum everywhere."""
    rng = np.random.default_rng(0)
    logmag = -3 + 2 * np.sin(np.linspace(0, 8 * np.pi, 1025)) + 0.5 * rng.standard_normal(1025)
    env = true_envelope(logmag, order=40)
    # a small amount of overshoot within the convergence threshold is allowed
    over = np.mean(logmag - env > 0.3)
    assert over < 0.05, f"envelope does not cover the spectrum: {over:.2%} overshoot"


def test_envelope_tracks_formants():
    """The envelope is higher at a formant than at the adjacent valley."""
    y = _formant_tone()
    freqs, env = envelope_from_audio(y, SR, 174.6)
    f1 = sample_envelope_db(freqs, env, [700])[0]  # main formant
    valley = sample_envelope_db(freqs, env, [1200])[0]  # valley
    high = sample_envelope_db(freqs, env, [3500])[0]  # high-frequency rolloff
    assert f1 > valley > high, f"wrong envelope shape: F1={f1:.2f} valley={valley:.2f} high={high:.2f}"


def test_sample_extrapolation_decays():
    """Outside support (beyond the Nyquist sampling limit) it should decay monotonically and not blow up."""
    freqs = np.linspace(0, 8000, 200)
    env = np.linspace(0, -5, 200)
    a = sample_envelope_db(freqs, env, [7000, 12000, 20000])
    assert a[0] > a[1] > a[2] >= 0
    assert np.all(np.isfinite(a))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
