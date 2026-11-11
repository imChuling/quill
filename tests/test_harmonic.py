"""Harmonic extraction unit tests: stage 1 Day2 acceptance: synthetic known harmonic signal -> extraction error < 1dB."""

import numpy as np
import pytest

from analysis.harmonic import f0_scalar, harmonics, track_f0

SR = 44100


def _make_harmonic_tone(f0, amps, dur=1.0, sr=SR, phases=None):
    """Synthesise y = Σ amps[k] * cos(2π (k+1) f0 t + φ_k)."""
    t = np.arange(int(sr * dur)) / sr
    if phases is None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, size=len(amps))
    y = np.zeros_like(t)
    for k, a in enumerate(amps):
        y += a * np.cos(2 * np.pi * (k + 1) * f0 * t + phases[k])
    return y.astype(np.float32)


def _extract_amps(H, lo=0.25, hi=0.75):
    """Median per-harmonic amplitude over the middle frames (avoiding the center padding edges)."""
    T = H.shape[1]
    sl = slice(int(T * lo), int(T * hi))
    return np.median(H[:, sl], axis=1)


@pytest.mark.parametrize(
    "f0",
    [
        215.33203125,  # = 20 * 44100/4096, harmonics fall exactly on bins (zero scalloping loss)
        233.7,  # off bin, tests the parabolic interpolation correction
        146.83,  # D3, low
    ],
)
def test_harmonic_extraction_within_1db(f0):
    amps = np.array([1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.07, 0.05])
    y = _make_harmonic_tone(f0, amps)
    H, times, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))

    est = _extract_amps(H)
    err_db = 20 * np.log10(est / amps)
    assert np.max(np.abs(err_db)) < 1.0, (
        f"f0={f0}: max harmonic error {np.max(np.abs(err_db)):.3f} dB >= 1dB\n"
        f"per-harmonic error(dB)={np.round(err_db, 3)}"
    )


def test_nyquist_truncation():
    """Harmonics where k*f0 crosses Nyquist should be truncated to 0 (pitfall)."""
    f0 = 3000.0  # Nyquist=22050 -> the 8th harmonic at 24000Hz should be truncated
    amps = np.ones(10)
    y = _make_harmonic_tone(f0, amps)
    H, _, _ = harmonics(y, f0, sr=SR, n_harmonics=10)
    est = _extract_amps(H)
    # the first 7 harmonics (<=21000Hz) should have energy; from the 8th (>=24000Hz) on they should be 0
    assert np.all(est[:7] > 0.5)
    assert np.all(est[7:] < 1e-3)


def test_low_f0_auto_bumps_nfft():
    """A low f0 makes n_fft rise to 8192 automatically (pitfall: low-note harmonics are too closely spaced)."""
    f0 = 65.41  # C2, below the 100Hz threshold
    amps = np.array([1.0, 0.7, 0.5, 0.3])
    y = _make_harmonic_tone(f0, amps, dur=1.5)
    H, _, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))
    assert info["n_fft"] == 8192
    est = _extract_amps(H)
    err_db = 20 * np.log10(est / amps)
    assert np.max(np.abs(err_db)) < 1.0, f"low-note error {np.round(err_db, 3)} dB"


def test_quasi_harmonic_partial_measurement():
    """Quasi-harmonic source: wider window + record measured partial frequencies (wine glass/bowl path)."""
    f0 = 300.0
    # deliberately make the upper partials deviate from k*f0 (simulating wine-glass partial stretch)
    partials = [300.0, 605.0, 915.0, 1230.0]  # slightly above k*300
    t = np.arange(int(SR * 1.0)) / SR
    y = sum(a * np.cos(2 * np.pi * fp * t) for fp, a in zip(partials, [1.0, 0.6, 0.4, 0.3])).astype(
        np.float32
    )
    H, _, info = harmonics(y, f0, sr=SR, n_harmonics=4, measure_partials=True)
    pf = info["partial_freqs"]
    assert pf is not None
    # measured frequencies should be close to the true partials (±15Hz)
    assert np.allclose(pf, partials, atol=15.0), f"measured partials {np.round(pf, 1)}"


def test_track_f0_on_steady_tone():
    """track_f0 should give a median f0 close to the true value on a steady sine."""
    f0_true = 220.0
    t = np.arange(int(SR * 1.0)) / SR
    y = (0.7 * np.sin(2 * np.pi * f0_true * t)).astype(np.float32)
    f0, conf, times = track_f0(y, sr=SR)
    med = f0_scalar(f0)
    cents = 1200 * np.log2(med / f0_true)
    assert abs(cents) < 50.0, f"median f0={med:.2f}Hz off by {cents:+.1f} cents"
    assert len(f0) == len(conf) == len(times)
