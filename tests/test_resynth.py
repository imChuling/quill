"""Additive resynthesis unit tests: stage 1 Day3 acceptance: analysis->resynthesis round-trip error + no aliasing."""

import numpy as np

from analysis.harmonic import harmonics
from synth.additive import resynth

SR = 44100


def _make_harmonic_tone(f0, amps, dur=1.0, sr=SR, seed=1):
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=len(amps))
    y = sum(a * np.cos(2 * np.pi * (k + 1) * f0 * t + phases[k]) for k, a in enumerate(amps))
    return y.astype(np.float32)


def test_additive_rms_matches_theory():
    """Feed a known H directly: output RMS should equal sqrt(Σ A_k²/2) (verifies the additive math)."""
    f0 = 200.0
    amps = np.array([1.0, 0.5, 0.25, 0.125])
    T = 100
    H = np.tile(amps[:, None], (1, T)).astype(np.float32)
    y = resynth(H, f0, sr=SR, hop=512)
    rms = np.sqrt(np.mean(y[2000:-2000] ** 2))
    rms_theory = np.sqrt(np.sum(amps**2) / 2.0)
    err_db = 20 * np.log10(rms / rms_theory)
    assert abs(err_db) < 0.5, f"RMS error {err_db:.3f} dB"


def test_analysis_resynth_roundtrip_within_1db():
    """Full round trip: original -> harmonics -> resynth -> reanalysis, per harmonic < 1dB."""
    f0 = 196.0  # G3
    amps = np.array([1.0, 0.7, 0.5, 0.35, 0.2, 0.12, 0.08])
    y = _make_harmonic_tone(f0, amps, dur=1.2)

    H, times, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))
    y_re = resynth(H, info["f0_per_frame"], sr=SR, hop=512, length=len(y))

    # run harmonic analysis on the resynthesis again and compare with the original amplitudes
    H2, _, _ = harmonics(y_re, f0, sr=SR, n_harmonics=len(amps))
    T2 = H2.shape[1]
    est = np.median(H2[:, int(T2 * 0.3) : int(T2 * 0.7)], axis=1)
    err_db = 20 * np.log10(est / amps)
    assert np.max(np.abs(err_db)) < 1.0, f"round-trip per-harmonic error(dB)={np.round(err_db, 3)}"


def test_resynth_no_aliasing_above_nyquist():
    """Resynthesis must not produce foldback peaks below Nyquist (pitfall: aliasing check).

    Use a high f0 so some harmonics exceed Nyquist; those harmonics should be dropped and not fold back.
    Criterion: every significant peak in the FFT lies near some real harmonic k*f0.
    """
    f0 = 2500.0  # from harmonic 9 (>22500Hz) on, above Nyquist
    amps = np.ones(12)
    T = 200
    H = np.tile(amps[:, None], (1, T)).astype(np.float32)
    y = resynth(H, f0, sr=SR, hop=512)
    y = y[1000:-1000]

    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    spec_db = 20 * np.log10(spec / (spec.max() + 1e-12) + 1e-12)

    # real harmonic frequencies (only those < Nyquist)
    valid_harm = [(k + 1) * f0 for k in range(12) if (k + 1) * f0 < SR / 2]
    peak_bins = np.where(spec_db > -40)[0]  # significant peaks (above -40dB)
    for b in peak_bins:
        nearest = min(abs(freqs[b] - h) for h in valid_harm)
        assert nearest < 60.0, (
            f"{freqs[b]:.1f}Hz: inharmonic peak (nearest harmonic {nearest:.1f}Hz), suspected aliasing foldback"
        )


def test_resynth_output_finite_and_bounded():
    """Resynthesis output must be finite and bounded (no NaN/inf)."""
    f0 = 261.63
    amps = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
    y = _make_harmonic_tone(f0, amps)
    H, _, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))
    y_re = resynth(H, info["f0_per_frame"], sr=SR)
    assert np.all(np.isfinite(y_re))
    assert np.max(np.abs(y_re)) < 100.0
