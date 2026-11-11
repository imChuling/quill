"""Tests for the SNT residual re-decomposition (analysis/snt.py)."""

import numpy as np
import pytest

from analysis.snt import (
    energy_split,
    spectral_flatness,
    temporal_concentration,
    transient_stochastic_split,
)

SR = 44100


def _click_plus_noise(sr=SR, dur=1.0):
    """Synthetic: a broadband transient click at t=0.3s + a low-level white noise floor throughout."""
    n = int(sr * dur)
    rng = np.random.default_rng(0)
    x = 0.02 * rng.standard_normal(n).astype(np.float32)  # steady noise floor
    c = int(0.3 * sr)
    x[c : c + 64] += np.hanning(64).astype(np.float32) * 0.8  # transient burst
    return x


def test_flatness_orders_noise_above_tone():
    sr = SR
    t = np.arange(sr) / sr
    rng = np.random.default_rng(1)
    white = rng.standard_normal(sr).astype(np.float32)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    assert spectral_flatness(white, sr) > 0.3  # white noise is flat
    assert spectral_flatness(sine, sr) < 0.01  # a pure tone is spiky
    assert spectral_flatness(white, sr) > spectral_flatness(sine, sr)


def test_split_separates_transient_from_floor():
    x = _click_plus_noise()
    trans, stoch = transient_stochastic_split(x, SR)
    # the transient is more concentrated in time, the stochastic floor more diffuse
    assert temporal_concentration(trans, SR) < temporal_concentration(stoch, SR)
    # transient energy near the click should be higher than elsewhere
    c = int(0.3 * SR)
    near = np.sum(trans[c - 256 : c + 256] ** 2)
    far = np.sum(trans[:512] ** 2)
    assert near > far


def test_energy_split_sums_sane():
    x = _click_plus_noise()
    trans, stoch = transient_stochastic_split(x, SR)
    sp = energy_split(x, trans, stoch)
    assert 0.0 <= sp["transient"] <= 1.5
    assert 0.0 <= sp["stochastic"] <= 1.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
