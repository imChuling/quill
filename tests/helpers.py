"""Shared Quill test fixtures."""

import numpy as np
import pytest

SR = 44100

ADSR_FAST = dict(attack_ms=10, decay_ms=20, sustain=0.7, release_ms=30)
VEL_CFG = dict(loud_exp=1.0, tilt_db_oct=6.0)
RT_CFG = dict(attack_fade_ms=5)


@pytest.fixture
def sr():
    return SR


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_test_instrument(K=8, f0_ref=220.0, noise_level=0.0, sr=SR):
    """Lightweight Instrument for voice/oscillator/engine unit tests."""
    from synth.instrument import Instrument

    H = np.exp(-np.arange(K, dtype=np.float64) * 0.3)
    ratio = np.arange(1, K + 1, dtype=np.float64)
    grain = np.zeros(512, dtype=np.float64)
    noise = np.zeros(sr, dtype=np.float64)
    return Instrument(H, ratio, grain, noise, f0_ref, sr, noise_level=noise_level)


def make_test_snapshot(K=60, M=40, f0=220.0, T=200, decay=False, sr=SR):
    """Lightweight snapshot dict for instrument/engine/library tests."""
    from snapshot import make_snapshot
    from synth.dsp import NOISE_NFFT, mel_filterbank

    k = np.arange(1, K + 1)
    shape = np.exp(-(((k * f0 - 700) / 600) ** 2)) + 0.05
    contour = np.exp(-np.linspace(0, 5, T)) if decay else np.ones(T)
    H = np.outer(shape, contour).astype(np.float32)
    _, centers = mel_filterbank(sr, NOISE_NFFT, M)
    return make_snapshot(
        f0,
        H,
        0.01 * np.ones((M, T), np.float32),
        np.zeros(2646, np.float32),
        noise_centers_hz=centers,
        sr=sr,
        harm_hop=512,
    )
