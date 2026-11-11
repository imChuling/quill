"""Filtered noise synthesis: the noise side of a snapshot.

Synthesis side of the SMS stochastic model (Serra & Smith 1990): shapes the residual's 40 mel-band energy envelope `N_env[40, T]`
back into broadband noise. Depends only on numpy and synth.dsp (STFT/ISTFT/mel share one implementation with the analysis side,
so band definitions match). Rendered with a fixed seed, results are reproducible.
"""

from __future__ import annotations

import numpy as np

from synth.dsp import (
    NOISE_HOP,
    NOISE_NFFT,
    istft_np,
    mel_filterbank,
    stft_np,
)

_NOISE_NFFT = NOISE_NFFT
_NOISE_HOP = NOISE_HOP


def synth_noise(
    N_env: np.ndarray,
    sr: int,
    length: int,
    n_fft: int = _NOISE_NFFT,
    hop: int = _NOISE_HOP,
    centers_hz=None,
    seed: int = 0,
) -> np.ndarray:
    """Synthesise `length` samples of shaped noise from the mel-band RMS envelope `N_env[n_mels, T0]`.

    Analysis-modification-synthesis: take the STFT phase of white noise plus the target magnitude envelope, then ISTFT. Magnitudes come straight
    from N_env interpolated from mel centres to linear bins, so levels are correct with no calibration constant. A fixed seed keeps it reproducible.
    """
    N_env = np.asarray(N_env, dtype=np.float64)
    n_mels = N_env.shape[0]
    if centers_hz is None:
        _, centers_hz = mel_filterbank(sr, n_fft, n_mels)
    centers_hz = np.asarray(centers_hz, dtype=np.float64)
    bin_hz = np.arange(n_fft // 2 + 1) * (sr / n_fft)

    rng = np.random.default_rng(seed)
    w = rng.standard_normal(length + n_fft)
    Wc = stft_np(w, n_fft, hop)
    n_bins, Tn = Wc.shape

    # N_env time axis (T0) -> number of render frames (Tn)
    T0 = N_env.shape[1]
    if T0 == 1:
        N_t = np.repeat(N_env, Tn, axis=1)
    else:
        src = np.linspace(0.0, 1.0, T0)
        dst = np.linspace(0.0, 1.0, Tn)
        N_t = np.empty((n_mels, Tn))
        for b in range(n_mels):
            N_t[b] = np.interp(dst, src, N_env[b])

    # mel bands -> linear bin magnitudes (centre interpolation, a flat envelope stays flat)
    M = np.empty((n_bins, Tn))
    for t in range(Tn):
        M[:, t] = np.interp(bin_hz, centers_hz, N_t[:, t], left=N_t[0, t], right=N_t[-1, t])

    shaped = M * np.exp(1j * np.angle(Wc))
    y = istft_np(shaped, n_fft, hop, length=length)
    return y.astype(np.float32)
