"""Shared DSP primitives: numpy STFT/ISTFT and mel filter bank.

analysis/ and synth/ share one implementation so band definitions stay identical and cross-layer imports are avoided. Depends only on numpy.
"""

from __future__ import annotations

import numpy as np

NOISE_NFFT = 1024
NOISE_HOP = 256
MEL_FMIN = 40.0


def _hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float64)


def stft_np(x: np.ndarray, n_fft: int = NOISE_NFFT, hop: int = NOISE_HOP) -> np.ndarray:
    """Real signal -> complex STFT [n_bins, T] (Hann window, no center padding)."""
    x = np.asarray(x, dtype=np.float64)
    w = _hann(n_fft)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for t in range(n_frames):
        seg = x[t * hop : t * hop + n_fft] * w
        out[:, t] = np.fft.rfft(seg)
    return out


def istft_np(
    spec: np.ndarray, n_fft: int = NOISE_NFFT, hop: int = NOISE_HOP, length: int | None = None
) -> np.ndarray:
    """Complex STFT -> real signal (WOLA, normalised so that istft(stft(x))==x)."""
    w = _hann(n_fft)
    n_frames = spec.shape[1]
    total = (n_frames - 1) * hop + n_fft
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    w2 = w * w
    for t in range(n_frames):
        fr = np.fft.irfft(spec[:, t], n=n_fft)
        pos = t * hop
        out[pos : pos + n_fft] += fr * w
        norm[pos : pos + n_fft] += w2
    norm = np.where(norm > 1e-9, norm, 1.0)
    out = out / norm
    if length is not None:
        out = out[:length] if length <= len(out) else np.pad(out, (0, length - len(out)))
    return out.astype(np.float64)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int, n_fft: int, n_mels: int = 40, fmin: float = MEL_FMIN, fmax: float | None = None
):
    """Return (fb[n_mels, n_bins], centers_hz[n_mels]), triangular filters."""
    fmax = float(fmax if fmax is not None else sr / 2.0)
    n_bins = n_fft // 2 + 1
    bin_hz = np.arange(n_bins) * (sr / n_fft)
    m_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    f_edges = mel_to_hz(m_edges)
    centers = f_edges[1:-1]
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for b in range(n_mels):
        lo, ce, hi = f_edges[b], f_edges[b + 1], f_edges[b + 2]
        left = (bin_hz - lo) / max(ce - lo, 1e-9)
        right = (hi - bin_hz) / max(hi - ce, 1e-9)
        tri = np.maximum(0.0, np.minimum(left, right))
        fb[b] = tri
    return fb, centers.astype(np.float64)
