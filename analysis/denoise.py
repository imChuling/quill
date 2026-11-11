"""Automatic denoising before timbre capture: spectral gating, pure numpy/scipy.

  1. noise profile = per-bin median of the quietest 15% of frames (a recording always has relatively quiet moments);
  2. soft mask = how far the magnitude exceeds the profile threshold (sigmoid soft transition in dB, avoids "bubbly" musical noise);
  3. the mask is smoothed in time and frequency, multiplied back onto the complex spectrum, and restored by iSTFT.
Cleaning before analyze_timbre benefits f0 tracking, harmonic extraction and the noise layer.
strength=0 passes through; 1=standard; 2=aggressive.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft


def spectral_gate(
    y: np.ndarray, sr: int, strength: float = 1.0, n_fft: int = 2048, hop: int = 512
) -> np.ndarray:
    """Spectral-gating denoise. Returns float32 of the same length as the input."""
    if strength <= 0 or len(y) < n_fft * 2:
        return np.asarray(y, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    f, t, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, padded=True)
    mag = np.abs(Z)  # [F, T]
    # pass through when there is no learnable noise floor: if the whole segment is continuous sound, the "quietest frames" are still signal and the threshold would suppress real harmonics.
    # criterion: if the level of the quietest 15% of frames is not clearly below a typical frame (<-10dB), there is no true noise floor;
    # noise in a steady sustained note is masked by the signal anyway, no denoising needed.
    frame_rms = np.sqrt((mag**2).mean(axis=0)) + 1e-12
    kq = max(2, int(len(frame_rms) * 0.15))
    floor_lvl = float(np.median(np.sort(frame_rms)[:kq]))
    if floor_lvl > 0.3 * float(np.median(frame_rms)):
        return np.asarray(y, dtype=np.float32)
    # noise profile: per-bin median of the quietest 15% of frames (robust; still finds a relative floor when the whole segment is voiced)
    k = max(2, int(mag.shape[1] * 0.15))
    quiet = np.sort(mag, axis=1)[:, :k]
    prof = np.median(quiet, axis=1, keepdims=True) + 1e-12
    thr = prof * (1.2 + 1.3 * strength)  # threshold rises with strength
    # soft mask: 0.5 at the threshold, smooth transition over ±soft (sigmoid in dB)
    ratio_db = 20.0 * np.log10(np.maximum(mag, 1e-12) / thr)
    mask = 1.0 / (1.0 + np.exp(-ratio_db / 3.0))
    # time-frequency smoothing (3x3 box) prevents isolated bins toggling = musical noise
    m = mask
    m = (m + np.roll(m, 1, 1) + np.roll(m, -1, 1)) / 3.0
    m = (m + np.roll(m, 1, 0) + np.roll(m, -1, 0)) / 3.0
    floor = 10.0 ** (-(12.0 + 12.0 * strength) / 20.0)  # residual floor (an all-zero mask sounds like a "vacuum")
    m = floor + (1.0 - floor) * m
    _, y_out = istft(Z * m, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    y_out = y_out[: len(y)]
    if len(y_out) < len(y):
        y_out = np.pad(y_out, (0, len(y) - len(y_out)))
    return y_out.astype(np.float32)
