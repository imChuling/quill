"""True Envelope spectral envelope estimation (Röbel & Rodet 2005).

Iterative cepstral true envelope: repeatedly "take the pointwise max of the spectrum and the current envelope -> cepstral lowpass smoothing" until the
envelope lies above the spectral peaks everywhere, giving a continuous, smooth spectral envelope through the harmonic peaks (formant structure).

Not used in the synthesis path: for cross-pitch fidelity, the smoothed envelope loses the harmonic-level detail needed to match the real spectrum,
so linear interpolation of the measured harmonic amplitudes works better. Kept as a tool for spectral-envelope visualisation / the noise side. Depends only on numpy.
"""

from __future__ import annotations

import numpy as np


def true_envelope(logmag, order, n_iter=200, thresh_db=2.0):
    """Estimate the True Envelope (log envelope of the same length) of the log magnitude spectrum `logmag` (rfft bins).

    order: cepstral order (number of low-quefrency coefficients kept; smaller is smoother). Empirically order≈sr/(2*f0),
           chosen slightly below the quefrency of the f0 period so the envelope smooths over harmonics but keeps formants.
    """
    logmag = np.asarray(logmag, dtype=np.float64)
    B = len(logmag)
    A = logmag.copy()
    V = np.full(B, -60.0)  # initial envelope (dB-scale floor)
    thresh = thresh_db / (20.0 / np.log(10.0))  # dB threshold -> natural-log threshold
    for _ in range(n_iter):
        A = np.maximum(A, V)  # the envelope must lie above the spectrum
        full = np.concatenate([A, A[-2:0:-1]])  # symmetric extension to the full DFT, length M=2(B-1)
        cep = np.fft.ifft(full).real  # real cepstrum (even symmetric)
        M = len(cep)
        lifter = np.zeros(M)  # lowpass lifter: keep low quefrencies (both ends due to symmetry)
        lifter[: order + 1] = 1.0
        if order > 0:
            lifter[M - order :] = 1.0
        Vfull = np.fft.fft(cep * lifter).real
        V = Vfull[:B]
        if np.max(A - V) < thresh:  # converged: the envelope covers the spectrum everywhere
            break
    return V


def envelope_from_audio(y, sr, f0_ref, n_fft=4096, order=None, fmax=None):
    """Estimate the True Envelope from steady audio, return (freqs, log_env) (for sampling by absolute frequency)."""
    y = np.asarray(y, dtype=np.float64)
    seg = y[len(y) // 5 : -len(y) // 5] if len(y) > 10 else y
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    # steady-state mean magnitude spectrum
    S = []
    for i in range(0, len(seg) - n_fft, n_fft // 2):
        S.append(np.abs(np.fft.rfft(seg[i : i + n_fft] * np.hanning(n_fft))))
    mag = np.mean(S, axis=0) if S else np.abs(np.fft.rfft(seg[:n_fft] * np.hanning(n_fft)))
    logmag = np.log(mag + 1e-9)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    if order is None:
        # slightly below the quefrency of the f0 period, so the envelope smooths over harmonics and keeps formants
        order = max(8, int(0.7 * sr / max(f0_ref, 1.0)))
    env = true_envelope(logmag, order=order)
    return freqs, env


def sample_envelope_db(freqs, log_env, query_hz, rolloff_db_oct=12.0):
    """Sample (freqs, log_env) at absolute frequencies -> linear amplitude; extrapolate outside support with rolloff decay."""
    query_hz = np.asarray(query_hz, dtype=np.float64)
    v = np.interp(query_hz, freqs, log_env, left=log_env[0], right=log_env[-1])
    hi = freqs[-1]
    above = query_hz > hi
    if np.any(above) and hi > 0:
        octv = np.log2(np.maximum(query_hz[above], hi) / hi)
        v[above] = log_env[-1] - (rolloff_db_oct * octv) * (np.log(10.0) / 20.0)
    return np.exp(v)
