"""Additive resynthesis and formant-preserving pitch redirection (offline version).

resynth(H, f0) -> y: phase-accumulating oscillator bank; per-frame H and f0 are linearly interpolated to the sample rate,
each harmonic is band-limited by construction (harmonics above Nyquist are zeroed, no BLEP or oversampling needed).
redirect_harmonics resamples the harmonic envelope by absolute frequency to the target f0; the real-time voice (synth.voice) uses the same math. Depends only on numpy.
"""

from __future__ import annotations

import numpy as np

_SR_DEFAULT = 44100
_HOP_DEFAULT = 512


def resynth(
    H: np.ndarray,
    f0,
    sr: int = _SR_DEFAULT,
    hop: int = _HOP_DEFAULT,
    partial_freqs=None,
    f0_ref: float | None = None,
    length: int | None = None,
) -> np.ndarray:
    """Reconstruct a waveform from the harmonic amplitude envelope H[K, T] and f0.

    Parameters
    ----
    H : float32[K, T]   true sinusoidal amplitude per harmonic per frame (output of harmonic.harmonics)
    f0 : scalar or float[T]
        Target f0. Scalar -> constant; array (length T) -> per frame, linearly interpolated to the sample rate.
    partial_freqs, f0_ref :
        Quasi-harmonic source: frequency of component k = partial_freqs[k] * (f0 / f0_ref),
        otherwise strictly k * f0.
    length : int or None
        Number of output samples; defaults to T*hop.

    Returns float32[N], phase-continuous, no clicks at frame boundaries.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"H must be 2-D [K, T], got {H.shape}")
    K, T = H.shape
    N = int(length) if length is not None else T * hop

    # --- frame-level f0 -> sample-level f0 (linear interpolation) ---
    f0_arr = np.atleast_1d(np.asarray(f0, dtype=np.float64))
    if f0_arr.size == 1:
        f0_samp = np.full(N, f0_arr[0], dtype=np.float64)
    else:
        f0_samp = _frames_to_samples(f0_arr, T, N, hop)

    # --- frame-level H -> sample-level amplitude (linear interpolation per harmonic) ---
    # frequency ratio per harmonic (strict harmonic k, or measured quasi-harmonic ratio)
    if partial_freqs is not None:
        pf = np.asarray(partial_freqs, dtype=np.float64)[:K]
        ref = float(f0_ref) if f0_ref else 1.0
        ratio_k = pf / ref if ref else np.arange(1, K + 1, dtype=np.float64)
    else:
        ratio_k = np.arange(1, K + 1, dtype=np.float64)

    nyquist = sr / 2.0
    y = np.zeros(N, dtype=np.float64)

    for k in range(K):
        amp_frames = H[k]
        if not np.any(amp_frames):
            continue
        amp_samp = _frames_to_samples(amp_frames, T, N, hop)
        # instantaneous frequency (per sample)
        freq_samp = ratio_k[k] * f0_samp
        # band limit: zero the amplitude of samples above Nyquist (the real-time kernel uses a soft gain ramp instead)
        amp_samp = np.where(freq_samp < nyquist, amp_samp, 0.0)
        if not np.any(amp_samp):
            continue
        # phase accumulation (continuous phase, no jumps across frames)
        phase = np.cumsum(2.0 * np.pi * freq_samp / sr)
        y += amp_samp * np.cos(phase)

    return y.astype(np.float32)


def _frames_to_samples(frame_vals: np.ndarray, T: int, N: int, hop: int) -> np.ndarray:
    """Linearly interpolate / upsample a frame-level sequence to N sample points.

    The centre of frame i maps to sample position i*hop (matches the librosa center=True frame grid to within a hop);
    absolute phase continuity is guaranteed by accumulating each harmonic independently.
    """
    frame_vals = np.asarray(frame_vals, dtype=np.float64)
    if T == 1:
        return np.full(N, frame_vals[0], dtype=np.float64)
    frame_pos = np.arange(T) * hop
    samp_idx = np.arange(N)
    return np.interp(samp_idx, frame_pos, frame_vals, left=frame_vals[0], right=frame_vals[-1])


# ---- Formant-preserving pitch redirection ----
def redirect_harmonics(
    H: np.ndarray,
    f0_ref: float,
    f0_tgt: float,
    partial_freqs=None,
    mode: str = "formant",
    rolloff_db_oct: float = 12.0,
):
    """Redirect the harmonic envelope H[K,T] at reference f0 f0_ref to target f0 f0_tgt.

    mode="formant": treat H[k] as samples of the spectral envelope at absolute frequency k·f0_ref; the amplitude
        of target component k' = envelope interpolated at that absolute frequency (simplified True Envelope, Röbel & Rodet 2005).
        Formant positions stay fixed on the frequency axis; no chipmunk effect within ±12 semitones.
    mode="index": control variant, H_tgt[k']=H[k'] carried over by harmonic index; formants shift with f0.

    Returns (H_tgt[K,T], partial_freqs_tgt or None):
      * strictly harmonic (partial_freqs=None): H_tgt lies at k'·f0_tgt, returns (H_tgt, None)
      * quasi-harmonic: returns (H_tgt, partial_freqs); frequencies are still driven by resynth as pf*(f0_tgt/f0_ref)

    When the target frequency exceeds the envelope support (above the highest reference component), extrapolate with a
    rolloff_db_oct dB/octave decay instead of np.interp's default flat clamping.
    """
    H = np.asarray(H, dtype=np.float64)
    K, T = H.shape

    if partial_freqs is not None:
        xp = np.asarray(partial_freqs, dtype=np.float64)[:K]  # absolute reference frequencies
        targets = xp * (f0_tgt / f0_ref)  # target component frequencies
        pf_tgt = xp
    else:
        xp = np.arange(1, K + 1, dtype=np.float64) * f0_ref
        targets = np.arange(1, K + 1, dtype=np.float64) * f0_tgt
        pf_tgt = None

    if mode == "index":
        # carry over by index: amplitudes unchanged, only the frequency axis scales; resynth just uses f0_tgt
        return H.astype(np.float32), pf_tgt

    if mode != "formant":
        raise ValueError(f"unknown mode={mode!r} (expected 'formant' or 'index')")

    H_tgt = np.empty_like(H)
    for t in range(T):
        H_tgt[:, t] = _sample_envelope(targets, xp, H[:, t], rolloff_db_oct)
    return H_tgt.astype(np.float32), pf_tgt


def _sample_envelope(freqs_q, xp, fp, rolloff_db_oct):
    """Sample the spectral envelope (xp, fp) at absolute frequencies, extrapolating outside support with a dB/octave decay."""
    freqs_q = np.asarray(freqs_q, dtype=np.float64)
    out = np.interp(freqs_q, xp, fp, left=fp[0], right=fp[-1])
    # high-frequency extrapolation: query points above the highest reference component decay the last value by rolloff
    hi = xp[-1]
    above = freqs_q > hi
    if np.any(above) and fp[-1] > 0 and hi > 0:
        octaves = np.log2(np.maximum(freqs_q[above], hi) / hi)
        out[above] = fp[-1] * 10.0 ** (-rolloff_db_oct * octaves / 20.0)
    return out
