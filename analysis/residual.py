"""Residual extraction, noise envelope, attack residual.

Analysis side of the SMS stochastic model (Serra & Smith 1990): res = y − harmonic model, then the residual is shaped into a
40 mel-band RMS envelope `N_env[40, T]` (fed to synth.noise).

Subtraction is done in the complex STFT domain: per frame, each harmonic's main-lobe magnitude is subtracted from |D| keeping the original phase, and ISTFT gives the residual.
Time-domain y−resynth cannot cancel because of phase mismatch; complex-domain subtraction both cancels harmonics and leaves the noise-floor phase at harmonic positions to the residual.
"""

from __future__ import annotations

import numpy as np

from synth.dsp import NOISE_HOP, NOISE_NFFT, mel_filterbank, stft_np

try:
    from quill_config import CFG

    _HCFG = CFG["harmonic"]
    _SR = int(CFG["audio"]["sr"])
except Exception:  # pragma: no cover
    _SR = 44100
    _HCFG = {"n_fft": 4096, "hop": 512, "nyquist_margin_hz": 50.0}

_ATTACK_MS = 60.0  # attack_res = 60ms@44.1k = 2646 samples
_LOBE_HALFWIDTH_BINS = 6  # half-width (bins) of the Hann main-lobe subtraction


def _window_lobe_lut(n_fft: int, halfwidth_bins: int, oversample: int = 64):
    """Precompute the DTFT main lobe |W(δ)| of the Hann window (δ in bins), DC normalised to the coherent gain Σwin.

    Returns (offsets[bin], wmag[bin]), used to sample the main-lobe shape for peaks at fractional frequencies.
    """
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    Npad = n_fft * oversample
    W = np.abs(np.fft.rfft(win, n=Npad))  # W[0] = Σwin = coherent gain
    # pad-bin j corresponds to an original bin offset j/oversample
    max_j = halfwidth_bins * oversample
    offsets = np.arange(max_j + 1) / oversample
    return offsets, W[: max_j + 1]


def extract_residual(
    y,
    H,
    f0_frame,
    sr: int = _SR,
    n_fft: int | None = None,
    hop: int | None = None,
    partial_freqs=None,
    f0_ref: float | None = None,
    lobe_halfwidth_bins: int = _LOBE_HALFWIDTH_BINS,
):
    """Phase-aligned subtraction gives the residual res (time domain, same length as y).

    Parameters match harmonic.harmonics: H[K, T] are harmonic amplitudes (true sinusoidal amplitude),
    f0_frame[T] is the per-STFT-frame f0; if partial_freqs is given, main lobes are located at the quasi-harmonic frequencies.
    """
    import librosa

    y = np.asarray(y, dtype=np.float64)
    n_fft = int(n_fft if n_fft is not None else _HCFG["n_fft"])
    hop = int(hop if hop is not None else _HCFG["hop"])
    H = np.asarray(H, dtype=np.float64)
    K, T = H.shape

    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    cg = win.sum()
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(D)
    phase = np.angle(D)
    n_bins, Td = mag.shape
    T = min(T, Td)
    bin_hz = sr / n_fft
    nyquist = sr / 2.0
    nyq_margin = float(_HCFG.get("nyquist_margin_hz", 50.0))

    offs, wlut = _window_lobe_lut(n_fft, lobe_halfwidth_bins)

    f0_frame = np.asarray(f0_frame, dtype=np.float64)
    if partial_freqs is not None:
        pf = np.asarray(partial_freqs, dtype=np.float64)[:K]
        ref = float(f0_ref) if f0_ref else float(np.nanmedian(f0_frame))
        ratio_k = pf / ref
    else:
        ratio_k = np.arange(1, K + 1, dtype=np.float64)

    harm_mag = np.zeros_like(mag)
    for t in range(T):
        f0_t = f0_frame[t] if t < len(f0_frame) else f0_frame[-1]
        if not (np.isfinite(f0_t) and f0_t > 0):
            continue
        for k in range(K):
            a = H[k, t]
            if a <= 0:
                continue
            target = ratio_k[k] * f0_t
            if target >= nyquist - nyq_margin:
                break
            p = target / bin_hz  # fractional bin
            peak = 0.5 * a * cg  # |D| peak (A=2peak/cg)
            lo = int(np.floor(p - lobe_halfwidth_bins))
            hi = int(np.ceil(p + lobe_halfwidth_bins))
            for m in range(max(0, lo), min(n_bins, hi + 1)):
                d = abs(m - p)
                if d <= offs[-1]:
                    harm_mag[m, t] += peak * (np.interp(d, offs, wlut) / cg)

    res_mag = np.maximum(mag - harm_mag, 0.0)
    R = res_mag * np.exp(1j * phase)
    res = librosa.istft(R, hop_length=hop, n_fft=n_fft, window="hann", length=len(y))
    return res.astype(np.float32)


def noise_envelope(
    res,
    sr: int = _SR,
    n_mels: int = 40,
    n_fft: int = NOISE_NFFT,
    hop: int = NOISE_HOP,
    guard_samples: int = 4096,
    f0_ref: float | None = None,
    notch_rel: float = 0.06,
    notch_harmonics: int = 6,
    hp_hz: float | None = None,
):
    """Residual -> mel-band RMS envelope N_env[n_mels, T] (same framework as synth.noise).

    Boundary artefacts: librosa's center=True istft leaves large transients within about one harmonic window length at each end
    (the harmonic model does not match at signal start/end), and these boundary frames would dominate N_env by ~100×.
    Countermeasures: (1) estimate the envelope only in the steady inner region excluding the start/end guard; (2) robust per-band cap at median+k·MAD.

    When f0_ref is given, two more cleanups (leakage from f0 jitter/amplitude fluctuation not absorbed by the steady harmonics piles up near f0,
    and played at another pitch it becomes a lump of fixed-frequency noise):
    (3) bins within ±notch_rel of the first notch_harmonics harmonics are excluded from the noise (higher harmonics are not removed,
    breath/friction noise lives mostly there); (4) frequencies below hp_hz are excluded (low-frequency rumble in recordings).
    """
    res = np.asarray(res, dtype=np.float64)
    g = int(min(guard_samples, max(0, (len(res) - n_fft) // 2)))
    interior = res[g : len(res) - g] if (len(res) - 2 * g) >= n_fft else res

    R = np.abs(stft_np(interior, n_fft, hop))
    power = R**2
    freqs = np.arange(power.shape[0]) * (sr / n_fft)
    keep = np.ones(power.shape[0], dtype=bool)
    if hp_hz:
        keep &= freqs >= float(hp_hz)
    if f0_ref and f0_ref > 0:
        for k in range(1, int(notch_harmonics) + 1):
            fc = k * float(f0_ref)
            # leakage is widest near f0 and the 2nd harmonic: remove ±25% for the first two, ±notch_rel for the rest
            rel = max(notch_rel, 0.25) if k <= 2 else notch_rel
            keep &= np.abs(freqs - fc) > rel * fc
    power = power * keep[:, None]
    fb, centers = mel_filterbank(sr, n_fft, n_mels)
    denom = fb.sum(axis=1, keepdims=True)
    denom = np.where(denom > 1e-9, denom, 1.0)
    band_power = (fb @ power) / denom  # [n_mels, T] weighted mean power
    N_env = np.sqrt(np.maximum(band_power, 0.0))

    # robust cap: single-frame transients must not dominate (median + 8·MAD)
    med = np.median(N_env, axis=1, keepdims=True)
    mad = np.median(np.abs(N_env - med), axis=1, keepdims=True)
    cap = med + 8.0 * 1.4826 * mad
    N_env = np.minimum(N_env, cap)
    return N_env.astype(np.float32), centers


def attack_residual(res, sr: int = _SR, ms: float = _ATTACK_MS, onset_thresh: float = 0.05):
    """Cut a 60ms attack residual (float32[2646]@44.1k).

    Take `ms` milliseconds starting from the first sample exceeding onset_thresh*peak, zero-padded on the right if too short.
    """
    res = np.asarray(res, dtype=np.float32)
    n = int(round(ms / 1000.0 * sr))
    if res.size == 0:
        return np.zeros(n, dtype=np.float32)
    peak = float(np.max(np.abs(res)))
    if peak > 0:
        above = np.where(np.abs(res) >= onset_thresh * peak)[0]
        start = int(above[0]) if above.size else 0
    else:
        start = 0
    seg = res[start : start + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    seg = seg.astype(np.float32).copy()
    # 2ms raised-cosine fade in/out at the grain ends, avoiding block-boundary clicks on playback
    fade = max(1, int(0.002 * sr))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade)))
    seg[:fade] *= ramp
    seg[-fade:] *= ramp[::-1]
    return seg
