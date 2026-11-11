"""f0 estimation and harmonic amplitude extraction.

* track_f0(y)         : PESTO primary + librosa.pyin cross-check, outputs (f0, conf, times)
* harmonics(y, f0)    : peak tracking of k<=60 harmonics on an STFT(4096/512) -> H[60, T]

Harmonic peak estimation uses parabolic interpolation (McAulay-Quatieri 1986): with f0 known, no full
partial tracking is needed; take the local peak within k*f0±search_cents and quadratically interpolate the log magnitude,
which corrects the Hann-window scalloping loss (amplitude error <1dB) and also measures the partial frequencies of quasi-harmonic sources.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    from quill_config import CFG
except Exception:  # pragma: no cover
    CFG = {
        "audio": {"sr": 44100},
        "f0": {
            "step_size_ms": 10.0,
            "pesto_model": "mir-1k_g7",
            "conf_threshold": 0.6,
            "octave_tol_cents": 50.0,
            "pyin_fmin": 55.0,
            "pyin_fmax": 2000.0,
        },
        "harmonic": {
            "n_harmonics": 60,
            "n_fft": 4096,
            "hop": 512,
            "search_cents": 30.0,
            "quasi_search_cents": 80.0,
            "low_f0_nfft_threshold": 100.0,
            "low_f0_nfft": 8192,
            "nyquist_margin_hz": 50.0,
        },
    }

_SR = int(CFG["audio"]["sr"])
_FCFG = CFG["f0"]
_HCFG = CFG["harmonic"]


# --------------------------------------------------------------------------- #
# f0 estimation
# --------------------------------------------------------------------------- #
def track_f0(
    y: np.ndarray,
    sr: int = _SR,
    step_size_ms: float | None = None,
    conf_threshold: float | None = None,
    cross_check: bool = True,
):
    """Estimate per-frame f0.

    Returns (f0, conf, times):
      f0    float32[T]  Hz, low-confidence frames set to NaN (octave errors often come with a confidence drop)
      conf  float32[T]  confidence 0-1
      times float32[T]  frame centres in seconds

    PESTO is primary (ISMIR'23, <30k parameters, real-time); with `cross_check=True`
    the median is compared against librosa.pyin and a warning is issued on octave errors (≈2x / 0.5x).
    Falls back to pyin when PESTO is unavailable.
    """
    step = float(step_size_ms if step_size_ms is not None else _FCFG["step_size_ms"])
    thr = float(conf_threshold if conf_threshold is not None else _FCFG["conf_threshold"])
    y = np.asarray(y, dtype=np.float32)

    f0, conf, times, backend = _pesto_f0(y, sr, step)
    if f0 is None:
        f0, conf, times = _pyin_f0(y, sr, step)
        warnings.warn("PESTO unavailable, fell back to librosa.pyin", RuntimeWarning, stacklevel=2)
    elif cross_check:
        _octave_cross_check(f0, conf, y, sr, thr)

    # confidence filter: f0 of frames below the threshold set to NaN, ignored downstream
    f0 = f0.astype(np.float32).copy()
    f0[conf < thr] = np.nan
    return f0, conf.astype(np.float32), times.astype(np.float32)


def _pesto_f0(y, sr, step):
    try:
        import pesto
        import torch
    except Exception:
        return None, None, None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts_ms, pitch_hz, conf, _act = pesto.predict(
                torch.from_numpy(y), sr, step_size=step, model_name=_FCFG["pesto_model"]
            )
        f0 = pitch_hz.detach().cpu().numpy().astype(np.float32)
        conf = conf.detach().cpu().numpy().astype(np.float32)
        times = ts_ms.detach().cpu().numpy().astype(np.float32) / 1000.0
        return f0, conf, times, "pesto"
    except Exception as exc:  # weight download failure / shape error -> let the caller handle it
        warnings.warn(f"PESTO inference error: {exc}", RuntimeWarning, stacklevel=2)
        return None, None, None, None


def _pyin_f0(y, sr, step):
    import librosa

    hop = max(1, int(round(step / 1000.0 * sr)))
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, sr=sr, fmin=float(_FCFG["pyin_fmin"]), fmax=float(_FCFG["pyin_fmax"]), hop_length=hop
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    conf = np.asarray(voiced_prob, dtype=np.float32)
    times = (np.arange(len(f0)) * hop / sr).astype(np.float32)
    return f0, conf, times


def _octave_cross_check(f0, conf, y, sr, thr):
    """Compare PESTO against the pyin median and warn on suspected octave errors (values are not changed)."""
    try:
        import librosa

        f0_py, _, vp = librosa.pyin(
            y, sr=sr, fmin=float(_FCFG["pyin_fmin"]), fmax=float(_FCFG["pyin_fmax"])
        )
    except Exception:
        return
    m_pesto = _voiced_median(f0, conf >= thr)
    m_pyin = _voiced_median(f0_py, np.asarray(vp) >= 0.5)
    if not (np.isfinite(m_pesto) and np.isfinite(m_pyin) and m_pyin > 0):
        return
    cents = 1200.0 * np.log2(m_pesto / m_pyin)
    if abs(cents) > float(_FCFG["octave_tol_cents"]):
        warnings.warn(
            f"f0 cross-check mismatch: PESTO≈{m_pesto:.1f}Hz vs pyin≈{m_pyin:.1f}Hz "
            f"({cents:+.0f}cents), suspected octave error",
            RuntimeWarning,
            stacklevel=2,
        )


def _voiced_median(f0, mask):
    vals = np.asarray(f0)[np.asarray(mask) & np.isfinite(np.asarray(f0))]
    vals = vals[vals > 0]
    return float(np.median(vals)) if vals.size else np.nan


def f0_scalar(f0: np.ndarray) -> float:
    """Median f0 of voiced frames, used as the TimbreSnapshot's f0_ref."""
    return _voiced_median(f0, np.ones_like(f0, dtype=bool))


# --------------------------------------------------------------------------- #
# Harmonic amplitude extraction
# --------------------------------------------------------------------------- #
def harmonics(
    y: np.ndarray,
    f0,
    sr: int = _SR,
    f0_times=None,
    n_harmonics: int | None = None,
    n_fft: int | None = None,
    hop: int | None = None,
    search_cents: float | None = None,
    partial_freqs=None,
    measure_partials: bool = False,
    min_prominence: float = 2.0,
):
    """Extract the amplitude envelope H[K, T] of the first `n_harmonics` harmonics per frame.

    Parameters
    ----
    f0 : scalar or array
        Scalar -> constant f0 throughout; array -> per-frame f0. With `f0_times` it is interpolated in time to
        STFT frame centres; otherwise resampled by frame index. NaN (unvoiced/low-confidence) frames are filled with the nearest voiced value.
    partial_freqs : float[K] or None
        Measured partial frequencies (@f0_ref) of a quasi-harmonic source (wine glass/bowl). If given, target frequency =
        partial_freqs * (f0/f0_ref), otherwise strictly k*f0.
    measure_partials : bool
        If True, widen the search window to quasi_search_cents and return the mean measured frequency of each harmonic
        (written to TimbreSnapshot.partial_freqs, so high overtones deviating from k*f0 are not lost).

    Returns
    ----
    H : float32[K, T]   harmonic amplitudes (true sinusoidal amplitude, corrected for the Hann-window coherent gain)
    times : float32[T]  frame centres in seconds
    info : dict         {"partial_freqs": float32[K] or None,
                         "f0_per_frame": float32[T], "n_fft": int}
    """
    import librosa

    y = np.asarray(y, dtype=np.float32)
    K = int(n_harmonics if n_harmonics is not None else _HCFG["n_harmonics"])
    hop = int(hop if hop is not None else _HCFG["hop"])
    cents = float(
        search_cents
        if search_cents is not None
        else (_HCFG["quasi_search_cents"] if measure_partials else _HCFG["search_cents"])
    )

    # per-frame f0 (used to decide whether to raise n_fft for low notes)
    f0_arr = np.atleast_1d(np.asarray(f0, dtype=np.float64))
    f0_med = _voiced_median(f0_arr, np.ones_like(f0_arr, dtype=bool))

    if n_fft is None:
        n_fft = int(_HCFG["n_fft"])
        if np.isfinite(f0_med) and f0_med < float(_HCFG["low_f0_nfft_threshold"]):
            n_fft = int(_HCFG["low_f0_nfft"])  # harmonics of low notes are too closely spaced, use a larger window
    n_fft = int(n_fft)

    win = np.hanning(n_fft).astype(np.float64)  # matches librosa's default 'hann'
    coherent_gain = win.sum()  # sinusoidal amplitude recovery constant: A = 2|X_peak|/cg

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(D).astype(np.float64)  # [n_fft/2+1, T]
    n_bins, T = mag.shape
    times = (np.arange(T) * hop / sr).astype(np.float32)
    bin_hz = sr / n_fft
    nyquist = sr / 2.0
    nyq_margin = float(_HCFG["nyquist_margin_hz"])

    f0_frame = _resample_f0(f0_arr, f0_times, times, sr, hop)

    # quasi-harmonic reference frequencies (ratio relative to f0_ref), strict k*f0 by default
    if partial_freqs is not None:
        pf = np.asarray(partial_freqs, dtype=np.float64)[:K]
        f0_ref = f0_med if np.isfinite(f0_med) and f0_med > 0 else f0_frame[0]
        ratio_k = pf / f0_ref  # target = ratio_k * f0_frame
    else:
        ratio_k = np.arange(1, K + 1, dtype=np.float64)  # strict harmonic k

    H = np.zeros((K, T), dtype=np.float32)
    measured = np.zeros((K, T), dtype=np.float64)
    measured_cnt = np.zeros(K, dtype=np.float64)
    fac = 2.0 ** (cents / 1200.0)  # half-width ratio of the search window

    for ti in range(T):
        f0_t = f0_frame[ti]
        if not (np.isfinite(f0_t) and f0_t > 0):
            continue
        col = mag[:, ti]
        for k in range(K):
            target = ratio_k[k] * f0_t
            if target >= nyquist - nyq_margin:  # stop once k*f0 reaches Nyquist
                break
            lo_bin = int(np.floor((target / fac) / bin_hz))
            hi_bin = int(np.ceil((target * fac) / bin_hz))
            lo_bin = max(1, lo_bin)
            hi_bin = min(n_bins - 2, hi_bin)
            if hi_bin < lo_bin:
                continue
            seg = col[lo_bin : hi_bin + 1]
            j = lo_bin + int(np.argmax(seg))
            # prominence gate: a true harmonic is a spike rising above the local noise floor (≥6dB); a "peak" picked from noise
            # is level with its neighbourhood. Without this check the noise floor of noisy recordings is harvested as high harmonics, which the synth
            # side replays as stable sinusoids, i.e. noise that the Air/Noise knobs cannot reach. Neighbourhood = side bands outside the main lobe
            # and inside the adjacent harmonic lobes, adapted to the harmonic spacing.
            spacing = f0_t / bin_hz
            o_lo = max(3, int(0.15 * spacing))
            o_hi = max(o_lo + 2, int(0.45 * spacing))
            a = max(1, j - o_hi)
            b = max(a + 1, j - o_lo + 1)
            c = min(n_bins - 1, j + o_lo)
            d = min(n_bins, j + o_hi + 1)
            nb = np.concatenate([col[a:b], col[c:d]])
            floor = float(np.median(nb)) if nb.size else 0.0
            if min_prominence > 0 and col[j] < min_prominence * floor:
                continue  # insufficient prominence -> treat as noise floor, skip this harmonic in this frame
            amp, true_bin = _parabolic_peak(col, j, coherent_gain)
            H[k, ti] = amp
            measured[k, ti] += true_bin * bin_hz
            measured_cnt[k] += 1.0

    info = {"f0_per_frame": f0_frame.astype(np.float32), "n_fft": n_fft, "partial_freqs": None}
    if measure_partials:
        with np.errstate(invalid="ignore", divide="ignore"):
            pf_out = np.where(
                measured_cnt > 0,
                measured.sum(axis=1) / np.maximum(measured_cnt, 1),
                np.arange(1, K + 1) * (f0_med if np.isfinite(f0_med) else 0.0),
            )
        info["partial_freqs"] = pf_out.astype(np.float32)
    return H, times, info


def _parabolic_peak(col: np.ndarray, j: int, coherent_gain: float):
    """Quadratic interpolation of the log magnitude at peak bin j, returns (sinusoidal amplitude, true bin position).

    Standard McAulay-Quatieri / Smith-Serra peak interpolation, corrects Hann-window scalloping loss.
    A = 2 * |X_peak| / coherent_gain (librosa.stft does not normalise the window).
    """
    eps = 1e-12
    if j <= 0 or j >= len(col) - 1:
        peak_lin = col[j]
        return float(2.0 * peak_lin / coherent_gain), float(j)
    a = 20.0 * np.log10(col[j - 1] + eps)
    b = 20.0 * np.log10(col[j] + eps)
    c = 20.0 * np.log10(col[j + 1] + eps)
    denom = a - 2.0 * b + c
    p = 0.5 * (a - c) / denom if abs(denom) > eps else 0.0
    p = float(np.clip(p, -0.5, 0.5))
    peak_db = b - 0.25 * (a - c) * p
    peak_lin = 10.0 ** (peak_db / 20.0)
    return float(2.0 * peak_lin / coherent_gain), float(j + p)


def _resample_f0(f0_arr, f0_times, frame_times, sr, hop):
    """Resample f0 to STFT frame centres; NaN filled with the nearest voiced value."""
    f0_arr = np.asarray(f0_arr, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if f0_arr.size == 1:
        return np.full(frame_times.shape, f0_arr[0], dtype=np.float64)

    filled = _fill_nan_nearest(f0_arr)
    if f0_times is not None:
        src_t = np.asarray(f0_times, dtype=np.float64)
    else:
        # assume f0 frames are spread uniformly over the signal duration
        total = frame_times[-1] if frame_times.size else (len(filled) - 1)
        src_t = np.linspace(0.0, total, num=len(filled))
    return np.interp(frame_times, src_t, filled)


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).copy()
    good = np.isfinite(a) & (a > 0)
    if not good.any():
        return np.full_like(a, np.nan)
    idx = np.arange(len(a))
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a
