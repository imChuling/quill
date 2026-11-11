"""SNT residual decomposition: split the SMS residual further into transient + stochastic layers.

SMS (Serra-Smith 1990) only splits the signal into harmonic + stochastic; the broadband attack transient (the "click" of a pick/lips/bow hair)
is neither harmonic nor stationary, and forcing it into the stochastic layer pollutes the noise floor (heard as a "hum"). The residual after harmonic
subtraction is separated with median filtering (Fitzgerald 2010 / Driedger HPSS):
  - median along time -> time-stationary component = stochastic noise floor
  - median along frequency -> broadband time-local component = transient
giving three layers S (sinusoidal) + N (stochastic) + T (transient). This module does not change the default synthesis path: it provides the separation tool
and lets eval/residual_quality.py quantify "the residual really is noise-like".
"""

from __future__ import annotations

import numpy as np

try:
    from quill_config import CFG

    _SR = int(CFG["audio"]["sr"])
except Exception:  # pragma: no cover
    _SR = 44100

_NFFT = 1024
_HOP = 256


def transient_stochastic_split(
    res, sr: int = _SR, n_fft: int = _NFFT, hop: int = _HOP, margin: float = 2.0
):
    """Residual -> (transient, stochastic), time domain, same length as res.

    margin>1 gives a "cleaner" (sparser) separation: a soft gap between the two masks, so energy is not counted twice.
    """
    import librosa

    res = np.asarray(res, dtype=np.float32)
    if res.size < n_fft:
        res = np.pad(res, (0, n_fft - res.size))
    D = librosa.stft(res, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    # HPSS: harmonic=time-stationary=stochastic floor; percussive=broadband transient
    D_stoch, D_trans = librosa.decompose.hpss(D, margin=(margin, margin))
    stoch = librosa.istft(D_stoch, hop_length=hop, n_fft=n_fft, length=len(res))
    trans = librosa.istft(D_trans, hop_length=hop, n_fft=n_fft, length=len(res))
    return trans.astype(np.float32), stoch.astype(np.float32)


def spectral_flatness(
    x, sr: int = _SR, n_fft: int = 2048, fmin: float = 200.0, fmax: float = 8000.0
):
    """Wiener entropy / spectral flatness ∈ (0,1]: 1=white noise (flat), ->0=pure tone (spiky).

    The noise floor should be highly flat; harmonic components should be low. Computed only within the signal band [fmin,fmax] (empty out-of-band bins
    would collapse the geometric mean); the floor is a relative lower bound of each frame's peak (-80dB), so empty bins do not dominate.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (f >= fmin) & (f <= fmax)
    win = np.hanning(n_fft)
    flats = []
    for i in range(0, len(x) - n_fft, n_fft // 2):
        p = np.abs(np.fft.rfft(x[i : i + n_fft] * win)) ** 2
        p = p[band]
        pk = p.max()
        if pk <= 0:
            continue
        p = np.maximum(p, pk * 1e-8)  # relative floor (-80dB), stabilises the geometric mean
        gm = np.exp(np.mean(np.log(p)))
        am = np.mean(p)
        flats.append(gm / am)
    return float(np.mean(flats)) if flats else 0.0


def temporal_concentration(x, sr: int = _SR, frac: float = 0.9):
    """Temporal concentration: the fraction of the duration holding the top `frac` of envelope energy (smaller is more "transient/local")."""
    x = np.asarray(x, dtype=np.float64)
    e = x**2
    tot = e.sum()
    if tot <= 0:
        return 1.0
    order = np.sort(e)[::-1]  # energy from large to small
    cum = np.cumsum(order) / tot
    n_needed = int(np.searchsorted(cum, frac)) + 1
    return n_needed / len(e)


def energy_split(y, transient, stochastic, harmonic_res=None):
    """Return a dict of per-layer energy shares (relative to the energy of input y)."""
    ey = float(np.sum(np.asarray(y, np.float64) ** 2)) + 1e-12
    et = float(np.sum(np.asarray(transient, np.float64) ** 2))
    es = float(np.sum(np.asarray(stochastic, np.float64) ** 2))
    out = {"transient": et / ey, "stochastic": es / ey}
    if harmonic_res is not None:
        out["residual"] = (et + es) / ey
        out["harmonic"] = 1.0 - (et + es) / ey
    return out
