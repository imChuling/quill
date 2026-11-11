"""Timbre analysis: audio -> TimbreSnapshot.

Chains track_f0 / harmonics / residual into one pipeline that extracts a TimbreSnapshot from an isolated note (or a stable
segment picked automatically from a natural recording). librosa is used only on the analysis side; the output is pure data.
"""

from __future__ import annotations

import numpy as np

from analysis.harmonic import f0_scalar, harmonics, track_f0
from analysis.residual import attack_residual, extract_residual, noise_envelope
from snapshot import make_snapshot

try:
    from quill_config import CFG

    _SR = int(CFG["audio"]["sr"])
    _NH = int(CFG["harmonic"]["n_harmonics"])
    _HOP = int(CFG["harmonic"]["hop"])
except Exception:  # pragma: no cover
    _SR, _NH, _HOP = 44100, 60, 512


def detect_vibrato(f0, times, fmin=4.0, fmax=8.0, min_depth_cents=8.0, prominence=4.0):
    """Detect vibrato from the f0 track -> {"rate_hz","depth_cents"} | None.

    Convert f0 to cents deviation, detrend, windowed FFT, look for a significant peak in 4-8Hz. Periodic vibrato is extracted first,
    then the caller judges slide monotonicity.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    good = np.isfinite(f0) & (f0 > 0)
    if good.sum() < 16 or times.size < 2:
        return None
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return None
    med = np.median(f0[good])
    idx = np.arange(len(f0))
    cents = np.interp(idx, idx[good], 1200.0 * np.log2(f0[good] / med))
    # linear detrend: remove the linear component of a monotonic slide, otherwise a slide's cents ramp leaves low-frequency energy
    # that is misjudged as vibrato. Vibrato is the oscillation that remains after detrending.
    coef = np.polyfit(idx, cents, 1)
    cents = cents - (coef[0] * idx + coef[1])

    n = len(cents)
    w = np.hanning(n)
    C = np.abs(np.fft.rfft(cents * w))
    freqs = np.fft.rfftfreq(n, dt)
    band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if band.size == 0:
        return None
    pk = band[int(np.argmax(C[band]))]
    amp_cents = 2.0 * C[pk] / np.sum(w)  # one-sided amplitude (peak deviation)
    bg = np.median(C[(freqs > 1.0) & (freqs < 20.0)]) + 1e-9
    if amp_cents < min_depth_cents or C[pk] < prominence * bg:
        return None
    return {"rate_hz": float(freqs[pk]), "depth_cents": float(amp_cents)}


def _snap_to_hint(f0, hint, max_dev_cents: float = 50.0):
    """Fix octave/harmonic errors in the f0 track using a known f0 hint (factor∈{1/3,1/2,1,2,3}).

    Used when dataset notes come with a ground-truth pitch: for low instruments (bass/tuba) PESTO easily locks onto the 2nd/3rd harmonic, and the whole
    snapshot then builds the wrong harmonics; pick the nearest simple ratio from median(f0)/hint and correct globally, keeping the relative contour (vibrato).
    For very low notes (<40Hz) PESTO fails outright and errors are not simple ratios: if still off by >max_dev_cents after correction, fall back to
    a constant hint (guarantees correct harmonic placement at the cost of vibrato; deep bass has almost no vibrato anyway).
    """
    f0 = np.asarray(f0, dtype=np.float64).copy()
    if hint <= 0:
        return f0
    good = np.isfinite(f0) & (f0 > 0)
    if good.sum() == 0:
        return np.full_like(f0, hint)  # no valid f0 at all -> constant hint
    med = float(np.median(f0[good]))
    if med > 0:
        cands = np.array([1 / 3, 1 / 2, 1.0, 2.0, 3.0])
        best = cands[int(np.argmin(np.abs(np.log((med / hint) / cands))))]
        if abs(np.log(best)) > 1e-6:
            f0[good] = f0[good] / best
    medc = float(np.median(f0[good])) if good.any() else 0.0
    if medc <= 0 or abs(1200.0 * np.log2(max(medc, 1e-6) / hint)) > max_dev_cents:
        f0[:] = hint  # very low note fallback: constant ground truth
    return f0


def select_stable_window(y, sr, f0, f0_times, min_s=0.30, max_s=2.0, stable_cents=150.0):
    """Automatically pick the stable pitch window in a recording best suited for timbre capture.

    In a natural recording (singing a line / humming) the pitch moves with the melody/words, and timbre capture only needs one sustained note
    (some held vowel). Find the longest contiguous region that is "pitched + locally pitch-stable (vibrato allowed) + has energy",
    return its sample range (a, b); if none is found (pure noise / fast speech / no sustained note) return None and let the gate reject it.

    stable_cents=150: the analysis side tracks f0 per frame, so expressive singing (vibrato/inflection) can still be analysed;
    tightening to 70 cents would discard segments with character and leave only rigid humming.
    """
    T = len(f0)
    if T < 4 or len(f0_times) != T:
        return None
    voiced = np.isfinite(f0) & (f0 > 0)
    if voiced.sum() < 4:
        return None
    med = float(np.median(f0[voiced]))
    cents = np.where(voiced, 1200.0 * np.log2(np.where(voiced, f0, med) / med), np.nan)
    dt = float(np.median(np.diff(f0_times))) if T > 1 else 0.01
    half = max(1, int(0.12 / max(dt, 1e-4)))  # local median window ±120ms (tolerates vibrato)
    win = max(1, int(dt * sr))
    dev = np.full(T, np.inf)
    energy = np.zeros(T)
    for i in range(T):
        a, b = max(0, i - half), min(T, i + half + 1)
        seg = cents[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size and np.isfinite(cents[i]):
            dev[i] = abs(cents[i] - np.median(seg))
        c = int(f0_times[i] * sr)
        e0, e1 = max(0, c - win), min(len(y), c + win)
        if e1 > e0:
            energy[i] = float(np.sqrt(np.mean(np.asarray(y[e0:e1], np.float64) ** 2)))
    emax = float(energy.max()) + 1e-12
    good = voiced & (dev < stable_cents) & (energy > 0.15 * emax)

    # candidate score = duration^0.3 × energy × timbral richness (high-band share). Selecting only by "most stable pitch"
    # systematically favours the darkest, most characterless closed-mouth humming; the good material is in segments that are "stable and have character".
    def _richness(a_i, b_i):
        c = int((f0_times[a_i] + f0_times[b_i]) / 2 * sr)
        seg = np.asarray(y[max(0, c - sr // 2) : c + sr // 2], np.float64)
        if len(seg) < 2048:
            return 0.3
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
        tot = float((S[(fr > 60) & (fr < 5000)] ** 2).sum()) + 1e-12
        hi = float((S[(fr > 700) & (fr < 5000)] ** 2).sum())
        return hi / tot

    runs = []
    i = 0
    while i < T:
        if good[i]:
            j = i
            while j + 1 < T and good[j + 1]:
                j += 1
            if (f0_times[j] - f0_times[i]) >= min_s:
                dur = float(f0_times[j] - f0_times[i])
                erg = float(energy[i : j + 1].mean()) / emax
                score = (dur**0.3) * erg * (0.25 + _richness(i, j))
                runs.append((score, i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return None
    _, best_a, best_b = max(runs)
    a_s = int(f0_times[best_a] * sr)
    b_s = int(min(len(y), f0_times[best_b] * sr + win))
    if (b_s - a_s) / sr > max_s:  # too long -> take max_s around the energy peak
        pk = best_a + int(np.argmax(energy[best_a : best_b + 1]))
        c = int(f0_times[pk] * sr)
        a_s = max(a_s, c - int(max_s * sr / 2))
        b_s = min(b_s, c + int(max_s * sr / 2))
    return a_s, b_s


def analyze_timbre(
    y,
    sr: int = _SR,
    name: str = "untitled",
    source_file: str = "",
    n_harmonics: int = _NH,
    n_mels: int = 40,
    quasi_harmonic: bool = False,
    f0_hint: float | None = None,
    denoise: bool = False,
    window: str | None = None,
    cross_check: bool = True,
) -> dict:
    """Isolated note -> TimbreSnapshot.

    quasi_harmonic=True is for wine glasses/bowls: widens the peak search window and records the measured partial_freqs.
    f0_hint: known ground-truth f0 (Hz), fixes PESTO octave/harmonic errors in the low register (used for dataset corpus building).
    denoise=True (used by the app capture path): subtract the steady noise floor from the noise envelope per band. Room hum/hiss is constant
    (≈ per-band low percentile over time) and is removed; breath/bow noise varies with playing and is kept. Default False: the steady noise of
    synthetic test signals is an intended component.
    """
    y = np.asarray(y, dtype=np.float32)
    f0, conf, ftimes = track_f0(y, sr=sr, cross_check=cross_check)
    if f0_hint is not None and f0_hint > 0:
        f0 = _snap_to_hint(f0, float(f0_hint))
    # automatic stable-window selection (window="auto", app capture path): pick a sustained segment usable for timbre capture from a natural recording.
    window_found = None
    window_mid_cut = False
    if window == "auto":
        w = select_stable_window(y, sr, f0, ftimes)
        window_found = w is not None
        if w is not None:
            a, b = w
            # when the cut falls mid-recording (not at a natural attack) the segment has no real attack, and the start of the residual is only the harmonic model's
            # mismatch transient at the artificial cut; storing it as an attack grain would replay a "click" on every key, so attack extraction is skipped here.
            # A plucked note's stable window naturally starts after the attack (pitch is unstable during the attack), so look back 0.45s: if there is
            # a real silence->burst attack, move the window back to include it and extract the grain as usual (the source of the plucked character).
            window_mid_cut = a > int(0.12 * sr)
            if window_mid_cut:
                lb = np.asarray(y[max(0, a - int(0.45 * sr)) : a], np.float64)
                head = np.asarray(y[a : a + int(0.25 * sr)], np.float64)
                if lb.size > int(0.1 * sr) and head.size:
                    hopq = 256
                    nq = len(lb) // hopq
                    rq = np.sqrt((lb[: nq * hopq].reshape(nq, hopq) ** 2).mean(1))
                    head_rms = float(np.sqrt((head**2).mean())) + 1e-12
                    if rq.size and float(rq.min()) < 0.15 * head_rms:
                        onset_i = (a - nq * hopq) + int(np.argmin(rq)) * hopq
                        a = max(0, onset_i)  # move back to the silence point, including the real attack
                        window_mid_cut = False
            fm = (ftimes >= a / sr) & (ftimes < b / sr)
            if int(fm.sum()) >= 4:
                y = np.ascontiguousarray(y[a:b])
                f0, conf, ftimes = f0[fm], conf[fm], ftimes[fm] - a / sr
    f0_ref = f0_scalar(f0)
    # musical-tone gate metric (computed after window selection; the cv of stable segments in sung melodies/sentences drops to musical-tone levels).
    # separation: musical tone cv≤0.008; white noise 0.44 / pseudo speech 0.39 / polyphonic chord 0.21.
    _good = np.isfinite(f0) & (f0 > 0)
    voiced_ratio = float(_good.mean()) if f0.size else 0.0
    f0_cv = float(np.std(f0[_good]) / np.mean(f0[_good])) if _good.sum() > 3 else 9.9

    H, htimes, info = harmonics(
        y, f0, sr=sr, f0_times=ftimes, n_harmonics=n_harmonics, measure_partials=quasi_harmonic
    )
    f0_frame = info["f0_per_frame"]
    partial_freqs = info["partial_freqs"] if quasi_harmonic else None

    res = extract_residual(
        y,
        H,
        f0_frame,
        sr=sr,
        n_fft=info["n_fft"],
        hop=_HOP,
        partial_freqs=partial_freqs,
        f0_ref=f0_ref,
    )
    N_env, centers = noise_envelope(
        res,
        sr=sr,
        n_mels=n_mels,
        f0_ref=f0_ref,
        # rumble below 0.6× f0 is not timbre
        hp_hz=min(500.0, max(60.0, 0.6 * float(f0_ref))),
    )
    if denoise and N_env.ndim == 2 and N_env.shape[1] >= 8:
        floor = np.percentile(N_env, 20, axis=1, keepdims=True)  # per-band steady noise floor (low percentile over time)
        N_env = np.sqrt(np.maximum(N_env**2 - (1.1 * floor) ** 2, 0.0))  # spectral subtraction in the power domain
    attack = np.zeros(0, dtype=np.float32) if window_mid_cut else attack_residual(res, sr=sr)
    vibrato = detect_vibrato(f0, ftimes)

    return make_snapshot(
        f0_ref,
        H,
        N_env,
        attack,
        partial_freqs=partial_freqs,
        vibrato=vibrato,
        noise_centers_hz=centers,
        name=name,
        source_file=source_file,
        sr=sr,
        harm_hop=_HOP,
        harm_nfft=int(info["n_fft"]),
        voiced_ratio=voiced_ratio,
        f0_cv=f0_cv,
        window_found=bool(window_found) if window_found is not None else None,
    )
