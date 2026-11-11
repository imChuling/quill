"""Performance capture: extract both the "timbre" (TimbreSnapshot) and the "gesture" (behaviour trajectories) from one performance recording.

A single steady note is just one point in the object's timbre space; the strongest information about instrument identity (attack transient,
spectral flux, excitation gesture) only exists in performance, and information absent from the recording cannot be recovered. So the recording
is a short performance on the object (several notes, with attacks/dynamics/gestures), and from that one recording we extract:
  1. TimbreSnapshot: taken from a complete representative note including its attack (mid-section not cropped), keeping transients and H_env variation over time;
  2. BehaviorTrajectories (.tplib v2): four continuous residual trajectories per event.
Output is still a standard snapshot dict plus an optional behavior list; the synthesis side does not distinguish the source.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from analysis.harmonic import track_f0  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from analysis.tplib import event_trajectories  # noqa: E402

SR = 44100
_MIN_EV = 0.18  # minimum event duration (shorter is not enough for timbre capture/trajectory extraction)
_PRE_ROLL = 0.03  # keep 30ms before the attack so the transient stays intact


def load_any(path) -> np.ndarray:
    """wav/m4a/mp3 all accepted; unified to 44.1k mono."""
    try:
        import soundfile as sf

        y, sr = sf.read(str(path))
        y = y.mean(axis=1) if getattr(y, "ndim", 1) > 1 else y
        if sr != SR:
            import librosa

            y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=SR)
    except Exception:
        import librosa

        y, _ = librosa.load(str(path), sr=SR, mono=True)
    return np.asarray(y, np.float32)


def find_events(y, sr=SR, max_events=12):
    """Performance segment -> [(t_on, t_off)]; falls back to the whole segment on failure."""
    spans = []
    try:
        from analysis.segment import onset_f0_events

        for e in onset_f0_events(y, sr=sr):
            t0, t1 = float(e.get("t_on", 0)), float(e.get("t_off", 0))
            if t1 - t0 >= _MIN_EV:
                spans.append((max(0.0, t0 - _PRE_ROLL), t1))
    except Exception:
        pass
    if not spans:
        import librosa

        on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
        for i, t0 in enumerate(on):
            t1 = on[i + 1] if i + 1 < len(on) else len(y) / sr
            if t1 - t0 >= _MIN_EV:
                spans.append((max(0.0, t0 - _PRE_ROLL), min(t1, t0 + 3.0)))
    return spans[:max_events] or [(0.0, len(y) / sr)]


def _spec_flux(seg, sr):
    """Spectral flux: mean adjacent-frame difference of the per-frame normalised log spectrum (one of Grey 1977's three perceptual dimensions)."""
    n_fft, hop = 2048, 512
    if len(seg) < n_fft * 2:
        return 0.0
    fr = np.lib.stride_tricks.sliding_window_view(seg, n_fft)[::hop]
    S = np.abs(np.fft.rfft(fr * np.hanning(n_fft), axis=1))
    S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-12)  # normalise per frame, only spectral-shape change matters
    return float(np.mean(np.abs(np.diff(np.log(np.maximum(S, 1e-9)), axis=0))))


def _score_event(y, sr, t0, t1):
    """Score for choosing the representative note for timbre capture; returns (score, median f0); (-1, None) if unusable.

    Rewards spectral flux and duration; energy is only a threshold. Pitch instability is not penalised: otherwise it would pick short, rigid notes
    and avoid gestures and spectral flux altogether.
    """
    seg = y[int(t0 * sr) : int(t1 * sr)].astype(np.float64)
    if len(seg) < int(_MIN_EV * sr):
        return -1.0, None
    e = float(np.sqrt((seg**2).mean()))
    if e < 1e-4:
        return -1.0, None
    f0, _, _ = track_f0(seg.astype(np.float32), sr=sr, cross_check=False)
    v = np.asarray(f0, np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 5:
        return -1.0, None
    dur = t1 - t0
    flux = _spec_flux(seg, sr)
    return (dur**0.7) * e * (0.3 + flux), float(np.median(v))


def capture_from_performance(
    path_or_audio, sr=SR, name="performance", quasi_harmonic=False, verbose=True
):
    """Performance segment -> (snapshot, behaviors, info).

    snapshot:  taken from a complete representative note including its attack (window=None, analyze_timbre uses the whole segment,
               keeping transients and H_env variation over time);
    behaviors: .tplib v2 four trajectories per event (pitch/energy/brightness/noise).
    """
    y = (
        load_any(path_or_audio)
        if isinstance(path_or_audio, (str, Path))
        else np.asarray(path_or_audio, np.float32)
    )
    spans = find_events(y, sr)
    scored = []
    for t0, t1 in spans:
        s, f0m = _score_event(y, sr, t0, t1)
        if s > 0:
            scored.append((s, t0, t1, f0m))
    if not scored:
        raise ValueError("No note usable for timbre capture found in the performance (too short / no pitch?)")
    scored.sort(reverse=True)
    _, bt0, bt1, bf0 = scored[0]

    seg = y[int(bt0 * sr) : int(bt1 * sr)]
    snap = analyze_timbre(
        seg,
        sr=sr,
        name=name,
        quasi_harmonic=quasi_harmonic,
        denoise=True,
        window=None,
        cross_check=False,
    )

    behaviors = []
    for t0, t1 in spans:
        tj = event_trajectories(y, sr, t0, t1)
        if tj["energy"]["confidence"] > 0:
            behaviors.append(tj)

    H = np.asarray(snap["H_env"], np.float64)
    flux = float(np.mean(np.abs(np.diff(np.log(np.maximum(H, 1e-9)), axis=1))))
    info = {
        "n_events": len(spans),
        "n_behaviors": len(behaviors),
        "pick_span": (round(bt0, 2), round(bt1, 2)),
        "pick_f0": round(bf0 or 0, 1),
        "H_frames": H.shape[1],
        "spectral_flux": round(flux, 4),
    }
    if verbose:
        print(
            f"[perf-capture] {info['n_events']} events -> representative note {info['pick_span']}s "
            f"@{info['pick_f0']}Hz; {info['n_behaviors']} behaviour trajectories; "
            f"spectral flux={info['spectral_flux']}"
        )
    return snap, behaviors, info
