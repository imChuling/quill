"""Per-event featurisation, event analysis orchestration, events.json I/O.

The features are a subset of the Peeters (JASA 2011) descriptors:
  log_attack · centroid_mean · centroid_slope · f0_slope_cents_s · polyphony · hnr
plus velocity_est (RMS -> 0-1), f0_track (including bend) and vibrato.

`analyze_events(y)` chains the dual-front-end segmentation and per-event features into the full pipeline, producing the events list of events.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.segment import (
    basic_pitch_events,
    ghost_octave_filter,
    hz_to_midi,
    onset_f0_events,
    union_events,
)

try:
    from quill_config import CFG

    _SR = int(CFG["audio"]["sr"])
except Exception:  # pragma: no cover
    _SR = 44100


# --------------------------------------------------------------------------- #
# Single-event features
# --------------------------------------------------------------------------- #
def _seg(y, sr, t0, t1):
    a = max(0, int(t0 * sr))
    b = min(len(y), int(t1 * sr))
    return y[a:b] if b > a else y[a : a + 1]


def log_attack_time(y_ev, sr, max_ms=150.0):
    """Peeters log-attack-time: log10(time in seconds to reach the energy peak)."""
    import librosa

    hop = 128
    rms = librosa.feature.rms(y=y_ev, frame_length=512, hop_length=hop)[0]
    if rms.size == 0:
        return float("nan")
    pk = int(np.argmax(rms))
    lat_s = max(pk * hop / sr, 1e-4)
    return float(np.log10(lat_s))


def spectral_centroid_stats(y_ev, sr):
    """Mean spectral centroid (Hz) and its time slope (Hz/s)."""
    import librosa

    if len(y_ev) < 512:
        return float("nan"), float("nan")
    c = librosa.feature.spectral_centroid(y=y_ev, sr=sr, hop_length=256)[0]
    if c.size == 0:
        return float("nan"), float("nan")
    t = np.arange(c.size) * 256 / sr
    mean = float(np.mean(c))
    slope = float(np.polyfit(t, c, 1)[0]) if c.size >= 2 and t[-1] > 0 else 0.0
    return mean, slope


def f0_track_for(f0, f0_times, t0, t1, max_points=40):
    """PESTO f0 track within the event window -> [[t,hz],...] (including bend; downsampled to max_points)."""
    f0 = np.asarray(f0)
    f0_times = np.asarray(f0_times)
    m = (f0_times >= t0) & (f0_times <= t1) & np.isfinite(f0) & (f0 > 0)
    ts, hz = f0_times[m], f0[m]
    if ts.size == 0:
        return []
    if ts.size > max_points:
        idx = np.linspace(0, ts.size - 1, max_points).astype(int)
        ts, hz = ts[idx], hz[idx]
    return [[round(float(a), 4), round(float(b), 2)] for a, b in zip(ts, hz)]


def f0_slope_cents_s(track):
    """Slope of the f0 track in cents/second (direct evidence of a slide)."""
    if len(track) < 2:
        return 0.0
    t = np.array([p[0] for p in track])
    hz = np.array([p[1] for p in track])
    med = np.median(hz)
    if med <= 0:
        return 0.0
    cents = 1200.0 * np.log2(hz / med)
    if t[-1] - t[0] < 1e-3:
        return 0.0
    return float(np.polyfit(t, cents, 1)[0])


def polyphony_at(ev, all_events):
    """Number of events overlapping this event in time (including itself) = polyphony."""
    a, b = ev["t_on"], ev["t_off"]
    cnt = 0
    for o in all_events:
        if o["t_on"] < b and o["t_off"] > a:
            cnt += 1
    return int(max(1, cnt))


def hnr_db(y_ev, sr, f0_hz, n_harm=40, cents=40.0):
    """Harmonic-to-noise ratio (dB): harmonic energy within ±cents vs the rest."""
    if len(y_ev) < 256 or not (np.isfinite(f0_hz) and f0_hz > 0):
        return float("nan")
    w = np.hanning(len(y_ev))
    S = np.abs(np.fft.rfft(y_ev * w)) ** 2
    f = np.fft.rfftfreq(len(y_ev), 1 / sr)
    nyq = sr / 2
    harm_mask = np.zeros_like(f, dtype=bool)
    fac = 2 ** (cents / 1200.0)
    for k in range(1, n_harm + 1):
        fk = k * f0_hz
        if fk >= nyq:
            break
        harm_mask |= (f >= fk / fac) & (f <= fk * fac)
    harm = S[harm_mask].sum()
    noise = S[~harm_mask].sum()
    return float(10 * np.log10(harm / (noise + 1e-12) + 1e-12))


# --------------------------------------------------------------------------- #
# Event analysis orchestration
# --------------------------------------------------------------------------- #
def analyze_events(y, sr=_SR, use_basic_pitch=True, bp_amp_thresh=0.4, return_diagnostics=False):
    """Audio -> events list. Union of two front ends + ghost-octave filtering + per-event features."""
    from analysis.harmonic import track_f0
    from analysis.timbre import detect_vibrato

    y = np.asarray(y, dtype=np.float32)
    f0, conf, ftimes = track_f0(y, sr=sr)

    onset_evs = onset_f0_events(y, sr=sr, f0=f0, f0_times=ftimes)
    ghosts = []
    if use_basic_pitch:
        bp = basic_pitch_events(y, sr=sr)
        bp, ghosts = ghost_octave_filter(bp, f0, ftimes, amp_thresh=bp_amp_thresh)
        merged, disagree = union_events(bp, onset_evs)
    else:
        merged, disagree = onset_evs, []

    # fragment guard: drop micro-events shorter than min_event_ms that are not in a dense cluster (strum)
    try:
        from quill_config import CFG as _C

        min_dur = float(_C.get("articulation", {}).get("min_event_ms", 30.0)) / 1000.0
        win = float(_C.get("articulation", {}).get("strum_window_ms", 80.0)) / 1000.0
    except Exception:
        min_dur, win = 0.030, 0.080
    if min_dur > 0 and merged:
        ts = sorted(e["t_on"] for e in merged)

        # ≥3 onsets within an 80ms window count as a cluster (strum), not dropped
        def _in_cluster(t):
            return sum(1 for x in ts if abs(x - t) < win) >= 3

        merged = [
            e for e in merged if (e["t_off"] - e["t_on"]) >= min_dur or _in_cluster(e["t_on"])
        ]

    # velocity reference: maximum RMS over events
    rms_list = []
    for ev in merged:
        s = _seg(y, sr, ev["t_on"], ev["t_off"])
        rms_list.append(float(np.sqrt(np.mean(s.astype(np.float64) ** 2))) if s.size else 0.0)
    ref = max(rms_list) if rms_list else 1.0
    ref = ref if ref > 1e-9 else 1.0

    events = []
    for i, ev in enumerate(merged):
        t0, t1 = ev["t_on"], ev["t_off"]
        s = _seg(y, sr, t0, t1)
        track = f0_track_for(f0, ftimes, t0, t1)
        pitch = ev.get("pitch_midi")
        if (pitch is None or not np.isfinite(pitch)) and track:
            hz_med = float(np.median([p[1] for p in track]))
            pitch = float(hz_to_midi(np.array([hz_med]))[0])
        f0_med = float(np.median([p[1] for p in track])) if track else float("nan")

        cmean, cslope = spectral_centroid_stats(s, sr)
        feats = {
            "log_attack": log_attack_time(s, sr),
            "centroid_mean": cmean,
            "centroid_slope": cslope,
            "f0_slope_cents_s": f0_slope_cents_s(track),
            "polyphony": polyphony_at(ev, merged),
            "hnr": hnr_db(s, sr, f0_med),
        }
        vib = (
            detect_vibrato(np.array([p[1] for p in track]), np.array([p[0] for p in track]))
            if len(track) >= 16
            else None
        )
        events.append(
            {
                "id": i,
                "t_on": round(float(t0), 4),
                "t_off": round(float(t1), 4),
                "pitch_midi": (
                    round(float(pitch), 3) if pitch is not None and np.isfinite(pitch) else None
                ),
                "velocity_est": round(min(1.0, rms_list[i] / ref), 4),
                "f0_track": track,
                "features": {
                    k: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else v)
                    for k, v in feats.items()
                },
                "label": "",  # written by articulation.classify_events
                "vibrato": vib,
                "_agree": bool(ev.get("agree", False)),
            }
        )

    if return_diagnostics:
        return events, {"disagreements": disagree, "ghosts": ghosts, "n_onset": len(onset_evs)}
    return events


def compute_event_features(y, sr, events):
    """Add features, f0_track and vibrato per event for events with known t_on/t_off/pitch (e.g. IDMT annotations).

    track_f0, the STFT and per-frame centroid/RMS are computed once per file; per-event values are derived by slicing frames, avoiding per-event
    librosa overhead. Feature definitions match analyze_events.
    """
    import librosa

    from analysis.harmonic import track_f0
    from analysis.timbre import detect_vibrato

    y = np.asarray(y, dtype=np.float32)
    # f0 with a 20ms hop: enough for slide/vibrato decisions, halves PESTO time
    f0, conf, ftimes = track_f0(y, sr=sr, step_size_ms=20.0, cross_check=False)

    n_fft, hop = 1024, 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))  # [bins, T]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    fr_t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    power = S**2
    cent_f = (freqs[:, None] * power).sum(0) / (power.sum(0) + 1e-12)  # per-frame centroid
    rms_f = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]
    nyq = sr / 2.0
    eps = 1e-12

    # velocity reference: maximum of the per-event mean frame RMS
    def _frames(t0, t1):
        return np.where((fr_t >= t0) & (fr_t < max(t1, t0 + 1e-3)))[0]

    rms_list = []
    for ev in events:
        idx = _frames(ev["t_on"], ev["t_off"])
        rms_list.append(float(np.mean(rms_f[idx])) if idx.size else 0.0)
    ref = max(rms_list) if rms_list else 1.0
    ref = ref if ref > 1e-9 else 1.0

    for i, ev in enumerate(events):
        t0, t1 = ev["t_on"], ev["t_off"]
        idx = _frames(t0, t1)
        track = f0_track_for(f0, ftimes, t0, t1)
        f0_med = float(np.median([p[1] for p in track])) if track else float("nan")

        if idx.size >= 2:
            cmean = float(np.mean(cent_f[idx]))
            tt = fr_t[idx] - t0
            cslope = float(np.polyfit(tt, cent_f[idx], 1)[0]) if tt[-1] > 0 else 0.0
        elif idx.size == 1:
            cmean, cslope = float(cent_f[idx[0]]), 0.0
        else:
            cmean, cslope = float("nan"), 0.0

        # log_attack: time of the RMS peak frame within the attack window
        aw = np.where((fr_t >= t0) & (fr_t < t0 + 0.15))[0]
        if aw.size:
            pk = aw[int(np.argmax(rms_f[aw]))]
            log_atk = float(np.log10(max(fr_t[pk] - t0, 1e-4)))
        else:
            log_atk = float("nan")

        # HNR: harmonic energy within ±40cents vs the rest in the event's mean frame spectrum
        if idx.size and np.isfinite(f0_med) and f0_med > 0:
            ps = power[:, idx].mean(1)
            hmask = np.zeros_like(freqs, dtype=bool)
            fac = 2 ** (40 / 1200.0)
            for k in range(1, 41):
                fk = k * f0_med
                if fk >= nyq:
                    break
                hmask |= (freqs >= fk / fac) & (freqs <= fk * fac)
            hnr = float(10 * np.log10(ps[hmask].sum() / (ps[~hmask].sum() + eps) + eps))
        else:
            hnr = float("nan")

        ev["f0_track"] = track
        ev["features"] = {
            "log_attack": log_atk,
            "centroid_mean": cmean,
            "centroid_slope": cslope,
            "f0_slope_cents_s": f0_slope_cents_s(track),
            "polyphony": polyphony_at(ev, events),
            "hnr": hnr,
        }
        ev["velocity_est"] = round(min(1.0, rms_list[i] / ref), 4)
        if not ev.get("vibrato") and len(track) >= 16:
            ev["vibrato"] = detect_vibrato(
                np.array([p[1] for p in track]), np.array([p[0] for p in track])
            )
    return events


# --------------------------------------------------------------------------- #
# events.json serialisation
# --------------------------------------------------------------------------- #
_PUBLIC_KEYS = (
    "id",
    "t_on",
    "t_off",
    "pitch_midi",
    "velocity_est",
    "f0_track",
    "features",
    "label",
    "vibrato",
)


def save_events(path, events):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: ev[k] for k in _PUBLIC_KEYS if k in ev} for ev in events]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, ensure_ascii=False, indent=2)
    return path


def load_events(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
