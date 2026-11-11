"""Event segmentation: union of two front ends, Basic Pitch and a custom onset/f0 detector.

  * Basic Pitch (Spotify, ICASSP 2022): lightweight polyphonic transcription with pitch bend
  * custom `librosa.onset.onset_detect(backtrack=True)` + PESTO frame-level f0 segmentation

Hit by both (same pitch, same onset) = high confidence; hit by only one = goes to disagreements for manual review.
Guitar harmonics make Basic Pitch produce ghost upper-octave notes, filtered by a confidence threshold + consistency with PESTO f0.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np

try:
    from quill_config import CFG

    _SR = int(CFG["audio"]["sr"])
except Exception:  # pragma: no cover
    _SR = 44100


# --------------------------------------------------------------------------- #
# Pitch/frequency conversion
# --------------------------------------------------------------------------- #
def hz_to_midi(f):
    f = np.asarray(f, dtype=np.float64)
    out = np.full_like(f, np.nan)
    pos = f > 0
    out[pos] = 69.0 + 12.0 * np.log2(f[pos] / 440.0)
    return out


def _f0_median_in(f0, f0_times, t0, t1, skip_ms=10.0):
    """Median PESTO f0 (Hz) within the event window [t0+skip, t1]; NaN if no valid value."""
    f0 = np.asarray(f0)
    f0_times = np.asarray(f0_times)
    a = t0 + skip_ms / 1000.0
    m = (f0_times >= a) & (f0_times <= max(t1, a + 1e-3)) & np.isfinite(f0) & (f0 > 0)
    return float(np.median(f0[m])) if np.any(m) else float("nan")


# --------------------------------------------------------------------------- #
# Front end 1: Basic Pitch
# --------------------------------------------------------------------------- #
def basic_pitch_events(y, sr=_SR, onset_thresh=0.5, frame_thresh=0.3, min_note_len_ms=60.0):
    """Basic Pitch -> [{t_on,t_off,pitch_midi,amp,bend}] (bend is a semitone offset trajectory)."""
    import soundfile as sf
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            tmp = fh.name
        sf.write(tmp, np.asarray(y, dtype=np.float32), sr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, note_events = predict(
                tmp,
                ICASSP_2022_MODEL_PATH,
                onset_threshold=onset_thresh,
                frame_threshold=frame_thresh,
                minimum_note_length=min_note_len_ms,
            )
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

    out = []
    for ne in note_events:
        t_on, t_off, pitch, amp = float(ne[0]), float(ne[1]), int(ne[2]), float(ne[3])
        bend = ne[4] if len(ne) > 4 else None
        out.append(
            {
                "t_on": t_on,
                "t_off": t_off,
                "pitch_midi": float(pitch),
                "amp": amp,
                "bend": (None if bend is None else np.asarray(bend, dtype=np.float32)),
                "source": "bp",
            }
        )
    out.sort(key=lambda e: e["t_on"])
    return out


# --------------------------------------------------------------------------- #
# Front end 2: custom onset + f0 segmentation
# --------------------------------------------------------------------------- #
def onset_f0_events(y, sr=_SR, f0=None, f0_times=None, hop=512):
    """librosa onset(backtrack) + PESTO f0 segmentation -> [{t_on,t_off,pitch_midi,...}]."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True, units="time")
    onsets = np.asarray(onsets, dtype=np.float64)
    dur = len(y) / sr
    if onsets.size == 0:
        return []

    # RMS envelope used to set the offset (energy drop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    floor = 0.15 * float(np.max(rms)) if rms.size else 0.0

    events = []
    for i, t_on in enumerate(onsets):
        next_on = onsets[i + 1] if i + 1 < len(onsets) else dur
        # offset = energy falls below floor or the next onset, whichever comes first
        seg = (rms_t >= t_on) & (rms_t < next_on)
        t_off = next_on
        if np.any(seg):
            below = np.where(seg & (rms < floor))[0]
            if below.size:
                t_off = float(rms_t[below[0]])
        t_off = max(t_off, t_on + 0.03)
        pitch_hz = (
            _f0_median_in(f0, f0_times, t_on, min(t_on + 0.2, t_off))
            if f0 is not None
            else float("nan")
        )
        pitch_midi = float(hz_to_midi(np.array([pitch_hz]))[0]) if pitch_hz > 0 else float("nan")
        events.append(
            {
                "t_on": float(t_on),
                "t_off": float(t_off),
                "pitch_midi": pitch_midi,
                "amp": float("nan"),
                "bend": None,
                "source": "onset",
            }
        )
    return events


# --------------------------------------------------------------------------- #
# Ghost-octave filtering
# --------------------------------------------------------------------------- #
def ghost_octave_filter(events, f0, f0_times, amp_thresh=0.0, tol_semitone=1.5):
    """Remove ghost-octave notes + weak events, compared against PESTO f0:

    the event pitch p is compared with the expected PESTO pitch e in the window; if |p-e|>tol but p±12 matches e
    (upper-octave harmonic ghost or lower-octave subharmonic ghost) => drop;
    also drop weak events with amplitude < amp_thresh (Basic Pitch confidence threshold).
    """
    kept, dropped = [], []
    for ev in events:
        if ev.get("amp") is not None and np.isfinite(ev["amp"]) and ev["amp"] < amp_thresh:
            dropped.append({**ev, "drop_reason": "low_amp"})
            continue
        e_hz = _f0_median_in(f0, f0_times, ev["t_on"], ev["t_off"])
        if e_hz > 0 and np.isfinite(ev["pitch_midi"]):
            e_midi = float(hz_to_midi(np.array([e_hz]))[0])
            p = ev["pitch_midi"]
            far = abs(p - e_midi) > tol_semitone
            octave_ghost = (
                abs((p - 12) - e_midi) <= tol_semitone  # upper-octave harmonic
                or abs((p + 12) - e_midi) <= tol_semitone
            )  # lower-octave subharmonic
            if far and octave_ghost:
                dropped.append({**ev, "drop_reason": "ghost_octave"})
                continue
        kept.append(ev)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Union of the two front ends
# --------------------------------------------------------------------------- #
def union_events(bp_events, onset_events, onset_tol_s=0.05, pitch_tol=1.0):
    """Union: hit by both front ends (same pitch ±pitch_tol, onset ±onset_tol) is marked agree;
    hit by only one goes to disagreements. Returns (events, disagreements).

    Basic Pitch events are primary (with pitch/offset/bend); the custom front end fills gaps + confirms consistency.
    """
    bp = sorted(bp_events, key=lambda e: e["t_on"])
    on = sorted(onset_events, key=lambda e: e["t_on"])
    used_on = [False] * len(on)

    merged, disagree = [], []
    for e in bp:
        match = None
        for j, o in enumerate(on):
            if used_on[j]:
                continue
            if abs(o["t_on"] - e["t_on"]) <= onset_tol_s and (
                not np.isfinite(o["pitch_midi"])
                or abs(o["pitch_midi"] - e["pitch_midi"]) <= pitch_tol
            ):
                match = j
                break
        ev = dict(e)
        ev["agree"] = match is not None
        if match is not None:
            used_on[match] = True
            ev["t_on"] = min(ev["t_on"], on[match]["t_on"])  # onset backtrack is more accurate
        else:
            disagree.append({**e, "only": "bp"})
        merged.append(ev)

    # add events only found by the custom front end (missed by BP)
    for j, o in enumerate(on):
        if not used_on[j] and np.isfinite(o["pitch_midi"]):
            ev = dict(o)
            ev["agree"] = False
            merged.append(ev)
            disagree.append({**o, "only": "onset"})

    merged.sort(key=lambda e: e["t_on"])
    return merged, disagree
