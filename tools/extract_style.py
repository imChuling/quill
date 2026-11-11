"""Build an articulation library (style library) from a recording in one command: wav in, npz out.

    python tools/extract_style.py input.wav -o my_style.npz

The output format is exactly the same as urmp_vn_trajectories.npz (usable directly by NeuralTake/ConditionedNN):
  features(N,7): midi, dur, prev_gap, next_gap, is_first, is_last, articulation
  trajectories(N,4,48), confidence(N,4), meta(json), traj_keys

Difference from tools/extract_trajectories_urmp.py (the URMP library build script): there are no ground-truth note annotations,
so note segmentation is fully automatic (monophonic assumption):
  librosa onset(backtrack) ∪ pyin f0 jump boundaries -> note segments;
  articulation labels use the same heuristics (vibrato = back-and-forth oscillation, slide = large one-way displacement).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from analysis.tplib import _TRAJ_KEYS, event_trajectories

ART_PLAIN, ART_VIBRATO, ART_SLIDE = 0, 1, 2
ART_NAMES = {0: "plain", 1: "vibrato", 2: "slide"}

F0_HOP = 256  # pyin hop (high time resolution, more accurate f0 jump boundaries)
MIN_NOTE_DUR = 0.10  # segments shorter than this are dropped (seconds)
PITCH_SPLIT_SEMI = 0.7  # if adjacent stable pitches differ by more than this many semitones -> add a boundary besides the onsets
STABLE_FRAMES = 4  # the pitch must be stable for this many consecutive frames to count as a "new note" (prevents vibrato being chopped up)


def classify_articulation(pitch_traj_cents, confidence):
    """The same rules as extract_trajectories_urmp.py (keeps labels consistent across libraries)."""
    if confidence < 0.5:
        return ART_PLAIN
    p = pitch_traj_cents
    rng = float(np.max(p) - np.min(p))
    if rng < 30.0:
        return ART_PLAIN
    net = abs(float(p[-4:].mean() - p[:4].mean()))
    d = np.diff(p)
    sign_changes = int(np.sum(np.abs(np.diff(np.sign(d[np.abs(d) > 1e-3]))) > 0))
    if net > 60.0 and net > 0.6 * rng:
        return ART_SLIDE
    if sign_changes >= 4 and rng >= 30.0:
        return ART_VIBRATO
    if rng >= 80.0:
        return ART_SLIDE
    return ART_PLAIN


def track_f0(y, sr):
    """Continuous f0 over the whole piece (pyin). Returns (hz[N], times[N]), unvoiced frames are NaN."""
    import librosa

    hz, vflag, _ = librosa.pyin(
        y, sr=sr, fmin=60.0, fmax=1600.0, frame_length=2048, hop_length=F0_HOP
    )
    hz = np.where(vflag, hz, np.nan).astype(np.float64)
    times = librosa.times_like(hz, sr=sr, hop_length=F0_HOP)
    return hz, times


def segment_notes(y, sr, f0, f0_times):
    """Monophonic note segmentation: onset boundaries ∪ f0 step-jump boundaries -> [(t_on, t_off, midi)].

    Onsets catch re-attacks (pluck/bow change/staccato); f0 jumps catch legato note changes (which onset detection misses).
    """
    import librosa

    dur_total = len(y) / sr
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True, units="time")
    boundaries = set(float(t) for t in onsets)

    # f0 jump boundaries: after a 3-frame median filter, step detection in the semitone domain
    semi = 12.0 * np.log2(np.maximum(f0, 1e-6) / 440.0) + 69.0
    m = semi.copy()
    for j in range(1, len(semi) - 1):
        win = semi[j - 1 : j + 2]
        if np.isfinite(win).sum() >= 2:
            m[j] = np.nanmedian(win)
    prev_stable = None
    run_val, run_len = None, 0
    for j in range(len(m)):
        v = m[j]
        if not np.isfinite(v):
            run_val, run_len = None, 0
            continue
        if run_val is not None and abs(v - run_val) < PITCH_SPLIT_SEMI * 0.5:
            run_len += 1
            run_val = 0.8 * run_val + 0.2 * v
        else:
            run_val, run_len = v, 1
        if run_len == STABLE_FRAMES:
            stable = run_val
            if prev_stable is not None and abs(stable - prev_stable) >= PITCH_SPLIT_SEMI:
                boundaries.add(float(f0_times[j - STABLE_FRAMES + 1]))
            prev_stable = stable

    # unvoiced segments (energy floor) as note-end candidates.
    # floor = peak -35dB (10% of the global peak is too high for a separated stem, 44% of frames would be misjudged as silent);
    # and it must stay below the floor for SILENCE_FRAMES consecutive frames to count as real silence (prevents vibrato/bow-change dips from cutting the note tail).
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    floor = float(np.max(rms)) * 10 ** (-35 / 20) if rms.size else 0.0
    SILENCE_FRAMES = 3
    silent = rms < floor
    sustained = np.zeros_like(silent)
    for j in range(len(silent) - SILENCE_FRAMES + 1):
        if silent[j : j + SILENCE_FRAMES].all():
            sustained[j] = True

    bounds = sorted(b for b in boundaries if 0.0 <= b < dur_total)
    if not bounds:
        return []
    notes = []
    for i, t_on in enumerate(bounds):
        next_b = bounds[i + 1] if i + 1 < len(bounds) else dur_total
        # offset = start of sustained silence or the next boundary, whichever comes first
        seg_m = (rms_t >= t_on) & (rms_t < next_b)
        t_off = next_b
        below = np.where(seg_m & sustained)[0]
        if below.size:
            t_off = max(float(rms_t[below[0]]), t_on + 0.03)
        if t_off - t_on < MIN_NOTE_DUR:
            continue
        # pitch = median f0 within the segment (first 30ms dropped)
        msk = (f0_times >= t_on + 0.03) & (f0_times <= t_off) & np.isfinite(f0)
        if msk.sum() < 3:
            continue
        midi = float(np.nanmedian(12.0 * np.log2(f0[msk] / 440.0) + 69.0))
        notes.append((t_on, t_off, midi))
    return notes


def extract(wav_path, out_path, verbose=True):
    y, sr = sf.read(str(wav_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    total_dur = len(y) / sr

    f0, f0_times = track_f0(y, sr)
    notes = segment_notes(y, sr, f0, f0_times)
    if verbose:
        print(f"{Path(wav_path).name}: {total_dur:.1f}s audio, {len(notes)} notes segmented")
    if not notes:
        raise SystemExit("no notes found — is the recording monophonic and pitched?")

    feats, trajs, confs, meta = [], [], [], []
    art_counts = {0: 0, 1: 0, 2: 0}
    for i, (t_on, t_off, midi) in enumerate(notes):
        traj = event_trajectories(y, sr, t_on, t_off)
        traj_arr = np.stack([traj[k]["pts"] for k in _TRAJ_KEYS])
        conf_arr = np.array([traj[k]["confidence"] for k in _TRAJ_KEYS], np.float32)
        art = classify_articulation(traj_arr[0], float(conf_arr[0]))
        art_counts[art] += 1
        prev_gap = float(t_on - notes[i - 1][1]) if i > 0 else 1.0
        next_gap = float(notes[i + 1][0] - t_off) if i < len(notes) - 1 else 1.0
        dur = t_off - t_on
        feats.append(
            np.array(
                [
                    midi,
                    dur,
                    prev_gap,
                    next_gap,
                    1.0 if (i == 0 or prev_gap > 0.5) else 0.0,
                    1.0 if (i == len(notes) - 1 or next_gap > 0.5) else 0.0,
                    float(art),
                ],
                np.float32,
            )
        )
        trajs.append(traj_arr)
        confs.append(conf_arr)
        meta.append(
            {
                "src": Path(wav_path).name,
                "note_idx": i,
                "t_on": round(t_on, 3),
                "midi": round(midi, 2),
                "dur": round(dur, 3),
                "art": art,
            }
        )

    features = np.stack(feats)
    trajectories = np.stack(trajs)
    confidence = np.stack(confs)
    np.savez_compressed(
        str(out_path),
        features=features,
        trajectories=trajectories,
        confidence=confidence,
        meta=json.dumps(meta),
        traj_keys=list(_TRAJ_KEYS),
    )
    if verbose:
        n = len(feats)
        print(f"saved -> {out_path}")
        print(f"  notes={n}  features={features.shape}  traj={trajectories.shape}")
        print(
            "  articulation: "
            + ", ".join(f"{ART_NAMES[a]}={c} ({c / n:.0%})" for a, c in art_counts.items())
        )
        for j, k in enumerate(_TRAJ_KEYS):
            print(f"  {k} avg confidence: {confidence[:, j].mean():.3f}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="extract a performance-style library from a recording")
    ap.add_argument("wav", help="monophonic pitched recording (~2 min is enough)")
    ap.add_argument("-o", "--out", default=None, help="output .npz (default: <wav>_style.npz)")
    args = ap.parse_args()
    out = args.out or str(Path(args.wav).with_suffix("")) + "_style.npz"
    extract(args.wav, out)


if __name__ == "__main__":
    main()
