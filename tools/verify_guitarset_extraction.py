"""Validate the quality of per-note trajectory extraction on GuitarSet solo_mic.

Runs event_trajectories on 3 solo files and checks:
1. JAMS annotations correctly yield note onset/offset
2. f0 tracking is stable on monophonic guitar
3. the confidence distribution of the 4×48 trajectories is reasonable (>0.8 for the majority)
4. output statistics: total notes, fraction of valid trajectories, distribution of trajectory std
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mirdata
import numpy as np
import soundfile as sf

from analysis.tplib import _TRAJ_KEYS, event_trajectories

DATASETS = os.environ.get("QUILL_DATASETS", str(Path.home() / "Desktop/MT/datasets"))
GSET_HOME = DATASETS + "/mirdata_guitarset"
MIC_DIR = DATASETS + "/GuitarSet_mic"

ds = mirdata.initialize("guitarset", data_home=GSET_HOME)

track_ids = list(ds.track_ids)
solo_ids = [t for t in track_ids if "solo" in t]
test_ids = solo_ids[:3]

print(f"Total tracks: {len(track_ids)}, solo: {len(solo_ids)}, testing: {len(test_ids)}")

results = []

for tid in test_ids:
    track = ds.track(tid)
    print(f"\n{'=' * 60}")
    print(f"Track: {tid}")

    mic_file = f"{MIC_DIR}/{tid}_mic.wav"
    try:
        y, sr = sf.read(mic_file, dtype="float32")
    except Exception as e:
        print(f"  SKIP: cannot read {mic_file}: {e}")
        continue

    if y.ndim > 1:
        y = y.mean(axis=1)
    print(f"  Audio: {len(y) / sr:.1f}s @ {sr}Hz")

    notes = track.notes_all
    if notes is None:
        print("  SKIP: no note annotations")
        continue

    intervals = notes.intervals
    pitches = notes.pitches
    print(f"  Notes: {len(intervals)}")

    note_stats = []
    for i, (interval, pitch_hz) in enumerate(zip(intervals, pitches)):
        t_on, t_off = float(interval[0]), float(interval[1])
        dur = t_off - t_on
        if dur < 0.05:
            continue

        traj = event_trajectories(y, sr, t_on, t_off)

        stats = {
            "note_idx": i,
            "t_on": t_on,
            "t_off": t_off,
            "dur": dur,
            "pitch_hz": float(pitch_hz),
        }
        for k in _TRAJ_KEYS:
            pts = traj[k]["pts"]
            conf = traj[k]["confidence"]
            stats[f"{k}_conf"] = conf
            stats[f"{k}_std"] = float(np.std(pts))
            stats[f"{k}_range"] = float(np.ptp(pts))
        note_stats.append(stats)

    if not note_stats:
        print("  SKIP: no valid notes")
        continue

    confs = {k: [s[f"{k}_conf"] for s in note_stats] for k in _TRAJ_KEYS}
    stds = {k: [s[f"{k}_std"] for s in note_stats] for k in _TRAJ_KEYS}

    print(f"  Valid notes: {len(note_stats)}")
    for k in _TRAJ_KEYS:
        c = np.array(confs[k])
        s = np.array(stds[k])
        high_conf = (c > 0.8).mean() * 100
        print(
            f"  {k:12s}: conf>{0.8:.0%} = {high_conf:5.1f}% | "
            f"std mean={s.mean():.4f} med={np.median(s):.4f} | "
            f"conf mean={c.mean():.3f}"
        )

    results.append(
        {
            "track_id": tid,
            "n_notes": len(note_stats),
            "confs": {k: float(np.mean(confs[k])) for k in _TRAJ_KEYS},
            "high_conf_pct": {k: float((np.array(confs[k]) > 0.8).mean()) for k in _TRAJ_KEYS},
        }
    )

print(f"\n{'=' * 60}")
print("SUMMARY")
for r in results:
    print(
        f"  {r['track_id']}: {r['n_notes']} notes, "
        f"pitch conf={r['confs']['pitch']:.2f}, "
        f"energy conf={r['confs']['energy']:.2f}"
    )
total_notes = sum(r["n_notes"] for r in results)
print(f"  Total: {total_notes} notes across {len(results)} tracks")
if results:
    avg_pitch_conf = np.mean([r["confs"]["pitch"] for r in results])
    avg_pitch_high = np.mean([r["high_conf_pct"]["pitch"] for r in results])
    print(f"  Avg pitch confidence: {avg_pitch_conf:.3f}")
    print(f"  Avg pitch high-conf rate: {avg_pitch_high * 100:.1f}%")
    verdict = "PASS" if avg_pitch_conf > 0.7 else "NEEDS HEX"
    print(f"\n  VERDICT: {verdict}")
    if verdict == "NEEDS HEX":
        print("  → solo_mic f0 tracking too noisy, need hexaphonic per-string audio")
