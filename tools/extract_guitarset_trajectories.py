"""批量从 GuitarSet solo_mic 提取逐音符行为轨迹,产出训练集 .npz。

输出: datasets/guitarset_trajectories.npz
  - features: (N, F) float32 — 每音符特征向量
  - trajectories: (N, 4, 48) float32 — pitch/energy/brightness/noise 轨迹
  - confidence: (N, 4) float32 — 每条轨迹的 confidence
  - meta: json string — track_id, note_idx, pitch_hz, duration 等
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import mirdata
import numpy as np
import soundfile as sf

from analysis.tplib import _TRAJ_KEYS, event_trajectories

DATASETS = os.environ.get("QUILL_DATASETS", str(Path.home() / "Desktop/MT/datasets"))
GSET_HOME = DATASETS + "/mirdata_guitarset"
MIC_DIR = DATASETS + "/GuitarSet_mic"
OUT_PATH = DATASETS + "/guitarset_trajectories.npz"

ds = mirdata.initialize("guitarset", data_home=GSET_HOME)
solo_ids = sorted([t for t in ds.track_ids if "solo" in t])
print(f"Processing {len(solo_ids)} solo tracks")

all_features = []
all_trajectories = []
all_confidence = []
all_meta = []

for idx, tid in enumerate(solo_ids):
    track = ds.track(tid)
    mic_file = f"{MIC_DIR}/{tid}_mic.wav"
    try:
        y, sr = sf.read(mic_file, dtype="float32")
    except Exception as e:
        print(f"  [{idx + 1}/{len(solo_ids)}] SKIP {tid}: {e}")
        continue
    if y.ndim > 1:
        y = y.mean(axis=1)

    notes = track.notes_all
    if notes is None:
        print(f"  [{idx + 1}/{len(solo_ids)}] SKIP {tid}: no notes")
        continue

    intervals = notes.intervals
    pitches = notes.pitches
    n_extracted = 0

    # mirdata guitarset notes_all 的 pitch_unit == "midi"(音号,非 Hz)
    for i, (interval, pitch_midi) in enumerate(zip(intervals, pitches)):
        t_on, t_off = float(interval[0]), float(interval[1])
        dur = t_off - t_on
        if dur < 0.05:
            continue

        traj = event_trajectories(y, sr, t_on, t_off)

        traj_arr = np.stack([traj[k]["pts"] for k in _TRAJ_KEYS])
        conf_arr = np.array([traj[k]["confidence"] for k in _TRAJ_KEYS], dtype=np.float32)

        midi_pitch = float(pitch_midi)
        prev_interval = float(intervals[i - 1][1] - intervals[i - 1][0]) if i > 0 else 0.0
        prev_gap = float(t_on - intervals[i - 1][1]) if i > 0 else 1.0
        next_gap = float(intervals[i + 1][0] - t_off) if i < len(intervals) - 1 else 1.0
        is_first = 1.0 if i == 0 or prev_gap > 0.5 else 0.0
        is_last = 1.0 if i == len(intervals) - 1 or next_gap > 0.5 else 0.0

        feat = np.array(
            [
                midi_pitch,
                dur,
                prev_gap,
                next_gap,
                is_first,
                is_last,
            ],
            dtype=np.float32,
        )

        all_features.append(feat)
        all_trajectories.append(traj_arr)
        all_confidence.append(conf_arr)
        all_meta.append(
            {
                "track_id": tid,
                "note_idx": i,
                "pitch_hz": float(440.0 * 2 ** ((midi_pitch - 69) / 12)),
                "midi": float(midi_pitch),
                "t_on": t_on,
                "t_off": t_off,
                "dur": dur,
            }
        )
        n_extracted += 1

    print(f"  [{idx + 1}/{len(solo_ids)}] {tid}: {n_extracted} notes")

features = np.stack(all_features)
trajectories = np.stack(all_trajectories)
confidence = np.stack(all_confidence)
meta_json = json.dumps(all_meta)

np.savez_compressed(
    OUT_PATH,
    features=features,
    trajectories=trajectories,
    confidence=confidence,
    meta=meta_json,
    traj_keys=list(_TRAJ_KEYS),
)

print(f"\nSaved to {OUT_PATH}")
print(f"  Notes: {len(all_features)}")
print(f"  Features: {features.shape}")
print(f"  Trajectories: {trajectories.shape}")
print(f"  Confidence: {confidence.shape}")
print("  Avg confidence per key:")
for i, k in enumerate(_TRAJ_KEYS):
    print(f"    {k}: {confidence[:, i].mean():.3f}")
