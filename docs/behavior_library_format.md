# Quill Behavior Library Format (v1.0)

A behavior library is an `.npz` file containing note-level performance trajectories extracted from real recordings. Each note is represented as four 48-frame trajectories (pitch, energy, brightness, noise) plus a 6-dimensional context feature vector used for retrieval.

## File structure

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `trajectories` | `(N, 4, 48)` | float32 | Performance trajectories, time-normalized to 48 frames per note |
| `features` | `(N, 6)` or `(N, 7)` | float32 | Context vectors for nearest-neighbor retrieval |
| `confidence` | `(N, 4)` | float32 | Per-dimension confidence (0–1), from extraction quality |
| `meta` | scalar | str (JSON) | Per-note metadata as JSON array |
| `traj_keys` | `(4,)` | str | Trajectory dimension names: `['pitch', 'energy', 'brightness', 'noise']` |
| `format_version` | scalar | str | `"1.0"` |
| `fps` | scalar | int32 | Frame rate: `250` (frames correspond to 4 ms each at playback) |
| `source` | scalar | str | Dataset or recording source description |
| `license` | scalar | str | License terms |
| `traj_units` | `(4,)` | str | Units per trajectory dimension |

## Trajectory dimensions

| Dim | Key | Unit | Description |
|---|---|---|---|
| 0 | `pitch` | cents from median | Micro-pitch deviation (vibrato, drift, scoop) |
| 1 | `energy` | z-score | Loudness envelope shape (attack, decay, sustain) |
| 2 | `brightness` | log2 residual | Spectral centroid deviation from expected |
| 3 | `noise` | log10 residual | Spectral flatness deviation from expected |

All trajectories are time-normalized to 48 frames regardless of note duration. At playback, they are linearly interpolated to the actual note length at 250 fps.

## Context features (6-D or 7-D)

| Idx | Name | Description |
|---|---|---|
| 0 | `midi` | MIDI pitch (fractional, from pYIN) |
| 1 | `dur` | Duration in seconds |
| 2 | `prev_gap` | Gap before this note (seconds; 1.0 if first note) |
| 3 | `next_gap` | Gap after this note (seconds; 1.0 if last note) |
| 4 | `is_first` | 1.0 if first note in phrase, else 0.0 |
| 5 | `is_last` | 1.0 if last note in phrase, else 0.0 |
| 6 | `art` | Articulation label (optional; 0=plain, 1=vibrato, 2=slide) |

Retrieval uses Euclidean distance on the feature vector. When `art` is present (7-D), it conditions retrieval to the same-label pool.

## Articulation labels

| Label | Value | Detection criteria |
|---|---|---|
| plain | 0 | No significant pitch modulation in 3–9 Hz band |
| vibrato | 1 | 3–9 Hz pitch modulation energy > threshold AND rate in 3.5–9.5 Hz |
| slide | 2 | Net pitch drift > 60 cents (portamento / glissando) |

Labels are assigned by automatic analysis (pYIN pitch extraction + FFT vibrato detection). They are not human-verified.

## Per-note metadata (JSON)

Each entry in the `meta` JSON array contains:

| Field | Type | Description |
|---|---|---|
| `track` or `track_id` | str | Source recording identifier |
| `note_idx` | int | Note index within the track |
| `midi` | float | Detected MIDI pitch |
| `dur` | float | Duration in seconds |
| `art` | int | Articulation label (0/1/2) — may be absent |

## Loading example

```python
import numpy as np

lib = np.load("checkpoints/urmp_vn_trajectories.npz", allow_pickle=True)
assert str(lib["format_version"]) == "1.0"

trajs = lib["trajectories"]   # (N, 4, 48)
feats = lib["features"]       # (N, 7)
fps = int(lib["fps"])          # 250

# Get the pitch trajectory of note 42
pitch_cents = trajs[42, 0, :]  # 48 frames, cents from median

# Stretch to actual note duration
import json
meta = json.loads(str(lib["meta"]))
dur = meta[42]["dur"]
n_frames = int(dur * fps)
pitch_stretched = np.interp(
    np.linspace(0, 1, n_frames),
    np.linspace(0, 1, 48),
    pitch_cents,
)
```

## Distribution of the bundled libraries

Only `glass_style.npz` (self-recorded, MIT) is published. `urmp_vn_trajectories.npz` and
`urmp_tpt_trajectories.npz` are derived from URMP, which has no standard license and
requires registration and citation; they stay repository-internal for paper reproduction
and are not part of any release. See `checkpoints/MODEL_CARDS.md` for the policy and
`tools/extract_trajectories_urmp.py` to rebuild them from a registered URMP download.

## Playback

See `tools/behavior_player.py` for a minimal player that takes a score (MIDI pitch, duration, articulation) and a library, retrieves matching trajectories by nearest-neighbor lookup, and outputs 250 fps control signals (f0 in Hz, loudness in z-score).
