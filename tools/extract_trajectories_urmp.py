"""Rebuild the URMP-derived behavior libraries (violin / trumpet) from URMP stems.

The two URMP libraries shipped in `checkpoints/` are NOT redistributed (see
`checkpoints/MODEL_CARDS.md`, "Distribution policy"). To rebuild them, register for
URMP, download the dataset, then run:

    python tools/extract_trajectories_urmp.py --inst vn  --urmp /path/to/URMP \\
        -o checkpoints/urmp_vn_trajectories.npz
    python tools/extract_trajectories_urmp.py --inst tpt --urmp /path/to/URMP \\
        -o checkpoints/urmp_tpt_trajectories.npz

Then add the v1.0 metadata fields:

    python tools/upgrade_library_format.py

Provenance. This is the script that produced the libraries used in the ISMIR LBD 2026
paper (first committed as the server-side violin script in 51fd779, removed in the
370da7c cleanup, restored here). Changes relative to that version are limited to:
  * instrument selected by `--inst` (the trumpet library was built with a copy whose
    only difference was the `_tpt_` filename glob), paths taken from the command line
    instead of hard-coded server paths, and v1.0 metadata written directly;
  * no change to note filtering, feature construction, trajectory extraction or the
    articulation classifier. `tests/test_urmp_extraction_rules.py` asserts that these
    rules reproduce 100% of the labels and context features stored in the shipped
    libraries.

URMP layout (one folder per piece):
    AuSep_<track>_<inst>_<piece>.wav   separated stem, 48 kHz mono
    Notes_<track>_<inst>_<piece>.txt   note annotations: onset(s)  pitch(Hz)  duration(s)

Output npz (same keys as `docs/behavior_library_format.md`):
    features(N,7): midi, dur, prev_gap, next_gap, is_first, is_last, articulation
    trajectories(N,4,48), confidence(N,4), meta(json), traj_keys,
    format_version, source, license, fps, traj_units
    articulation: 0=plain, 1=vibrato, 2=slide
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.tplib import _TRAJ_KEYS, event_trajectories  # noqa: E402

ART_PLAIN, ART_VIBRATO, ART_SLIDE = 0, 1, 2
MIN_NOTE_DUR = 0.05  # notes shorter than this (s) are dropped
PHRASE_GAP = 0.5  # a rest longer than this (s) starts / ends a phrase
FPS = 250
FORMAT_VERSION = "1.0"

INST_NAMES = {"vn": "violin", "tpt": "trumpet"}
SOURCE = (
    "URMP {inst} (University of Rochester Multi-Modal Music Performance dataset), "
    "separated stems + note annotations"
)
LICENSE = (
    "Derived trajectories (no audio); source dataset license unconfirmed — requires "
    "registration and citation. Not redistributed by the Quill project."
)


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.maximum(f, 1e-6) / 440.0)


def load_notes(txt_path):
    """Notes_*.txt: onset(s), pitch(Hz), duration(s), whitespace separated."""
    notes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            t_on, hz, dur = float(parts[0]), float(parts[1]), float(parts[2])
            notes.append((t_on, dur, float(hz_to_midi(hz))))
    notes.sort(key=lambda x: x[0])
    return notes


def classify_articulation(pitch_traj_cents, confidence):
    """Rule-based plain / vibrato / slide label.

    pitch_traj_cents: 48 points, cents relative to the note's median pitch.
    Identical to `tools/extract_style.py::classify_articulation`.
    """
    if confidence < 0.5:
        return ART_PLAIN  # untrusted pitch track: never label noise as a slide
    p = pitch_traj_cents
    rng = float(np.max(p) - np.min(p))
    if rng < 30.0:
        return ART_PLAIN
    # net displacement vs oscillation: slide = one-way, vibrato = back-and-forth
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


def context_features(notes, i, art):
    """6-D context + articulation for note i of a stem's note list.

    notes: [(t_on, dur, midi)] sorted by onset (all notes, before the MIN_NOTE_DUR
    filter, so gaps are measured against the real neighbours).
    """
    t_on, dur, midi = notes[i]
    t_off = t_on + dur
    prev_gap = float(t_on - (notes[i - 1][0] + notes[i - 1][1])) if i > 0 else 1.0
    next_gap = float(notes[i + 1][0] - t_off) if i < len(notes) - 1 else 1.0
    is_first = 1.0 if i == 0 or prev_gap > PHRASE_GAP else 0.0
    is_last = 1.0 if i == len(notes) - 1 or next_gap > PHRASE_GAP else 0.0
    return np.array([midi, dur, prev_gap, next_gap, is_first, is_last, float(art)], dtype=np.float32)


def extract_stem(wav_path, notes_path):
    """One stem → (features, trajectories, confidence, meta) lists."""
    y, sr = sf.read(str(wav_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    notes = load_notes(str(notes_path))
    feats, trajs, confs, meta = [], [], [], []
    for i, (t_on, dur, midi) in enumerate(notes):
        if dur < MIN_NOTE_DUR:
            continue
        traj = event_trajectories(y, sr, t_on, t_on + dur)
        traj_arr = np.stack([traj[k]["pts"] for k in _TRAJ_KEYS])
        conf_arr = np.array([traj[k]["confidence"] for k in _TRAJ_KEYS], dtype=np.float32)
        art = classify_articulation(traj_arr[0], float(conf_arr[0]))
        feats.append(context_features(notes, i, art))
        trajs.append(traj_arr)
        confs.append(conf_arr)
        meta.append({"track": wav_path.stem, "note_idx": i, "midi": midi, "dur": dur, "art": art})
    return feats, trajs, confs, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inst", choices=sorted(INST_NAMES), required=True, help="URMP instrument code")
    ap.add_argument("--urmp", required=True, help="URMP root (searched recursively for AuSep_*_<inst>_*.wav)")
    ap.add_argument("-o", "--out", required=True, help="output .npz")
    args = ap.parse_args()

    sep_files = sorted(Path(args.urmp).rglob(f"AuSep_*_{args.inst}_*.wav"))
    print(f"Found {len(sep_files)} {INST_NAMES[args.inst]} stems under {args.urmp}")
    if not sep_files:
        sys.exit(1)

    all_f, all_t, all_c, all_m = [], [], [], []
    for n_done, wav_path in enumerate(sep_files, 1):
        notes_path = wav_path.parent / wav_path.name.replace("AuSep_", "Notes_").replace(".wav", ".txt")
        if not notes_path.exists():
            print(f"  SKIP {wav_path.name}: no Notes txt")
            continue
        f, t, c, m = extract_stem(wav_path, notes_path)
        if not f:
            print(f"  SKIP {wav_path.name}: no usable notes")
            continue
        all_f += f
        all_t += t
        all_c += c
        all_m += m
        print(f"  [{n_done}] {wav_path.stem}: {len(f)} notes (total {len(all_f)})")

    features = np.stack(all_f)
    trajectories = np.stack(all_t)
    confidence = np.stack(all_c)
    np.savez_compressed(
        args.out,
        features=features,
        trajectories=trajectories,
        confidence=confidence,
        meta=json.dumps(all_m),
        traj_keys=list(_TRAJ_KEYS),
        format_version=np.array(FORMAT_VERSION),
        source=np.array(SOURCE.format(inst=INST_NAMES[args.inst])),
        license=np.array(LICENSE),
        fps=np.array(FPS, dtype=np.int32),
        traj_units=np.array(["cents_from_median", "z_score", "log2_residual", "log10_residual"]),
    )

    n = len(all_f)
    counts = np.bincount(features[:, 6].astype(int), minlength=3)
    print(f"\nSaved {args.out}")
    print(f"  Notes: {n}, features={features.shape}, traj={trajectories.shape}")
    print(
        f"  Articulation: plain={counts[0]} ({counts[0] / n:.1%}), "
        f"vibrato={counts[1]} ({counts[1] / n:.1%}), slide={counts[2]} ({counts[2] / n:.1%})"
    )
    for i, k in enumerate(_TRAJ_KEYS):
        print(f"  {k} avg confidence: {confidence[:, i].mean():.3f}")


if __name__ == "__main__":
    main()
