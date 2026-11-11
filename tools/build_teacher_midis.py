#!/usr/bin/env python3
"""Build a source-disjoint monophonic MIDI query set from GuitarSet solos.

The transform is deterministic and records every selected source and decision in
``manifest.json``.  Overlapping guitar-note annotations are trimmed to the next
onset; near-simultaneous notes are resolved to the higher pitch so MIDI-DDSP's
monophonic expression generator never receives chords.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pretty_midi


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_notes(path: Path, min_duration=0.06):
    payload = json.loads(path.read_text(encoding="utf-8"))
    notes = []
    for annotation in payload.get("annotations", []):
        if annotation.get("namespace") != "note_midi":
            continue
        for item in annotation.get("data", []):
            start = float(item["time"])
            duration = float(item["duration"])
            pitch = int(round(float(item["value"])))
            if duration >= min_duration and 21 <= pitch <= 108:
                notes.append([start, start + duration, pitch])
    notes.sort(key=lambda note: (note[0], -note[2], note[1]))

    # Collapse near-simultaneous annotations to one melodic note.
    collapsed = []
    for note in notes:
        if collapsed and note[0] - collapsed[-1][0] <= 0.01:
            if note[2] > collapsed[-1][2]:
                collapsed[-1] = note
        else:
            collapsed.append(note)

    mono = []
    for note in collapsed:
        if mono and note[0] < mono[-1][1]:
            mono[-1][1] = note[0]
            if mono[-1][1] - mono[-1][0] < min_duration:
                mono.pop()
        mono.append(note)
    return [note for note in mono if note[1] - note[0] >= min_duration]


def _select_sources(paths, per_player, seed):
    grouped = {}
    for path in paths:
        player = path.name.split("_", 1)[0]
        grouped.setdefault(player, []).append(path)
    rng = np.random.default_rng(seed)
    selected = []
    for player in sorted(grouped):
        choices = sorted(grouped[player])
        if len(choices) < per_player:
            raise ValueError(f"player {player} has only {len(choices)} solo files")
        idx = np.sort(rng.choice(len(choices), size=per_player, replace=False))
        selected.extend(choices[int(i)] for i in idx)
    return selected


def main():
    parser = argparse.ArgumentParser(description="build Plan-D teacher MIDI queries")
    parser.add_argument("--guitarset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-player", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-notes", type=int, default=24)
    args = parser.parse_args()

    sources = sorted(args.guitarset.glob("*_solo.jams"))
    selected = _select_sources(sources, args.per_player, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for source in selected:
        notes = _read_notes(source)
        if len(notes) < args.min_notes:
            raise ValueError(f"{source.name}: only {len(notes)} monophonic notes")
        # Remove leading silence while retaining 100 ms model warm-up.
        shift = max(0.0, notes[0][0] - 0.1)
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        instrument = pretty_midi.Instrument(program=40, name="violin_teacher_query")
        for start, end, pitch in notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=96,
                    pitch=pitch,
                    start=start - shift,
                    end=end - shift,
                )
            )
        midi.instruments.append(instrument)
        output = args.output / f"{source.stem}.mid"
        midi.write(str(output))
        records.append(
            {
                "source": str(source.resolve()),
                "source_sha256": _sha256(source),
                "midi": str(output.resolve()),
                "notes": len(notes),
                "duration_s": round(max(note[1] for note in notes) - shift, 3),
                "player": source.name.split("_", 1)[0],
            }
        )

    manifest = {
        "schema": "quill.plan_d.teacher_queries.v1",
        "seed": args.seed,
        "per_player": args.per_player,
        "selection": "uniform count per GuitarSet player; seeded without replacement",
        "monophonic_rule": "within 10 ms keep higher pitch; trim previous note to next onset",
        "velocity": 96,
        "instrument_program": 40,
        "records": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(records)} MIDI queries -> {args.output}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
