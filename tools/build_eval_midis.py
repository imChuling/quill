#!/usr/bin/env python3
"""Export the four eval melodies (lyrical / scale_run / leaps / folk) as MIDI files.

These feed export_mididdsp_teacher.py in the Docker environment so the same
note sequences drive both Quill and MIDI-DDSP, enabling a fair cross-comparison.

Usage:
    python tools/build_eval_midis.py -o checkpoints/eval/midis/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mido

ART_NAMES = {0: "plain", 1: "vibrato", 2: "slide"}

MELODIES = {
    "lyrical": [
        (62, 0.8, 1),
        (65, 0.4, 0),
        (67, 1.2, 1),
        (65, 0.4, 0),
        (69, 0.4, 2),
        (72, 1.6, 1),
        (67, 0.8, 0),
        (62, 1.6, 1),
    ],
    "scale_run": [
        (62, 0.2, 0),
        (64, 0.2, 0),
        (65, 0.2, 0),
        (67, 0.2, 0),
        (69, 0.2, 0),
        (71, 0.2, 0),
        (72, 0.2, 0),
        (74, 0.2, 0),
        (76, 0.4, 0),
        (74, 0.4, 0),
        (72, 1.6, 1),
    ],
    "leaps": [
        (57, 0.4, 0),
        (64, 0.4, 0),
        (62, 0.4, 2),
        (69, 0.8, 1),
        (65, 0.4, 0),
        (72, 0.4, 2),
        (71, 0.8, 0),
        (64, 1.2, 1),
    ],
    "folk": [
        (69, 0.4, 0),
        (71, 0.2, 0),
        (72, 0.4, 0),
        (69, 0.4, 0),
        (67, 0.8, 1),
        (64, 0.4, 0),
        (67, 0.4, 2),
        (69, 1.2, 1),
        (67, 0.4, 0),
        (64, 0.4, 0),
        (62, 1.6, 1),
    ],
}


def melody_to_midi(name: str, notes: list, tempo_bpm: float = 120.0) -> mido.MidiFile:
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    events = []
    t = 0.0
    for midi_note, dur_s, _art in notes:
        on_tick = int(round(t * 480 * tempo_bpm / 60.0))
        off_tick = int(round((t + dur_s) * 480 * tempo_bpm / 60.0))
        events.append((on_tick, "note_on", midi_note, 100))
        events.append((off_tick, "note_off", midi_note, 0))
        t += dur_s

    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

    prev_tick = 0
    for tick, msg_type, note, vel in events:
        delta = tick - prev_tick
        track.append(mido.Message(msg_type, note=note, velocity=vel, time=delta))
        prev_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def main():
    parser = argparse.ArgumentParser(description="build eval MIDI files from hardcoded melodies")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("checkpoints/eval/midis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for name, notes in MELODIES.items():
        mid = melody_to_midi(name, notes)
        path = args.output_dir / f"{name}.mid"
        mid.save(str(path))
        manifest_records.append({"midi": str(path), "name": name})
        total_dur = sum(d for _, d, _ in notes)
        print(f"  {name}: {len(notes)} notes, {total_dur:.1f}s -> {path}")

    import json

    manifest = {
        "schema": "quill.eval_midis.v1",
        "records": manifest_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest -> {manifest_path}")
    print("\nNext: run in MIDI-DDSP Docker container:")
    print("  python tools/export_mididdsp_teacher.py \\")
    print(f"    --manifest {manifest_path} \\")
    print("    --output-dir checkpoints/eval/mididdsp_exports/ \\")
    print("    --audio-dir checkpoints/eval/mididdsp_exports/audio/ \\")
    print("    --instrument violin --seed 42 --resume")


if __name__ == "__main__":
    main()
