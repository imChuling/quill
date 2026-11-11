"""CLI: MIDI of any dialect -> Quill unified IR.

Run: conda run -n mt python tools/import_midi.py <file.mid> [--json out.json]

Prints the detection report and the converted score; with --json the machine-readable result is also saved.
Import only, no rendering (rendering goes through the existing transfer_demo / perform pipeline).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.midi_dialect import ART_NAMES, DrumFileError, import_midi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi_file")
    ap.add_argument("--json", help="save {score, report} as JSON")
    ap.add_argument("--bend-range", type=float, default=None, help="pitch-bend range (semitones), default ±2")
    args = ap.parse_args()

    try:
        score, rep = import_midi(args.midi_file, bend_range_st=args.bend_range)
    except DrumFileError as e:
        print(f"Rejected: {e}")
        sys.exit(2)

    print(
        f"Dialect: {rep.dialect} (confidence {rep.confidence:.2f}"
        + (f", alternatives {rep.candidates}" if rep.candidates else "")
        + ")"
    )
    print(f"Notes: {rep.n_notes_in} in -> {rep.n_notes_out} out")
    acts = []
    if rep.n_keyswitches_stripped:
        acts.append(f"stripped keyswitches ×{rep.n_keyswitches_stripped} (mapping {rep.ks_map_used})")
    if rep.n_channels_merged:
        acts.append(f"merged channels ×{rep.n_channels_merged}")
    if rep.n_bend_slide or rep.n_bend_vibrato:
        acts.append(f"bend→slide ×{rep.n_bend_slide}, bend→vibrato ×{rep.n_bend_vibrato}")
    if rep.n_chord_notes_dropped or rep.n_overlaps_trimmed:
        acts.append(f"removed chord notes ×{rep.n_chord_notes_dropped}, trimmed overlaps ×{rep.n_overlaps_trimmed}")
    if rep.n_drum_tracks_dropped:
        acts.append(f"dropped drum tracks ×{rep.n_drum_tracks_dropped}")
    if rep.cc_energy_source:
        acts.append(f"energy hint source {rep.cc_energy_source}")
    for a in acts:
        print(f"  - {a}")
    for w in rep.warnings:
        print(f"  ! {w}")

    print("\nscore (first 12 notes):")
    for n in score[:12]:
        extra = f"  cc_energy={n['cc_energy']}" if "cc_energy" in n else ""
        print(
            f"  onset={n['onset']:7.3f}  midi={n['midi']:3d}  "
            f"dur={n['dur']:.3f}  art={ART_NAMES[n['art']]}{extra}"
        )
    if len(score) > 12:
        print(f"  … {len(score)} notes in total")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"score": score, "report": rep.__dict__}, ensure_ascii=False, indent=2)
        )
        print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()
