#!/usr/bin/env python3
"""Convert a raw MIDI-DDSP export to Quill's 7-feature / 4x48 schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural.behavior_distill import raw_teacher_to_dataset


def main():
    parser = argparse.ArgumentParser(description="build a Quill teacher dataset")
    parser.add_argument("raw", type=Path, help="export_mididdsp_teacher.py output")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--vibrato-threshold", type=float, default=0.05)
    parser.add_argument("--phrase-gap", type=float, default=0.5)
    args = parser.parse_args()
    result = raw_teacher_to_dataset(
        args.raw,
        args.output,
        vibrato_threshold=args.vibrato_threshold,
        phrase_gap=args.phrase_gap,
    )
    print(f"saved {len(result['features'])} teacher notes -> {args.output}")


if __name__ == "__main__":
    main()
