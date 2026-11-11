#!/usr/bin/env python3
"""Export MIDI-DDSP teacher controls without importing Quill or PyTorch.

Run this script in the isolated Python 3.8 / TensorFlow MIDI-DDSP environment.
Its output is a safe NumPy archive consumed by ``build_mididdsp_teacher.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import wave
from pathlib import Path

import numpy as np

RAW_SCHEMA = "quill.mididdsp_teacher.raw.v1"
FRAME_RATE = 250.0
EXPRESSION_KEYS = (
    "volume",
    "vol_fluc",
    "vibrato",
    "brightness",
    "attack",
    "vol_peak_pos",
)
NOTE_KEYS = ("pitch", "onset", "offset", "note_length") + EXPRESSION_KEYS


def _set_seed(seed):
    """Reset every RNG used by the released TensorFlow teacher."""
    import tensorflow as tf

    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def _source_seed(path, base_seed):
    """Derive a stable per-score seed from bytes, independent of batch order."""
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("ascii"))
    digest.update(Path(path).read_bytes())
    return int.from_bytes(digest.digest()[:8], "big") % (2**31 - 1)


def _numpy(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    value = np.asarray(value)
    if value.ndim >= 2 and value.shape[0] == 1:
        value = value[0]
    return value.astype(np.float32)


def _named_controls(midi_control_params, midi_synth_params):
    """Handle both the released tuple API and later dictionary variants."""
    keys = ("f0_hz", "amplitudes", "harmonic_distribution", "noise_magnitudes")
    if isinstance(midi_control_params, dict) and set(keys) <= set(midi_control_params):
        return {key: _numpy(midi_control_params[key]) for key in keys}
    if isinstance(midi_control_params, (tuple, list)) and len(midi_control_params) >= 4:
        # Official release order (documented in midi_ddsp_synthesize.py:190-193).
        return {
            "f0_hz": _numpy(midi_control_params[0]),
            "amplitudes": _numpy(midi_control_params[1]),
            "harmonic_distribution": _numpy(midi_control_params[2]),
            "noise_magnitudes": _numpy(midi_control_params[3]),
        }
    synth = midi_synth_params
    if isinstance(synth, dict) and "inputs" in synth:
        synth = synth["inputs"]
    if isinstance(synth, dict) and set(keys) <= set(synth):
        return {key: _numpy(synth[key]) for key in keys}
    raise TypeError("unrecognized MIDI-DDSP control output; expected tuple[4] or named dict")


def _write_audio(path, audio, sample_rate=16000):
    """Write a dependency-free 16-bit listening copy (not used for training)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = max(float(np.max(np.abs(audio))), 1e-8)
    pcm = np.clip(audio / max(1.0, peak), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def load_runtime(args):
    _set_seed(args.seed)
    try:
        from midi_ddsp.data_handling.instrument_name_utils import INST_NAME_TO_ID_DICT
        from midi_ddsp.midi_ddsp_synthesize import load_pretrained_model
        from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi
    except ImportError as exc:
        raise SystemExit(
            "MIDI-DDSP is unavailable. Run this exporter in its isolated Python 3.8 "
            "environment after `pip install midi-ddsp` and "
            "`midi_ddsp_download_model_weights`."
        ) from exc

    if args.instrument not in INST_NAME_TO_ID_DICT:
        names = ", ".join(sorted(INST_NAME_TO_ID_DICT))
        raise SystemExit(f"unsupported MIDI-DDSP instrument {args.instrument!r}; choose: {names}")

    synthesis_generator, expression_generator = load_pretrained_model(
        synthesis_generator_path=args.synthesis_weights,
        expression_generator_path=args.expression_weights,
    )
    instrument_id = int(INST_NAME_TO_ID_DICT[args.instrument])
    return synthesis_generator, expression_generator, instrument_id, synthesize_mono_midi


def export(args, runtime=None):
    if runtime is None:
        runtime = load_runtime(args)
    synthesis_generator, expression_generator, instrument_id, synthesize_mono_midi = runtime
    source_seed = _source_seed(args.midi, args.seed)
    _set_seed(source_seed)
    audio, control_params, synth_params, conditioning_df = synthesize_mono_midi(
        synthesis_generator,
        expression_generator,
        str(args.midi),
        instrument_id,
        output_dir=None,
        display_progressbar=not args.quiet,
    )
    controls = _named_controls(control_params, synth_params)

    missing = [key for key in NOTE_KEYS if key not in conditioning_df.columns]
    if missing:
        raise ValueError(f"MIDI-DDSP conditioning table is missing columns: {missing}")
    note_table = conditioning_df.loc[:, list(NOTE_KEYS)].to_numpy(dtype=np.float32)
    note_table = note_table[note_table[:, 0] > 0]
    if len(note_table) == 0:
        raise ValueError("MIDI-DDSP produced no pitched notes")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_midi": str(Path(args.midi).resolve()),
        "instrument": args.instrument,
        "instrument_id": instrument_id,
        "midi_ddsp_version": importlib.metadata.version("midi-ddsp"),
        "synthesis_weights": args.synthesis_weights or "package-default",
        "expression_weights": args.expression_weights or "package-default",
        "base_seed": int(args.seed),
        "source_seed": int(source_seed),
        "exporter": "Quill Plan D",
    }
    np.savez_compressed(
        output,
        schema=np.asarray(RAW_SCHEMA),
        frame_rate=np.asarray(FRAME_RATE, dtype=np.float32),
        note_keys=np.asarray(NOTE_KEYS),
        note_table=note_table,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
        **controls,
    )
    if args.audio_output:
        _write_audio(args.audio_output, _numpy(audio), sample_rate=16000)
    if not args.quiet:
        print(f"saved {len(note_table)} notes and {len(controls['f0_hz'])} frames -> {output}")


def _valid_raw(path):
    """Only resume from an archive that satisfies the complete raw contract."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema",
                "frame_rate",
                "note_keys",
                "note_table",
                "meta",
                "f0_hz",
                "amplitudes",
                "harmonic_distribution",
                "noise_magnitudes",
            }
            if not required <= set(data.files) or str(data["schema"]) != RAW_SCHEMA:
                return False
            n_frames = len(data["f0_hz"])
            if n_frames == 0 or len(data["note_table"]) == 0:
                return False
            for key in ("f0_hz", "amplitudes", "harmonic_distribution", "noise_magnitudes"):
                value = data[key]
                if len(value) != n_frames or not np.isfinite(value).all():
                    return False
            return np.isfinite(data["note_table"]).all()
    except (OSError, ValueError, KeyError):
        return False


def export_manifest(args):
    manifest = json.loads(Path(args.manifest).read_text())
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must contain a non-empty records list")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    if audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime(args)
    completed = []
    for index, record in enumerate(records, start=1):
        midi = Path(record["midi"])
        output = output_dir / f"{midi.stem}.npz"
        if args.resume and _valid_raw(output):
            print(f"[{index}/{len(records)}] validated existing {output}", flush=True)
        else:
            one = argparse.Namespace(**vars(args))
            one.midi = midi
            one.output = output
            one.audio_output = audio_dir / f"{midi.stem}.wav" if audio_dir else None
            one.quiet = True
            export(one, runtime=runtime)
            if not _valid_raw(output):
                raise RuntimeError(f"post-write validation failed: {output}")
            print(f"[{index}/{len(records)}] saved {output}", flush=True)
        completed.append(str(output))

    summary = {
        "schema": "quill.mididdsp_teacher.batch.v1",
        "source_manifest": str(Path(args.manifest).resolve()),
        "instrument": args.instrument,
        "base_seed": int(args.seed),
        "outputs": completed,
    }
    summary_path = Path(args.summary) if args.summary else output_dir / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"batch complete: {len(completed)} validated archives -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="export direct MIDI-DDSP teacher controls")
    parser.add_argument("--midi", type=Path, help="monophonic MIDI input")
    parser.add_argument("--manifest", type=Path, help="teacher MIDI manifest for batch export")
    parser.add_argument("--instrument", default="violin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="raw teacher .npz")
    parser.add_argument("--output-dir", type=Path, help="batch raw-teacher directory")
    parser.add_argument("--audio-output", type=Path, help="optional listening WAV")
    parser.add_argument("--audio-dir", type=Path, help="optional batch listening-WAV directory")
    parser.add_argument("--summary", type=Path, help="batch summary JSON path")
    parser.add_argument(
        "--resume", action="store_true", help="skip only contract-valid batch outputs"
    )
    parser.add_argument("--synthesis-weights")
    parser.add_argument("--expression-weights")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        if args.midi or args.output or not args.output_dir:
            parser.error("batch mode requires --manifest and --output-dir, without --midi/--output")
        export_manifest(args)
    else:
        if not args.midi or not args.output:
            parser.error("single-file mode requires --midi and --output")
        export(args)


if __name__ == "__main__":
    main()
