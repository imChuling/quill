"""Generate AB comparison audio for Plan D perceptual validation.

Condition 1: Quill retrieval vs distilled (same violin decoder)
Condition 2: Quill distilled on violin vs glass decoder

All pairs use identical MIDI, matched noise seed, and peak-normalized loudness.
Output: eval/plan_d_ab/<condition>_<melody>_<variant>.wav
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural.neural_take import NeuralTake

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

ART_NAMES = {0: "plain", 1: "vibrato", 2: "slide"}

OUT_DIR = Path(__file__).resolve().parents[1] / "eval" / "plan_d_ab"


def notes_from_melody(melody):
    notes = []
    t = 0.0
    for midi, dur, art in melody:
        notes.append(
            {
                "midi": midi,
                "start": t,
                "len": dur,
                "vel": 100,
                "art": ART_NAMES[art],
            }
        )
        t += dur
    return notes


def render_with_seed(nt, notes, instrument, behavior, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    audio, sr = nt.render(notes, instrument=instrument, behavior=behavior)
    return audio, sr


def write_wav(path, audio, sr):
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


def main():
    nt = NeuralTake()

    sources = nt.list_behavior_sources()
    distilled = [s for s in sources if s["name"] == "mididdsp_distilled"]
    if not distilled:
        print("ERROR: mididdsp_distilled not found. Check checkpoint.")
        sys.exit(1)
    print(f"Distilled behavior source: {distilled[0]}")

    has_glass_ckpt = Path(__file__).resolve().parents[1] / "checkpoints"
    glass_ckpt = list(has_glass_ckpt.glob("*glass*"))
    print(f"Glass DDSP available: {bool(glass_ckpt)}")

    seed = 42
    generated = []

    for mname, melody in MELODIES.items():
        notes = notes_from_melody(melody)
        print(f"\n--- {mname} ({len(notes)} notes) ---")

        # Condition 1a: violin retrieval (baseline)
        audio_ret, sr = render_with_seed(nt, notes, "violin", "violin", seed)
        path = OUT_DIR / f"c1_{mname}_retrieval_violin.wav"
        write_wav(path, audio_ret, sr)
        print(f"  retrieval/violin: {len(audio_ret) / sr:.2f}s -> {path.name}")
        generated.append(path)

        # Condition 1b: violin distilled
        audio_dist, sr = render_with_seed(nt, notes, "violin", "mididdsp_distilled", seed)
        path = OUT_DIR / f"c1_{mname}_distilled_violin.wav"
        write_wav(path, audio_dist, sr)
        print(f"  distilled/violin: {len(audio_dist) / sr:.2f}s -> {path.name}")
        generated.append(path)

        # Condition 2: distilled on glass (if available)
        if glass_ckpt:
            # 2a: retrieval on glass (baseline for cross-decoder comparison)
            audio_glass_ret, sr = render_with_seed(nt, notes, "glass", "violin", seed)
            path = OUT_DIR / f"c2_{mname}_retrieval_glass.wav"
            write_wav(path, audio_glass_ret, sr)
            print(f"  retrieval/glass:  {len(audio_glass_ret) / sr:.2f}s -> {path.name}")
            generated.append(path)

            # 2b: distilled on glass
            audio_glass, sr = render_with_seed(nt, notes, "glass", "mididdsp_distilled", seed)
            path = OUT_DIR / f"c2_{mname}_distilled_glass.wav"
            write_wav(path, audio_glass, sr)
            print(f"  distilled/glass:  {len(audio_glass) / sr:.2f}s -> {path.name}")
            generated.append(path)

    print(f"\nGenerated {len(generated)} files in {OUT_DIR}")

    # Manifest for easy review
    manifest = {
        "seed": seed,
        "conditions": {
            "c1": "retrieval vs distilled, same violin decoder",
            "c2": "distilled on violin vs glass decoder",
        },
        "files": [str(p.relative_to(OUT_DIR)) for p in generated],
    }
    import json

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("Wrote manifest.json")


if __name__ == "__main__":
    main()
