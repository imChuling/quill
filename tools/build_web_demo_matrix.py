"""Prerender the web demo's timbre × behavior matrix.

    python tools/build_web_demo_matrix.py            # → web_demo/audio/*.wav + manifest.json

4 timbres × 4 behavior libraries × 2 melodies (lyrical, folk) = 32 clips, rendered with the
same NeuralTake.render path and UI defaults the desktop app uses. Each clip is 16 kHz mono WAV.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_quill import MELODIES  # noqa: E402

from neural.neural_take import NeuralTake, available_instruments  # noqa: E402

OUT = ROOT / "web_demo" / "audio"
TIMBRES = ["violin", "trumpet", "glass"]  # guitar DDSP excluded: plucked-envelope quality (same as the paper's audio-level eval)
BEHAVIORS = ["violin", "trumpet", "guitar", "glass"]
MELODY_NAMES = ["lyrical", "folk"]
ART = {0: "pluck", 1: "vibrato", 2: "slide"}


def melody_notes(name):
    t, out = 0.0, []
    for midi, dur, art in MELODIES[name]:
        out.append({"midi": midi, "start": round(t, 4), "len": dur, "vel": 100, "art": ART[art]})
        t += dur
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.wav"):
        stale.unlink()
    avail = available_instruments()
    missing = [t for t in TIMBRES if t not in avail]
    if missing:
        sys.exit(f"missing checkpoints for {missing}")
    nt = NeuralTake(device="cpu")
    manifest = {"melodies": {}, "timbres": TIMBRES, "behaviors": BEHAVIORS, "clips": []}
    for mel in MELODY_NAMES:
        notes = melody_notes(mel)
        manifest["melodies"][mel] = {"notes": notes, "dur_s": round(sum(n["len"] for n in notes), 2)}
        for tim in TIMBRES:
            for beh in BEHAVIORS:
                t0 = time.time()
                y, sr = nt.render(notes, instrument=tim, behavior=beh, tail_s=0.4)
                y = np.asarray(y, np.float32)
                peak = float(np.max(np.abs(y))) or 1.0
                y = y / peak * 0.89  # same loudness across the grid
                fn = f"{mel}_{tim}T_{beh}B.wav"
                sf.write(OUT / fn, y, sr)
                manifest["clips"].append({"melody": mel, "timbre": tim, "behavior": beh, "file": f"audio/{fn}", "dur_s": round(len(y) / sr, 2)})
                print(f"{fn:34s} {len(y) / sr:5.2f}s  ({time.time() - t0:.1f}s)")
    (ROOT / "web_demo" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in OUT.glob("*.wav")) / 1e6
    print(f"\n{len(manifest['clips'])} clips, {total:.1f} MB → {OUT}")


if __name__ == "__main__":
    main()
