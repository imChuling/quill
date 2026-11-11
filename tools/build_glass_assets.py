"""Glass 双资产构建(可溯源):13 段 studio wav → 拼接 → style library。

跑: conda run -n mt python tools/build_glass_assets.py
产出:
  assets/glass/glass_perf_studio.wav        拼接音频(0.5s 静音间隔)
  assets/glass/glass_concat_manifest.json   逐文件时间轴(溯源)
  checkpoints/glass_style.npz               24-note style library

Timbre 模型训练(同一批文件):
  python -W ignore -m neural.train_ddsp --glob "assets/glass/timbre_train/*.wav" \
      --name glass --steps 6000 --out assets/glass/tracer2
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

QUILL = Path(__file__).resolve().parent.parent
TRAIN_DIR = QUILL / "assets/glass/timbre_train"
OUT_WAV = QUILL / "assets/glass/glass_perf_studio.wav"
MANIFEST = QUILL / "assets/glass/glass_concat_manifest.json"
OUT_NPZ = QUILL / "checkpoints/glass_style.npz"

SR = 44100
GAP_S = 0.5


def main():
    import librosa
    import soundfile as sf

    files = sorted(TRAIN_DIR.glob("*.wav"))
    gap = np.zeros(int(GAP_S * SR), np.float32)
    segs, manifest, t = [], [], 0.0
    for f in files:
        y, _ = librosa.load(str(f), sr=SR, mono=True)
        y = y / (np.abs(y).max() + 1e-9) * 0.8
        manifest.append(
            {"file": f.name, "start_s": round(t, 3), "end_s": round(t + len(y) / SR, 3)}
        )
        segs.append(y)
        segs.append(gap)
        t += len(y) / SR + GAP_S
    out = np.concatenate(segs)
    sf.write(str(OUT_WAV), out, SR)
    with open(str(MANIFEST), "w") as fh:
        json.dump(
            {
                "sr": SR,
                "gap_s": GAP_S,
                "n_files": len(files),
                "total_s": round(len(out) / SR, 1),
                "segments": manifest,
            },
            fh,
            indent=2,
        )
    print(f"{len(files)} files -> {OUT_WAV.name} ({len(out) / SR:.1f}s)")
    print(f"manifest -> {MANIFEST.name}")

    r = subprocess.run(
        [sys.executable, str(QUILL / "tools/extract_style.py"), str(OUT_WAV), "-o", str(OUT_NPZ)]
    )
    if r.returncode:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
