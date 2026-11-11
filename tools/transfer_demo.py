"""跨乐器行为迁移 demo:行为(检索库+包络)与音色(DDSP)解耦的反事实组合。

2×2 矩阵,同一段 MIDI:
  vnB_vnT   小提琴行为 × 小提琴音色(本尊)
  gtB_gtT   吉他行为   × 吉他音色(本尊)
  gtB_vnT   吉他行为   × 小提琴音色 ← "小提琴被当吉他弹"(拨弦包络+吉他微观动态)
  vnB_gtT   小提琴行为 × 吉他音色   ← "吉他被当小提琴拉"(持续包络+揉弦)

叙事(ISMIR LBD):行为轨迹是乐器无关的表示,可作用于任何音色模型。
跑: cd quill && python tools/transfer_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from neural.neural_take import NeuralTake

OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "transfer_demo"

# 中速旋律:既有短音(吉他拨弦感清晰)也有长音(小提琴揉弦/持续感清晰)
MELODY = [
    {"midi": 62, "start": 0.0, "len": 0.4, "vel": 100},
    {"midi": 65, "start": 0.4, "len": 0.4, "vel": 105},
    {"midi": 69, "start": 0.8, "len": 0.8, "vel": 110},
    {"midi": 67, "start": 1.6, "len": 0.4, "vel": 100},
    {"midi": 65, "start": 2.0, "len": 0.4, "vel": 100},
    {"midi": 62, "start": 2.4, "len": 1.2, "vel": 108, "art": "vibrato"},
    # 乐句 2(隔 0.3s)
    {"midi": 69, "start": 3.9, "len": 0.4, "vel": 105},
    {"midi": 72, "start": 4.3, "len": 0.4, "vel": 110, "art": "slide"},
    {"midi": 74, "start": 4.7, "len": 1.6, "vel": 115, "art": "vibrato"},
]

COMBOS = [
    ("vnB_vnT", "violin", "violin"),
    ("gtB_gtT", "guitar", "guitar"),
    ("gtB_vnT", "guitar", "violin"),
    ("vnB_gtT", "violin", "guitar"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nt = NeuralTake()
    for name, behavior, timbre in COMBOS:
        t0 = time.time()
        y, sr = nt.render(MELODY, instrument=timbre, behavior=behavior)
        out = OUT_DIR / f"{name}.wav"
        sf.write(str(out), y, sr)
        print(
            f"  {name}: behavior={behavior:7s} timbre={timbre:7s} "
            f"{len(y) / sr:.2f}s in {time.time() - t0:.2f}s -> {out.name}"
        )
    print(f"\n=== Transfer Demo Complete -> {OUT_DIR} ===")
    print(
        "耳测: gtB_vnT 应有拨弦式起音+吉他微观动态但小提琴音色;vnB_gtT 应有持续包络+揉弦但吉他音色"
    )


if __name__ == "__main__":
    main()
