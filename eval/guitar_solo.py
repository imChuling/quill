"""Solo 电吉他 demo —— DDSP Fender + 快起音延音模板 + 推弦/揉弦绽放的 blues 独奏。

配方(用户耳测定稿,6.23):
  - 音色:DDSP Fender Strat(clean)。
  - 包络:**14ms 干脆起音(凸曲线,punchy)+ 从峰值 tau≈3.5s 平滑指数衰减**。
    不用 note_template 的原始慢爬(172ms 才到峰 → "渐强又收"),也不压平地板(→ "断气")。
  - 演奏法:bend 170ms 快速到位(render_midi 默认)、揉弦绽放(solo 提前起振)。
用法:python -m eval.guitar_solo
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural.perform import FPS, note_template, render_midi, render_strum  # noqa: E402

FENDER = "assets/demos/ddsp_fender_strat.pt"
EM_CHORD = [52, 55, 59, 64]  # Em(E3 G3 B3 E4)
DATASETS = os.environ.get("QUILL_DATASETS", str(Path.home() / "Desktop/MT/datasets"))  # IDMT-SMT-Guitar lives under $QUILL_DATASETS
NOTE_GLOB = (
    DATASETS + "/IDMT-SMT-GUITAR_V2/"
    "dataset1/Fender Strat Clean Neck SC/audio/*.wav"
)

# blues 独奏:拨 → 推全音+揉(cry) → 勾弦 → 滑 → 长推弦+重揉(climax)
SOLO = [
    {"midi": 59, "dur": 0.45},
    {"midi": 62, "dur": 0.95, "art": "bend", "bend": 2, "vibrato": 40},
    {"midi": 59, "dur": 0.40},
    {"midi": 57, "dur": 0.40, "art": "legato"},
    {"midi": 55, "dur": 0.50},
    {"midi": 59, "dur": 0.50, "art": "slide"},
    {"midi": 62, "dur": 0.45},
    {"midi": 64, "dur": 1.90, "art": "bend", "bend": 2, "vibrato": 50},
]


def solo_template(note_src, tau=3.5, atk_ms=14, n_frames=900):
    """快起音 + 从峰值平滑指数衰减的 solo 响度模板(log-loudness)。

    tau   = 衰减时间常数(秒,越大越 sing);atk_ms = 起音时长(越小越脆)。
    取真实 Fender 音的峰值电平做参考,但**形状全合成**(避免模板慢爬/跌落伪影)。
    """
    peak = float(np.exp(note_template(note_src)).max())
    a = max(2, int(atk_ms / 1000.0 * FPS))
    out = np.empty(n_frames)
    out[:a] = peak * np.linspace(0, 1, a) ** 0.4  # 快起音(凸,punchy)
    tp = np.arange(n_frames - a) / FPS
    out[a:] = peak * np.exp(-tp / tau)  # 从峰值平滑衰减
    return np.log(out + 1e-6)


def _to44(y):
    return AF.resample(torch.tensor(np.asarray(y, np.float32))[None], 16000, 44100).numpy()[0]


def tail_fade(y, sr=44100, start_s=3.5):
    """尾部升余弦缓降 1→0,把 DDSP 余韵底噪平滑收到静音(保留前段自然衰减)。"""
    y = y.copy()
    t0 = min(len(y) - 1, int(start_s * sr))
    nf = len(y) - t0
    if nf > 1:
        y[t0:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, nf)))
    return y


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Solo 电吉他 demo（solo + 扫弦）")
    ap.add_argument("--tau", type=float, default=3.5, help="solo 衰减时间常数(秒)")
    args = ap.parse_args(argv)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    notes = sorted(glob.glob(NOTE_GLOB))
    if not notes:
        print("缺真实 Fender 音(IDMT-SMT-GUITAR)做峰值参考")
        return 1
    note = notes[0]

    # ① blues 独奏
    y = _to44(render_midi(FENDER, SOLO, solo_template(note, tau=args.tau), dev))
    sf.write(
        "assets/demos/guitar_solo.wav",
        (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32),
        44100,
    )
    print(f"  solo  -> guitar_solo.wav ({len(y) / 44100:.1f}s)")

    # ② Em 扫弦,完整衰减 + 尾部缓降收干净
    tmpl_long = solo_template(note, tau=2.2, n_frames=2200)
    yc = _to44(render_strum(FENDER, EM_CHORD, tmpl_long, dur=7.5, spread_ms=70, device=dev))
    yc = tail_fade(yc, 44100, start_s=3.5)
    sf.write(
        "assets/demos/guitar_strum.wav",
        (yc / (np.max(np.abs(yc)) + 1e-9) * 0.92).astype(np.float32),
        44100,
    )
    print(f"  strum -> guitar_strum.wav ({len(yc) / 44100:.1f}s, Em 完整衰减+收尾)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
