"""7.8 英雄 demo —— 同一段乐句 × 多取色音色,**按八度对齐到各音色自然音区**。

修正 fr9 草版的硬伤:固定绝对音区 → 每音色被强行移调 [-28,+17] 半音,共振峰外推塌掉。
这里每个音色把乐句**整体平移整数个八度**到最接近其 f0_ref 的音区(同旋律/同演奏法、
移调幅度最小 → 每个音色都在自己的甜区发声;也更像真实多乐器编曲:贝斯在低、笛在高)。

用法:python -m eval.hero_demo
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.tplib import load_tplib  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

SR = 44100
DEMO = Path(__file__).resolve().parents[1] / "assets" / "demos"

# 较窄(~1 八度)、悦耳、且充分用到演奏法的乐句。中心 ~MIDI 57。
PHRASE = [
    (0.20, "on", 55, 100),
    (0.55, "on", 60, 95),  # +5 -> slide
    (1.05, "on", 58, 85),  # -2 -> legato
    (1.45, "off", 58, 0),
    (1.45, "off", 60, 0),
    (1.45, "off", 55, 0),
    (1.65, "on", 53, 100),
    (2.0, "on", 60, 110),  # +7 -> slide(快)
    (2.6, "off", 60, 0),
    (2.6, "off", 53, 0),
    (2.85, "on", 53, 100),
    (2.87, "on", 57, 100),
    (2.89, "on", 60, 100),  # strum 和弦
    (3.1, "cc", -1, 100),  # 颤音
    (3.9, "off", 53, 0),
    (3.9, "off", 57, 0),
    (3.9, "off", 60, 0),
    (4.1, "on", 62, 100),
    (4.45, "on", 57, 95),  # -5 下滑
    (5.0, "off", 57, 0),
    (5.0, "off", 62, 0),
]
DUR = 5.3
PHRASE_CENTER = float(np.mean([e[2] for e in PHRASE if e[1] in ("on",) and e[2] > 0]))


def octave_align(phrase, f0_ref):
    """把乐句整体平移整数个八度,使其中心最接近快照自然音区(移调最小)。"""
    midi_ref = 69 + 12 * np.log2(max(f0_ref, 1.0) / 440.0)
    shift = int(round((midi_ref - PHRASE_CENTER) / 12.0)) * 12
    out = []
    for t, typ, m, v in phrase:
        out.append((t, typ, (m + shift if typ in ("on", "off") and m > 0 else m), v))
    return out, shift


def render(snap, lib, noise=0.2):
    align, shift = octave_align(PHRASE, snap["f0_ref"])
    inst = build_instrument(snap, noise_level=noise)
    eng = Engine(inst, eval_mode=True)
    gi = GestureInterpreter(library=lib)
    gi.apply_to_engine(eng)
    cooked = gi.process(align)
    y = eng.render_offline(cooked, DUR)
    # 报告移调幅度(诊断:应都落在 ±9 内)
    notes = [m for _, typ, m, _ in align if typ == "on" and m > 0]
    midi_ref = round(69 + 12 * np.log2(max(snap["f0_ref"], 1.0) / 440.0))
    red = [min(notes) - midi_ref, max(notes) - midi_ref]
    return y, shift, red


def main(argv=None):
    DEMO.mkdir(parents=True, exist_ok=True)
    lib = load_tplib("assets/libraries/guitar.tplib")
    timbres = [
        ("hero2_guitar", "assets/snapshots/guitar.npz", 0.30, "吉他(锚)"),
        ("hero2_voice", "assets/snapshots/voice_tone_mid.npz", 0.20, "人声"),
        ("hero2_pad", "assets/snapshots/synth_pad.npz", 0.15, "合成pad"),
        ("hero2_wineglass", "assets/snapshots/wineglass.npz", 0.25, "酒杯"),
    ]
    clips = []
    print("音色            八度移调   乐句相对自然音区   clean")
    for name, path, noise, label in timbres:
        snap = load_snapshot(path)
        y, shift, red = render(snap, lib, noise=noise)
        clean = bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0)
        sf.write(
            DEMO / f"{name}.wav", (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32), SR
        )
        print(f"  {label:10s}  {shift:+4d}半音   [{red[0]:+d},{red[1]:+d}] 半音      {clean}")
        clips.append((name, label, y))

    # RMS 对齐蒙太奇
    gap = np.zeros(int(0.6 * SR), np.float32)
    seq = []
    for _, _, y in clips:
        yy = y * (0.12 / (np.sqrt(np.mean(y**2)) + 1e-9))
        seq.append(np.clip(yy, -0.99, 0.99).astype(np.float32))
        seq.append(gap)
    sf.write(DEMO / "hero2_montage.wav", np.concatenate(seq).astype(np.float32), SR)
    print(
        f"\n  -> hero2_montage.wav  ({sum(len(c[2]) for c in clips) / SR + 0.6 * len(clips):.1f}s)"
    )
    print("     " + " → ".join(c[1] for c in clips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
