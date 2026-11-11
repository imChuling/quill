"""从 IDMT-SMT-Bass 扒贝斯演奏法 -> bass.tplib + 指弹/slap/滑音 对比 demo。

IDMT-SMT-Bass 布局：PS/{FS,PK,MU,ST,SP}（拨弦风格，NO 表情）、ES/{NO,VIF,SLU,SLD,...}。
单音带已知技法（从路径），无需分类——直接按风格扒起音颗粒 + 从 SLU 扒滑音曲线 + 取
一个干净音建 sustain 快照。第二个从真演奏数据扒出演奏法的家族（取色笔·跨家族）。
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.harmonic import harmonics, track_f0
from analysis.io import load
from analysis.residual import attack_residual, extract_residual
from analysis.timbre import analyze_timbre
from analysis.tplib import save_tplib

SR = 44100
PLUCK_STYLES = {"FS": "finger", "PK": "pick", "MU": "mute", "ST": "slap_thumb", "SP": "slap_pluck"}


def _attack_grain(y, sr=SR):
    """从一个贝斯单音扒起音残差颗粒（谐波减除后的瞬态）。"""
    f0, conf, ft = track_f0(y, sr=sr)
    H, ht, info = harmonics(y, f0, sr=sr, f0_times=ft, n_harmonics=60)
    res = extract_residual(y, H, info["f0_per_frame"], sr=sr, n_fft=info["n_fft"], hop=512)
    return attack_residual(res, sr=sr)


def build(root, out="assets/libraries/bass.tplib", per_style=6):
    root = str(root)
    lib = {
        "pluck": [],
        "slide": [],
        "legato": [],
        "strum": [],
        "sustain": None,
        "meta": {"instrument_guess": "bass", "source": "IDMT-SMT-Bass", "format_version": 1},
    }
    # 1) 每拨弦风格的起音颗粒（slap-thumb/slap-pluck 是贝斯标志性技法）
    for code, name in PLUCK_STYLES.items():
        files = sorted(glob.glob(os.path.join(root, "PS", code, "*.wav")))[:per_style]
        grains = []
        for f in files:
            try:
                grains.append(_attack_grain(load(f, sr=SR)[0]))
            except Exception:
                continue
        if grains:
            g = np.mean(np.stack(grains), axis=0).astype(np.float32)  # 代表颗粒
            rms = float(np.sqrt(np.mean(g**2)))
            lib["pluck"].append({"rms_bucket": 1, "grain": g, "style": name})
            print(f"  pluck[{name:10s}] 颗粒 from {len(grains)} 音, rms={rms:.4f}")

    # 2) 滑音曲线 g(τ)∈[0,1]：从 SLU（滑上）扒 f0 轨迹归一
    slu = sorted(glob.glob(os.path.join(root, "ES", "SLU", "*.wav")))[:12]
    curves = []
    for f in slu:
        f0, conf, ft = track_f0(load(f, sr=SR)[0], sr=SR)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        if len(f0) < 12:
            continue
        rise = np.log2(f0 / f0[0])
        if rise[-1] <= 1e-3:
            continue
        g = rise / rise[-1]  # 归一到 [0,1]
        curves.append(np.interp(np.linspace(0, 1, 8), np.linspace(0, 1, len(g)), g))
    if curves:
        curve = np.clip(np.mean(np.stack(curves), axis=0), 0, 1).astype(np.float32)
        lib["slide"].append({"interval_st": 2.0, "dur_s": 0.25, "curve_ctrl": curve})
        print(f"  slide 曲线 from {len(curves)} 滑音: {np.round(curve, 2)}")

    # 3) sustain 快照：取一个干净 finger 音建 TimbreSnapshot
    fs = sorted(glob.glob(os.path.join(root, "PS", "FS", "*.wav")))
    if fs:
        snap = analyze_timbre(load(fs[len(fs) // 2], sr=SR)[0], name="bass_FS")
        lib["sustain"] = snap
        print(f"  sustain 快照 f0={snap['f0_ref']:.1f}Hz")

    lib["meta"]["stats"] = {k: len(lib[k]) for k in ("pluck", "slide", "legato", "strum")}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_tplib(out, lib)
    print(f"-> {out}  (pluck {len(lib['pluck'])} 风格, slide {len(lib['slide'])})")
    return lib


def demo(lib, out_dir="assets/demos"):
    """指弹 vs slap vs 滑音 对比 demo（同一贝斯线）。"""
    import soundfile as sf

    from runtime.engine import Engine
    from synth.instrument import build_instrument

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    snap = lib["sustain"]
    grains = {p["style"]: np.asarray(p["grain"]) for p in lib["pluck"]}

    def riff_engine(style_grain):
        inst = build_instrument(snap, pick=0.6)
        if style_grain is not None and style_grain.size:
            inst.attack_grain = np.ascontiguousarray(style_grain, dtype=np.float64)
        return Engine(inst, eval_mode=True)

    # 一段贝斯 riff（低音区根-五-八-滑）
    riff = [
        (0.1, "on", 40, 110, 0.0),
        (0.45, "off", 40, 0, 0.0),
        (0.5, "on", 40, 110, 0.0),
        (0.85, "off", 40, 0, 0.0),
        (0.9, "on", 47, 100, 0.0),
        (1.25, "off", 47, 0, 0.0),
        (1.3, "on", 52, 100, 0.0),
        (1.95, "slide", 55, 100, 0.0, 52),
        (2.4, "off", 55, 0, 0.0),
    ]
    for style in ("finger", "slap_thumb"):
        eng = riff_engine(grains.get(style))
        eng.set_gesture_templates(
            slide_curve=np.asarray(lib["slide"][0]["curve_ctrl"]) if lib["slide"] else None
        )
        y = np.asarray(eng.render_offline(riff, dur_s=2.8), np.float32)
        y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
        sf.write(os.path.join(out_dir, f"bass_{style}.wav"), y, SR)
        print(f"  贝斯 riff × {style} -> {out_dir}/bass_{style}.wav")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../../datasets/IDMT-SMT-BASS")
    ap.add_argument("--no-demo", action="store_true")
    args = ap.parse_args(argv)
    lib = build(args.root)
    if not args.no_demo:
        demo(lib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
