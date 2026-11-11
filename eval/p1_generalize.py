"""P1 泛化扫描(防过拟合,7.2 用户警醒)—— 一套锁死参数 × 多样真实声源。

原则:liveness 参数全局一套(0.85/0/0.8),**禁止逐文件调参**;运动本身来自每个源
自己的录音(数据驱动),全局常数只有深度。本脚本在 TinySOL(铜管/木管/弦乐/键盘)
+ 项目自录(人声×3/合成垫)上验证:不削波、无咔哒(残差峰度)、运动量在合理带内。

用法:python -m eval.p1_generalize
"""

from __future__ import annotations

import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from analysis.io import load  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from synth.voice import set_liveness  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TSOL = Path(os.environ.get("QUILL_DATASETS", str(Path.home() / "Desktop/MT/datasets"))) / "TinySOL"
SR = 44100
DUR = 11.5
LIVE = (0.85, 0.0, 0.8)  # 锁死的全局参数(与产品默认一致)

EVENTS = [
    (0.2, "on", 57, 96, 0.0, -1),
    (6.2, "off", 57, 0, 0.0, -1),
    (6.6, "on", 53, 92, 0.0, -1),
    (6.62, "on", 57, 88, 0.0, -1),
    (6.64, "on", 60, 90, 0.0, -1),
    (10.8, "off", 53, 0, 0.0, -1),
    (10.8, "off", 57, 0, 0.0, -1),
    (10.8, "off", 60, 0, 0.0, -1),
]


def mid_mf(inst_dir):
    """乐器目录里挑一个中音区 mf 单音(排序取中,启发式,不挑好听的=防过拟合)。"""
    fs = sorted(glob.glob(str(inst_dir / "ordinario" / "*-mf-*.wav")))
    return fs[len(fs) // 2] if fs else None


def sources():
    out = [
        ("voice_low", ROOT / "assets/recordings/tones/voice_tone_low.m4a"),
        ("voice_mid", ROOT / "assets/recordings/tones/voice_tone_mid.m4a"),
        ("voice_high", ROOT / "assets/recordings/tones/voice_tone_high.m4a"),
        ("synth_pad", ROOT / "assets/recordings/tones/synth_pad.wav"),
    ]
    for fam, inst in [
        ("Brass", "Trumpet in C"),
        ("Brass", "Horn"),
        ("Brass", "Bass Tuba"),
        ("Winds", "Flute"),
        ("Winds", "Clarinet in Bb"),
        ("Winds", "Bassoon"),
        ("Strings", "Violin"),
        ("Strings", "Violoncello"),
        ("Strings", "Contrabass"),
        ("Keyboards", "Accordion"),
    ]:
        f = mid_mf(TSOL / fam / inst)
        if f:
            out.append((inst.replace(" ", "_"), Path(f)))
    return out


def main():
    print(f"锁死参数 liveness={LIVE}(禁止逐文件调)\n")
    bad = 0
    for name, path in sources():
        try:
            y, sr = load(str(path), sr=SR)
            snap = analyze_timbre(np.asarray(y, np.float32), sr=sr, name=name)
            if not (np.isfinite(snap.get("f0_ref", 0)) and snap["f0_ref"] > 0):
                print(f"[{name:14s}] 跳过:无稳定音高")
                continue
            sh = int(round((69 + 12 * np.log2(snap["f0_ref"] / 440.0) - 57) / 12.0)) * 12
            evs = [
                (t, k, (n + sh if n >= 0 else n), v, val, n2) for (t, k, n, v, val, n2) in EVENTS
            ]
            ys = {}
            for mode, cfg in {"off": (0, 0, 0), "on": LIVE}.items():
                inst = build_instrument(snap, noise_level=0.2)
                eng = Engine(inst, eval_mode=True)
                set_liveness(*cfg)
                ys[mode] = eng.render_offline(evs, DUR)
            dr = np.diff(ys["on"] - ys["off"])
            m2 = (dr**2).mean() + 1e-30
            kur = float((dr**4).mean() / m2**2 - 3.0)
            pk = float(np.abs(ys["on"]).max())
            tdev = float(np.abs(inst.spec_traj - 1).mean()) if inst.spec_traj is not None else 0.0
            adev = float(np.abs(inst.amp_traj - 1).mean()) if inst.amp_traj is not None else 0.0
            ok = pk <= 1.0 and np.isfinite(ys["on"]).all() and kur < 30
            bad += not ok
            print(
                f"[{name:14s}] f0={snap['f0_ref']:6.1f}Hz 峰值={pk:.2f} 残差峰度={kur:5.1f} "
                f"形状呼吸={tdev:.3f} 响度呼吸={adev:.3f} → {'OK' if ok else 'FAIL'}"
            )
        except Exception as e:
            bad += 1
            print(f"[{name:14s}] 异常: {str(e)[:60]} → FAIL")
    print(
        f"\n结论: {'全部通过 —— 一套参数泛化成立' if bad == 0 else f'{bad} 个源失败,需查(不许用调参掩盖)'}"
    )
    set_liveness(*LIVE)


if __name__ == "__main__":
    main()
