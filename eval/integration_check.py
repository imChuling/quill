"""中途集成检查点（6.25）—— sustain-only 端到端全数据通路。

按素材采集协议：单音 -> 快照 -> 键盘实时演奏，验证 analysis -> assets -> runtime
整条通路在 6.25 就跑通（首次全链路集成不留到 6.29）。接口问题打印出来记入阶段6联调清单。

本机离线版（无声卡/麦克风也可跑）：合成稳态单音替代现场录音，其余通路完全一致：
  analyze_timbre -> save_snapshot(.npz) -> load_snapshot -> build_instrument -> Engine.render_offline

用法：
  python -m eval.integration_check                 # 合成单音跑通通路
  python -m eval.integration_check --record 4      # 现场录 4s 单音（需麦克风）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.io import normalize  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot, save_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

SR = 44100
SNAP_DIR = Path(__file__).resolve().parents[1] / "assets" / "snapshots"


def _synth_tone(f0=196.0, dur=2.0, sr=SR):
    """稳态谐波单音（替代现场录音；通路其余环节一致）。"""
    t = np.arange(int(sr * dur)) / sr
    y = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 25))
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (dur - t) / 0.1)
    return normalize((y * env).astype(np.float32))


def _dominant_hz(y, sr):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(S)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill 中途集成检查点（sustain-only）")
    ap.add_argument("--record", type=float, default=0.0, help="现场录音秒数")
    args = ap.parse_args(argv)

    issues = []  # 接口问题清单 -> 阶段6 联调

    # ① analysis：录音/合成 -> 快照
    if args.record > 0:
        from analysis.io import record

        print(f"现场录音 {args.record}s 单音 …")
        y = record(args.record, sr=SR)
    else:
        print("（离线）合成稳态单音替代现场录音 …")
        y = _synth_tone()
    snap = analyze_timbre(y, sr=SR, name="integration_check")
    print(
        f"  analyze_timbre -> f0_ref={snap['f0_ref']:.1f}Hz, "
        f"H{snap['H_env'].shape}, N{snap['N_env'].shape}"
    )

    # ② assets：快照序列化往返
    p = save_snapshot(SNAP_DIR / "integration_check.npz", snap)
    snap2 = load_snapshot(p)
    if not np.allclose(snap["H_env"], snap2["H_env"]):
        issues.append("snapshot H_env 序列化往返不一致")
    print(
        f"  save/load snapshot -> {p.name}  往返一致={np.allclose(snap['H_env'], snap2['H_env'])}"
    )

    # ③ runtime：快照 -> 乐器 -> 引擎实时演奏（离线渲染同一 process_block 路径）
    inst = build_instrument(snap2, noise_level=0.3)
    eng = Engine(inst, eval_mode=True)
    note = 64  # E4
    y_play = eng.render_offline([(0.1, "on", note, 100, 0.0), (0.9, "off", note, 0, 0.0)], 1.2)
    f0_target = 440.0 * 2 ** ((note - 69) / 12.0)
    got = _dominant_hz(y_play[int(0.2 * SR) : int(0.8 * SR)], SR)
    cents = 1200 * np.log2(got / f0_target)
    clean = bool(np.all(np.isfinite(y_play)) and np.max(np.abs(y_play)) <= 1.0)
    if abs(cents) > 30:
        issues.append(f"演奏音高偏差 {cents:.0f} cents（note {note} 期望 {f0_target:.1f}Hz）")
    if not clean:
        issues.append("引擎输出非有限/越界")
    if eng.error_flag:
        issues.append("引擎 error_flag 置位")
    print(
        f"  engine 演奏 note {note}: f0={got:.1f}Hz (期望 {f0_target:.1f}, {cents:+.0f}cents), "
        f"clean={clean}"
    )

    print("\n---------------- 集成检查汇总 ----------------")
    ok = not issues
    print(f"  analysis -> assets -> runtime 全通路：{'PASS' if ok else 'FAIL'}")
    if issues:
        print("  接口问题（记入阶段6联调清单）：")
        for s in issues:
            print(f"   - {s}")
    else:
        print("  无接口问题；sustain-only 端到端打通（首次全链路集成完成）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
