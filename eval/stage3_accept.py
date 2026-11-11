"""阶段3 验收 —— FR-4 ｜ 出口：实测延迟数字 + 满复音无 xrun。

本机离线可验证（不需声卡）：
  ① 压力验收：16 复音满载 process_block 回调耗时 < 块预算 50%（NFR-2 最坏情况口径）
  ② 黄金文件回归：固定事件序列离线渲染 -> 与基线逐样本比对（eval_mode 可复现）
  ③ 渲染一段试听短曲（和弦+旋律）-> assets/demos/stage3_phrase.wav

需声卡才能跑（本脚本会尝试并容错）：
  ④ 延迟报告：sd.OutputStream 实测 latency；环回法测端到端延迟的操作步骤见下方打印

用法：
  python -m eval.stage3_accept                  # 用 assets/snapshots/guitar.npz（或合成兜底）
  python -m eval.stage3_accept voice            # 指定快照名
  python -m eval.stage3_accept --update-golden  # 重置黄金基线
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.engine import Engine, Event  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import Instrument, build_instrument  # noqa: E402

try:
    from quill_config import CFG
except Exception:  # pragma: no cover
    CFG = None

SR = 44100
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "assets" / "demos"
SNAP_DIR = ROOT / "assets" / "snapshots"
GOLDEN = ROOT / "eval" / "golden" / "stage3_golden.npy"


def _load_instrument(name):
    p = SNAP_DIR / f"{name}.npz"
    if p.exists():
        snap = load_snapshot(p)
        nl = (CFG or {}).get("runtime", {}).get("noise_level", 0.5)
        return build_instrument(snap, noise_level=nl), f"快照 {p.name}"
    # 合成兜底：60 谐波 1/k 滚降
    K = 60
    H = 1.0 / np.arange(1, K + 1)
    inst = Instrument(
        H,
        np.arange(1, K + 1, dtype=float),
        np.zeros(0),
        np.zeros(0),
        220.0,
        SR,
        name="synthetic",
        noise_level=0.3,
    )
    return inst, "合成兜底乐器(60谐波)"


# --------------------------------------------------------------------------- #
# ① 压力验收：16 复音满载回调耗时
# --------------------------------------------------------------------------- #
def stress_test(inst, n_blocks=600):
    eng = Engine(inst, cfg=CFG, eval_mode=True)
    bs = eng.blocksize
    budget_ms = bs / eng.sr * 1000.0

    # 起 16 个不同音 + bend，进入满载稳态
    eng.reset()
    for i in range(eng.max_voices):
        eng._apply_event(Event("on", 45 + i, 110, 0.0))
    out = np.zeros(bs, dtype=np.float32)

    # 预热若干块（JIT/缓存就绪）
    for _ in range(20):
        eng.process_block(out, bs, 0)

    # 演奏模式：测量时禁 GC（铁律 #1，复现实时回调条件——GC 是最大隐藏 xrun 源）
    import gc

    gc.collect()
    try:
        gc.freeze()
    except Exception:
        pass
    gc.disable()
    try:
        times = np.empty(n_blocks)
        for b in range(n_blocks):
            t0 = time.perf_counter()
            eng.process_block(out, bs, b * bs)
            times[b] = (time.perf_counter() - t0) * 1000.0
    finally:
        gc.enable()

    active = sum(1 for v in eng.pool if v.active)
    return {
        "budget_ms": budget_ms,
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "max": float(times.max()),
        "active_voices": active,
        "blocksize": bs,
        "err": eng.error_flag,
    }


# --------------------------------------------------------------------------- #
# ② 黄金文件回归 + ③ 试听短曲
# --------------------------------------------------------------------------- #
_PHRASE = [
    # 和弦（C major）三音齐发，再叠两段旋律
    (0.00, "on", 60, 100, 0.0),
    (0.00, "on", 64, 95, 0.0),
    (0.00, "on", 67, 95, 0.0),
    (0.80, "off", 60, 0, 0.0),
    (0.80, "off", 64, 0, 0.0),
    (0.80, "off", 67, 0, 0.0),
    (0.85, "on", 72, 110, 0.0),
    (1.05, "off", 72, 0, 0.0),
    (1.05, "on", 71, 100, 0.0),
    (1.25, "off", 71, 0, 0.0),
    (1.25, "on", 69, 90, 0.0),
    (1.55, "off", 69, 0, 0.0),
    (1.55, "on", 67, 105, 0.0),
    (1.7, "bend", -1, 0, 2.0),  # pitch bend +2 半音
    (2.2, "off", 67, 0, 0.0),
]
_PHRASE_DUR = 2.6


def golden_regression(inst, update=False):
    eng = Engine(inst, cfg=CFG, eval_mode=True)
    y = eng.render_offline(_PHRASE, _PHRASE_DUR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(
        DEMO_DIR / "stage3_phrase.wav",
        (y / (np.max(np.abs(y)) + 1e-9) * 0.97).astype(np.float32),
        SR,
    )

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    status = "n/a"
    if update or not GOLDEN.exists():
        np.save(GOLDEN, y)
        status = "基线已写入" if update else "首次创建基线"
    else:
        ref = np.load(GOLDEN)
        if ref.shape == y.shape and np.allclose(ref, y, atol=1e-6):
            status = "PASS（与基线一致）"
        else:
            md = (
                float(np.max(np.abs(ref[: len(y)] - y[: len(ref)])))
                if ref.shape == y.shape
                else float("nan")
            )
            status = f"DIFF（最大样本差 {md:.2e}）"
    return y, status, bool(eng.error_flag)


# --------------------------------------------------------------------------- #
# ④ 延迟报告（需声卡；容错）
# --------------------------------------------------------------------------- #
def latency_report(inst):
    budget_ms = inst and (Engine(inst, cfg=CFG).blocksize / SR * 1000.0)
    lines = [f"块预算延迟（buffer）: {budget_ms:.2f} ms（blocksize/sr）"]
    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="output")
        lines.append(f"默认输出设备: {dev['name']}")
        with sd.OutputStream(
            samplerate=SR, blocksize=int(budget_ms * SR / 1000), channels=1, dtype="float32"
        ) as s:
            lines.append(f"sd 报告 stream.latency: {s.latency * 1000:.2f} ms")
    except Exception as exc:
        lines.append(f"（无声卡/不可用，跳过实测 stream 延迟：{type(exc).__name__}）")
    lines.append("环回法端到端延迟（在有声卡的机器上手动跑）：")
    lines.append("  把输出接回输入(或同房放录)，引擎发一个脉冲音，同步录输出，")
    lines.append("  量 MIDI/触发到录得 onset 的采样差 -> 端到端延迟（进延迟分解表）。")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill 阶段3 验收（FR-4）")
    ap.add_argument(
        "snapshot", nargs="?", default="guitar", help="assets/snapshots/<name>.npz（默认 guitar）"
    )
    ap.add_argument("--update-golden", action="store_true")
    args = ap.parse_args(argv)

    inst, src = _load_instrument(args.snapshot)
    print(f"乐器来源: {src}  (K={inst.K} 谐波, sr={inst.sr})")

    print("\n[①] 16 复音满载压力测试 …")
    s = stress_test(inst)
    pct = s["p95"] / s["budget_ms"] * 100
    pass1 = s["p95"] < 0.5 * s["budget_ms"] and not s["err"]
    print(f"  活跃声部     : {s['active_voices']}/{16}")
    print(f"  块预算       : {s['budget_ms']:.2f} ms (blocksize={s['blocksize']})")
    print(
        f"  回调耗时     : mean {s['mean']:.3f} / median {s['median']:.3f} / "
        f"p95 {s['p95']:.3f} / max {s['max']:.3f} ms"
    )
    print(f"  p95 占预算   : {pct:.1f}%   判据<50%  -> {'PASS' if pass1 else 'FAIL'}")

    print("\n[②] 黄金文件回归 + [③] 试听短曲 …")
    y, gstatus, gerr = golden_regression(inst, update=args.update_golden)
    pass2 = ("PASS" in gstatus) or ("基线" in gstatus) or ("创建" in gstatus)
    print(
        f"  短曲 -> assets/demos/stage3_phrase.wav  ({len(y) / SR:.2f}s, "
        f"peak {np.max(np.abs(y)):.3f}, finite {bool(np.all(np.isfinite(y)))})"
    )
    print(f"  黄金回归     : {gstatus}")

    print("\n[④] 延迟报告 …")
    for ln in latency_report(inst):
        print("  " + ln)

    ok = pass1 and pass2 and not gerr
    print("\n---------------- 阶段3 验收汇总 ----------------")
    print(
        f"  [{'OK ' if pass1 else 'FAIL'}] 满复音无 xrun：16 声部 p95 回调 {pct:.1f}% 预算 (<50%)"
    )
    print(f"  [{'OK ' if pass2 else 'FAIL'}] 离线渲染确定性/黄金回归：{gstatus}")
    print("  延迟：缓冲预算 5.8ms；端到端 <30ms 需在有声卡机器用环回法实测填表")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
