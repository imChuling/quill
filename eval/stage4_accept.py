"""阶段4 验收 —— FR-1 完整 + FR-5 ｜ 出口：events.json 召回 ≥90%。

闭环自动验收：用阶段3 引擎渲染**已知**事件序列 -> FR-5 分析 -> 与已知真值算召回。
（真实素材的 50 事件召回需 `eval/label_tool.py` 手标后用同一 compute_recall 比对。）

产物：assets/demos/stage4_events.json（冻结的 events.json）+ stage4_recall.png

用法：
  python -m eval.stage4_accept                # 闭环（渲染已知曲 -> 分析 -> 召回）
  python -m eval.stage4_accept guitar.wav     # 真实素材：出 events.json + 启动手标工具
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.features import analyze_events, save_events  # noqa: E402
from eval.label_tool import compute_recall  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

SR = 44100
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "assets" / "demos"
SNAP_DIR = ROOT / "assets" / "snapshots"

# 已知测试曲：上行旋律（8 单音）+ C 大三和弦（测复音）
_MELODY = [
    (0.20, 60),
    (0.70, 62),
    (1.20, 64),
    (1.70, 65),
    (2.20, 67),
    (2.70, 69),
    (3.20, 71),
    (3.70, 72),
]
_CHORD = [(4.30, 48), (4.30, 52), (4.30, 55)]
_DUR = 5.2


def _render_known(snapshot="guitar"):
    p = SNAP_DIR / f"{snapshot}.npz"
    if p.exists():
        inst = build_instrument(load_snapshot(p), noise_level=0.0)  # 干净测信号
        src = f"快照 {p.name}（noise=0 干净测信号）"
    else:
        K = 40
        inst = build_instrument  # type: ignore
        from synth.instrument import Instrument

        inst = Instrument(
            1.0 / np.arange(1, K + 1),
            np.arange(1, K + 1, dtype=float),
            np.zeros(0),
            np.zeros(0),
            220.0,
            SR,
            noise_level=0.0,
        )
        src = "合成兜底乐器"
    eng = Engine(inst, eval_mode=True)
    evs, gt = [], []
    for t, pch in _MELODY:
        evs += [(t, "on", pch, 100, 0.0), (t + 0.4, "off", pch, 0, 0.0)]
        gt.append({"t_on": t, "pitch_midi": float(pch)})
    for t, pch in _CHORD:
        evs += [(t, "on", pch, 100, 0.0), (t + 0.6, "off", pch, 0, 0.0)]
        gt.append({"t_on": t, "pitch_midi": float(pch)})
    y = eng.render_offline(evs, _DUR)
    return y, gt, src


def _plot(y, gt, det, recall):
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(y)) / SR
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y, lw=0.3, color="gray", alpha=0.7)
    for g in gt:
        ax.axvline(g["t_on"], color="C2", lw=1.2, alpha=0.7)
    for d in det:
        ax.axvline(d["t_on"], color="C1", lw=0.8, ls="--", alpha=0.8)
    ax.plot([], [], color="C2", label="ground truth onset")
    ax.plot([], [], color="C1", ls="--", label="detected onset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("amp")
    ax.set_title(
        f"Stage 4 FR-5: event detection  recall={recall * 100:.0f}%  "
        f"(GT={len(gt)}, detected={len(det)})"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DEMO_DIR / "stage4_recall.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill 阶段4 验收（FR-5）")
    ap.add_argument("wav", nargs="?", help="真实素材 wav（出 events.json + 手标）")
    ap.add_argument("--snapshot", default="guitar")
    ap.add_argument("--label", action="store_true", help="真实素材跑完启动手标工具")
    args = ap.parse_args(argv)

    if args.wav:
        from analysis.io import load

        print(f"分析真实素材: {args.wav}")
        y, _ = load(args.wav, sr=SR)
        det, diag = analyze_events(y, return_diagnostics=True)
        out = save_events(DEMO_DIR / "stage4_events.json", det)
        print(
            f"  events.json -> {out}  ({len(det)} 事件, "
            f"{len(diag['ghosts'])} 幽灵已滤, {len(diag['disagreements'])} 待裁决)"
        )
        print("  真实素材召回需手标：")
        print(
            "    from eval.label_tool import LabelTool; "
            "LabelTool(y, 44100, det).run()  # 标 50 事件后与 compute_recall 比对"
        )
        if args.label:
            from eval.label_tool import LabelTool

            LabelTool(y, SR, det).run()
        return 0

    print("闭环验收：渲染已知曲 -> FR-5 分析 -> 召回")
    y, gt, src = _render_known(args.snapshot)
    print(f"  测信号来源: {src}")
    det, diag = analyze_events(y, return_diagnostics=True)
    r = compute_recall(det, gt, onset_tol_s=0.06, pitch_tol=1.0)
    save_events(DEMO_DIR / "stage4_events.json", det)
    _plot(y, gt, det, r["recall"])

    print(f"\n  真值事件        : {r['n_ref']}（{len(_MELODY)} 单音 + {len(_CHORD)} 复音和弦）")
    print(f"  检测事件        : {r['n_det']}")
    print(f"  幽灵八度已滤    : {len(diag['ghosts'])}")
    print(f"  双前端待裁决    : {len(diag['disagreements'])}")
    print(f"  召回 recall     : {r['recall'] * 100:.1f}%   (匹配 {r['matched']}/{r['n_ref']})")
    print(f"  精确 precision  : {r['precision'] * 100:.1f}%")
    if r["misses"]:
        print(
            f"  漏检            : {[(round(m['t_on'], 2), m['pitch_midi']) for m in r['misses']]}"
        )

    # 抽查一个事件的特征（Part B 契约）
    if det:
        e = det[len(det) // 2]
        print(
            f"\n  样例事件 #{e['id']}: pitch={e['pitch_midi']} "
            f"vel={e['velocity_est']} feats={e['features']}"
        )

    ok = r["recall"] >= 0.90
    print("\n---------------- 阶段4 验收汇总 ----------------")
    print(f"  [{'OK ' if ok else 'FAIL'}] events.json 召回 {r['recall'] * 100:.1f}% (判据 ≥90%)")
    print(f"  events.json -> {DEMO_DIR / 'stage4_events.json'}  | 图 -> stage4_recall.png")
    print("  注：闭环用合成已知曲验证 FR-5 管线；真实吉他 50 事件召回用 label_tool 手标对照")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
