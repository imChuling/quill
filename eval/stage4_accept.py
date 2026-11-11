"""Stage 4 acceptance: FR-1 complete + FR-5 | exit: events.json recall ≥90%.

Closed-loop automatic acceptance: render a **known** event sequence with the stage 3 engine -> FR-5 analysis -> recall against the known ground truth.
(Recall on 50 events of real material needs hand labelling with `eval/label_tool.py`, then comparison with the same compute_recall.)

Outputs: assets/demos/stage4_events.json (frozen events.json) + stage4_recall.png

Usage:
  python -m eval.stage4_accept                # closed loop (render known piece -> analyse -> recall)
  python -m eval.stage4_accept guitar.wav     # real material: write events.json + launch the hand-labelling tool
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

# known test piece: ascending melody (8 single notes) + C major triad (tests polyphony)
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
        inst = build_instrument(load_snapshot(p), noise_level=0.0)  # clean test signal
        src = f"snapshot {p.name} (noise=0 clean test signal)"
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
        src = "synthetic fallback instrument"
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
    ap = argparse.ArgumentParser(description="Quill stage 4 acceptance (FR-5)")
    ap.add_argument("wav", nargs="?", help="real material wav (writes events.json + hand labelling)")
    ap.add_argument("--snapshot", default="guitar")
    ap.add_argument("--label", action="store_true", help="launch the hand-labelling tool after processing real material")
    args = ap.parse_args(argv)

    if args.wav:
        from analysis.io import load

        print(f"Analysing real material: {args.wav}")
        y, _ = load(args.wav, sr=SR)
        det, diag = analyze_events(y, return_diagnostics=True)
        out = save_events(DEMO_DIR / "stage4_events.json", det)
        print(
            f"  events.json -> {out}  ({len(det)} events, "
            f"{len(diag['ghosts'])} ghosts filtered, {len(diag['disagreements'])} awaiting review)"
        )
        print("  Recall on real material needs hand labelling:")
        print(
            "    from eval.label_tool import LabelTool; "
            "LabelTool(y, 44100, det).run()  # label 50 events, then compare with compute_recall"
        )
        if args.label:
            from eval.label_tool import LabelTool

            LabelTool(y, SR, det).run()
        return 0

    print("Closed-loop acceptance: render known piece -> FR-5 analysis -> recall")
    y, gt, src = _render_known(args.snapshot)
    print(f"  test signal source: {src}")
    det, diag = analyze_events(y, return_diagnostics=True)
    r = compute_recall(det, gt, onset_tol_s=0.06, pitch_tol=1.0)
    save_events(DEMO_DIR / "stage4_events.json", det)
    _plot(y, gt, det, r["recall"])

    print(f"\n  ground-truth events        : {r['n_ref']} ({len(_MELODY)} single notes + {len(_CHORD)} chord notes)")
    print(f"  detected events        : {r['n_det']}")
    print(f"  ghost octaves filtered    : {len(diag['ghosts'])}")
    print(f"  dual front-end awaiting review    : {len(diag['disagreements'])}")
    print(f"  recall     : {r['recall'] * 100:.1f}%   (matched {r['matched']}/{r['n_ref']})")
    print(f"  precision  : {r['precision'] * 100:.1f}%")
    if r["misses"]:
        print(
            f"  misses            : {[(round(m['t_on'], 2), m['pitch_midi']) for m in r['misses']]}"
        )

    # spot-check the features of one event (Part B contract)
    if det:
        e = det[len(det) // 2]
        print(
            f"\n  sample event #{e['id']}: pitch={e['pitch_midi']} "
            f"vel={e['velocity_est']} feats={e['features']}"
        )

    ok = r["recall"] >= 0.90
    print("\n---------------- Stage 4 acceptance summary ----------------")
    print(f"  [{'OK ' if ok else 'FAIL'}] events.json recall {r['recall'] * 100:.1f}% (criterion ≥90%)")
    print(f"  events.json -> {DEMO_DIR / 'stage4_events.json'}  | plot -> stage4_recall.png")
    print("  Note: the closed loop validates the FR-5 pipeline with a synthetic known piece; recall on 50 real guitar events uses label_tool hand labels for comparison")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
