"""从演奏段构建 .tplib 演奏法库 —— 冲刺 A①（真吉他链路）。

load 演奏段 -> FR-5 事件 -> FR-6 分类 -> FR-7 库；音色快照从最干净的 sustain 事件
抽取（理想应来自 tones/ 的孤立单音，未录时用此兜底）。

用法：
  python -m eval.build_library assets/recordings/performance/guitar_perf_01.m4a \
         --name guitar --out assets/libraries/guitar.tplib
  # 若已有孤立单音做音色：--tone assets/recordings/tones/guitar_tone_mid.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.articulation import classify_events  # noqa: E402
from analysis.features import analyze_events  # noqa: E402
from analysis.io import load  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from analysis.tplib import build_library, load_tplib, save_tplib  # noqa: E402

SR = 44100
LIB_DIR = Path(__file__).resolve().parents[1] / "assets" / "libraries"


def _best_sustain_clip(y, sr, events, min_dur=0.35):
    """挑最干净的 sustain 事件（单复音、时长够、起音不太硬）做音色快照来源。"""
    cands = [
        e
        for e in events
        if e["label"] == "sustain"
        and (e["t_off"] - e["t_on"]) >= min_dur
        and e["features"].get("polyphony", 9) <= 1
    ]
    if not cands:
        cands = [e for e in events if (e["t_off"] - e["t_on"]) >= min_dur]
    if not cands:
        return None, None
    e = max(cands, key=lambda x: x["t_off"] - x["t_on"])
    a, b = int(e["t_on"] * sr), int(e["t_off"] * sr)
    return y[a:b], e


def main(argv=None):
    ap = argparse.ArgumentParser(description="构建 .tplib 演奏法库")
    ap.add_argument("perf", help="演奏段 wav/m4a")
    ap.add_argument("--tone", default=None, help="孤立单音（音色快照来源，可选）")
    ap.add_argument("--name", default="guitar")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    print(f"载入演奏段: {args.perf}")
    y, sr = load(args.perf, sr=SR)
    print(f"  {len(y) / sr:.1f}s @ {sr}Hz")

    print("FR-5 事件切分 + FR-6 分类 …")
    events = analyze_events(y, sr=sr)
    classify_events(events)
    import collections

    dist = collections.Counter(e["label"] for e in events)
    print(f"  {len(events)} 事件  分布: {dict(dist)}")

    # 音色快照
    if args.tone:
        print(f"音色快照来自孤立单音: {args.tone}")
        yt, _ = load(args.tone, sr=sr)
        snap = analyze_timbre(yt, sr=sr, name=args.name)
        snap_src = Path(args.tone).name
    else:
        clip, e = _best_sustain_clip(y, sr, events)
        if clip is not None:
            print(
                f"音色快照来自演奏段最长 sustain 事件 t={e['t_on']:.2f}-{e['t_off']:.2f}s "
                f"（兜底；建议补录 tones/ 孤立单音）"
            )
            snap = analyze_timbre(clip, sr=sr, name=args.name)
            snap_src = f"perf_sustain@{e['t_on']:.2f}s"
        else:
            snap, snap_src = None, "无"

    print("FR-7 构建 .tplib …")
    lib = build_library(
        events,
        y,
        sr,
        sustain_snapshot=snap,
        instrument_guess=args.name,
        source=Path(args.perf).name,
    )
    out = Path(args.out) if args.out else LIB_DIR / f"{args.name}.tplib"
    p = save_tplib(out, lib)

    # 往返校验
    lib2 = load_tplib(p)
    ok = all(len(lib[k]) == len(lib2[k]) for k in ("pluck", "slide", "legato", "strum"))
    print(
        f"\n库内容: pluck×{len(lib['pluck'])} slide×{len(lib['slide'])} "
        f"legato×{len(lib['legato'])} strum×{len(lib['strum'])} "
        f"sustain={'有(' + snap_src + ')' if lib['sustain'] else '无'}"
    )
    print(f".tplib -> {p}  往返一致: {ok}")
    if lib["slide"]:
        iv = [round(s["interval_st"], 1) for s in lib["slide"][:6]]
        print(f"  slide 音程样例(半音): {iv}")
    if lib["strum"]:
        print(f"  strum 样例: {len(lib['strum'])} 组, 首组方向={lib['strum'][0]['direction']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
