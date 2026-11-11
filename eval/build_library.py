"""Build a .tplib articulation library from a performance segment: sprint A1 (real guitar chain).

load performance segment -> FR-5 events -> FR-6 classification -> FR-7 library; the timbre snapshot is taken from the cleanest sustain event
(ideally it should come from an isolated note in tones/; this is the fallback when none was recorded).

Usage:
  python -m eval.build_library assets/recordings/performance/guitar_perf_01.m4a \
         --name guitar --out assets/libraries/guitar.tplib
  # if an isolated note is available for the timbre: --tone assets/recordings/tones/guitar_tone_mid.wav
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
    """Pick the cleanest sustain event (monophonic, long enough, attack not too hard) as the timbre snapshot source."""
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
    ap = argparse.ArgumentParser(description="Build a .tplib articulation library")
    ap.add_argument("perf", help="performance segment wav/m4a")
    ap.add_argument("--tone", default=None, help="isolated note (timbre snapshot source, optional)")
    ap.add_argument("--name", default="guitar")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    print(f"Loading performance segment: {args.perf}")
    y, sr = load(args.perf, sr=SR)
    print(f"  {len(y) / sr:.1f}s @ {sr}Hz")

    print("FR-5 event segmentation + FR-6 classification ...")
    events = analyze_events(y, sr=sr)
    classify_events(events)
    import collections

    dist = collections.Counter(e["label"] for e in events)
    print(f"  {len(events)} events  distribution: {dict(dist)}")

    # timbre snapshot
    if args.tone:
        print(f"Timbre snapshot from isolated note: {args.tone}")
        yt, _ = load(args.tone, sr=sr)
        snap = analyze_timbre(yt, sr=sr, name=args.name)
        snap_src = Path(args.tone).name
    else:
        clip, e = _best_sustain_clip(y, sr, events)
        if clip is not None:
            print(
                f"Timbre snapshot from the longest sustain event in the performance t={e['t_on']:.2f}-{e['t_off']:.2f}s "
                f"(fallback; recording an isolated note in tones/ is recommended)"
            )
            snap = analyze_timbre(clip, sr=sr, name=args.name)
            snap_src = f"perf_sustain@{e['t_on']:.2f}s"
        else:
            snap, snap_src = None, "none"

    print("FR-7 building .tplib ...")
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

    # round-trip check
    lib2 = load_tplib(p)
    ok = all(len(lib[k]) == len(lib2[k]) for k in ("pluck", "slide", "legato", "strum"))
    print(
        f"\nLibrary contents: pluck×{len(lib['pluck'])} slide×{len(lib['slide'])} "
        f"legato×{len(lib['legato'])} strum×{len(lib['strum'])} "
        f"sustain={'yes(' + snap_src + ')' if lib['sustain'] else 'none'}"
    )
    print(f".tplib -> {p}  round trip identical: {ok}")
    if lib["slide"]:
        iv = [round(s["interval_st"], 1) for s in lib["slide"][:6]]
        print(f"  slide interval examples (semitones): {iv}")
    if lib["strum"]:
        print(f"  strum examples: {len(lib['strum'])} groups, first group direction={lib['strum'][0]['direction']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
