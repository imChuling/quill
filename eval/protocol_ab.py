"""Protocol A/B: old protocol (steady sustained note) vs new protocol (performance segment). **The same recording**, a fair comparison.

OLD: matches the product's current capture path: `analyze_timbre(window="auto")` automatically picks a stable segment and cuts a mid-section window
     (skipping the attack grain), giving the timbre of "one steady point".
NEW: `capture_from_performance`: takes a complete representative note including its attack (keeping transients and spectral flux),
     and extracts .tplib behaviour trajectories from the same recording.

Both arms are rendered with **the same phrase**, RMS-matched. NEW also gets a version "with behaviour trajectories" (using the gesture as well).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from analysis.capture_performance import SR, capture_from_performance, load_any  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402

PHRASE = [
    (55, 0.00, 0.42, "pluck"),
    (57, 0.45, 0.42, "legato"),
    (59, 0.90, 0.42, "legato"),
    (62, 1.35, 0.50, "pluck"),
    (67, 1.95, 1.60, "slide"),
    (62, 3.65, 1.10, "slide"),
]


def rms(y, r=0.08):
    v = np.sqrt((np.asarray(y, np.float64) ** 2).mean())
    return (np.asarray(y, np.float64) * (r / max(v, 1e-9))).astype(np.float32)


def play_phrase(snap, noise=0.02):
    inst = build_instrument(snap, noise_level=noise)
    eng = Engine(inst, eval_mode=False)
    eng.reset()
    evs, prev = [], -1
    for midi, t_on, dur, art in PHRASE:
        if art == "legato" and prev >= 0:
            evs.append((t_on, "legato", midi, 96, 0.04, prev))
        elif art == "slide" and prev >= 0:
            evs.append((t_on, "slide", midi, 105, 0.12, prev))
        else:
            evs.append((t_on, "on", midi, 100, 0.0, -1))
        evs.append((t_on + dur, "off", midi, 0, 0.0, -1))
        prev = midi
    total = PHRASE[-1][1] + PHRASE[-1][2] + 0.9
    return eng.render_offline(evs, total).astype(np.float64)


def flux_of(snap):
    H = np.asarray(snap["H_env"], np.float64)
    return float(np.mean(np.abs(np.diff(np.log(np.maximum(H, 1e-9)), axis=1))))


def attack_energy(snap):
    a = np.asarray(snap.get("attack_res", []), np.float64)
    return float(np.sqrt((a**2).mean())) if a.size else 0.0


def main(src=None, tag=None, quasi=False):
    import soundfile as sf

    src = Path(src or ROOT / "assets/recordings/performance/guitar_perf_01.m4a")
    tag = tag or src.stem
    out = ROOT / "assets/ear_tests/protocol_ab"
    out.mkdir(parents=True, exist_ok=True)
    y = load_any(src)
    print(f"material {src.name}  {len(y) / SR:.1f}s")

    # OLD: the product's current path (automatic stable window + mid-section cut)
    snap_old = analyze_timbre(
        y,
        sr=SR,
        name=f"{tag}_old",
        quasi_harmonic=quasi,
        denoise=True,
        window="auto",
        cross_check=False,
    )
    # NEW: performance capture (complete representative note with attack + behaviour trajectories)
    snap_new, behaviors, info = capture_from_performance(
        y, sr=SR, name=f"{tag}_new", quasi_harmonic=quasi
    )

    sf.write(str(out / f"{tag}_OLD_sustained.wav"), rms(play_phrase(snap_old)), SR)
    sf.write(str(out / f"{tag}_NEW_performance.wav"), rms(play_phrase(snap_new)), SR)

    res = {
        "source": str(src.name),
        "events": info["n_events"],
        "behaviors": info["n_behaviors"],
        "OLD": {
            "H_frames": int(np.asarray(snap_old["H_env"]).shape[1]),
            "spectral_flux": round(flux_of(snap_old), 4),
            "attack_rms": round(attack_energy(snap_old), 5),
            "f0_ref": round(float(snap_old["f0_ref"]), 1),
        },
        "NEW": {
            "H_frames": int(np.asarray(snap_new["H_env"]).shape[1]),
            "spectral_flux": round(flux_of(snap_new), 4),
            "attack_rms": round(attack_energy(snap_new), 5),
            "f0_ref": round(float(snap_new["f0_ref"]), 1),
        },
    }
    (ROOT / f"results/protocol_ab_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"\n{'':14s} {'H frames':>5s} {'spec flux':>8s} {'attack RMS':>9s} {'f0':>7s}")
    for k in ("OLD", "NEW"):
        r = res[k]
        print(
            f"{k + ' protocol':14s} {r['H_frames']:5d} {r['spectral_flux']:8.4f} "
            f"{r['attack_rms']:9.5f} {r['f0_ref']:7.1f}"
        )
    print(f"\n→ {out.relative_to(ROOT)}/{tag}_{{OLD_sustained,NEW_performance}}.wav")
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else None, a[1] if len(a) > 1 else None, quasi="--quasi" in a)
