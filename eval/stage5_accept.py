"""Stage 5 acceptance: FR-6 basics + FR-7 | exit: lossless .tplib round trip + visual check ≥80%.

Builds a set of synthetic events with **known articulations** (pluck/sustain/slide/legato/strum, with vibrato modifiers),
runs the rule priority tree classification -> accuracy against ground truth (a reproducible substitute for the "visual check of self-recorded material ≥80%" criterion);
a random forest fallback serves as the ablation; then builds a .tplib from the classification and verifies a lossless read/write round trip.

(The formal IDMT macro F1 dual criteria are in stage 8; this stage validates the rule-tree pipeline + library round trip.)

Usage: python -m eval.stage5_accept
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.articulation import classify_events, forest_predict, train_forest  # noqa: E402
from analysis.features import (  # noqa: E402
    f0_slope_cents_s,
    f0_track_for,
    hnr_db,
    log_attack_time,
    polyphony_at,
    spectral_centroid_stats,
)
from analysis.harmonic import track_f0  # noqa: E402
from analysis.io import normalize  # noqa: E402
from analysis.timbre import analyze_timbre, detect_vibrato  # noqa: E402
from analysis.tplib import build_library, load_tplib, save_tplib  # noqa: E402

SR = 44100
DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demos"


# --------------------------------------------------------------------------- #
# Synthetic articulation generators (known ground truth)
# --------------------------------------------------------------------------- #
def gen_pluck(f0, sr=SR, dur=0.5):
    """Fast attack + high brightness + pluck noise -> small log_attack, high centroid."""
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t)
    for k in range(1, 60):
        fk = k * f0
        if fk >= sr / 2:
            break
        y += (1.0 / k**0.55) * np.exp(-t * (6 + 0.3 * k)) * np.sin(2 * np.pi * fk * t)
    rng = np.random.default_rng(int(f0))
    y += 0.4 * rng.standard_normal(len(t)) * np.exp(-t / 0.004)
    return normalize(y.astype(np.float32))


def gen_sustain(f0, sr=SR, dur=1.0, vib=False):
    """Slow attack + low brightness (few harmonics) -> not pluck; vib=True adds 5.5Hz vibrato."""
    t = np.arange(int(sr * dur)) / sr
    # fmod must be an array: with a scalar 0.0 when vib=False, cumsum collapses to length 1 (the carrier is lost)
    fmod = np.ones_like(t)
    if vib:
        fmod = fmod + 0.017 * np.sin(2 * np.pi * 5.5 * t)  # ~30 cents
    ph = np.cumsum(2 * np.pi * f0 * fmod / sr)
    y = sum((1.0 / k) * np.sin(k * ph) for k in range(1, 14))
    env = np.minimum(1.0, t / 0.04) * np.minimum(1.0, (dur - t) / 0.08)
    return normalize((y * env).astype(np.float32))


def gen_slide(f0a, semis, sr=SR, dur=0.45):
    """Continuous slide of semis semitones -> large, monotonic net f0 slope."""
    t = np.arange(int(sr * dur)) / sr
    f0t = f0a * 2 ** (semis * (t / dur) / 12.0)
    ph = np.cumsum(2 * np.pi * f0t / sr)
    y = sum((1.0 / k) * np.sin(k * ph) for k in range(1, 14))
    env = np.minimum(1.0, t / 0.02) * np.minimum(1.0, (dur - t) / 0.05)
    return normalize((y * env).astype(np.float32))


def gen_legato_target(f0, sr=SR, dur=0.6):
    """legato target note: soft attack (no transient), steady, low brightness."""
    return gen_sustain(f0, sr, dur, vib=False)


def _dominant_pitch(clip, sr):
    """FFT main peak -> midi (fallback when PESTO confidence is low on purely synthetic quiet signals)."""
    seg = clip[len(clip) // 8 : -len(clip) // 8] if len(clip) > 16 else clip
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    band = f > 40
    if not band.any():
        return None
    hz = float(f[band][np.argmax(S[band])])
    return 69 + 12 * np.log2(hz / 440.0) if hz > 0 else None


def _mk_event(eid, clip, sr, t_on):
    """Compute features from a clip and assemble the event (with f0_track/vibrato, time anchored at t_on)."""
    f0, conf, ft = track_f0(clip, sr=sr)
    dur = len(clip) / sr
    track_rel = f0_track_for(f0, ft, 0, dur)
    track = [[round(t_on + p[0], 4), p[1]] for p in track_rel]
    cmean, cslope = spectral_centroid_stats(clip, sr)
    f0_med = float(np.median([p[1] for p in track_rel])) if track_rel else float("nan")
    pitch = 69 + 12 * np.log2(f0_med / 440.0) if f0_med > 0 else _dominant_pitch(clip, sr)
    vib = (
        detect_vibrato(np.array([p[1] for p in track_rel]), np.array([p[0] for p in track_rel]))
        if len(track_rel) >= 16
        else None
    )
    return {
        "id": eid,
        "t_on": round(t_on, 4),
        "t_off": round(t_on + dur, 4),
        "pitch_midi": round(float(pitch), 3) if pitch else None,
        "velocity_est": float(np.clip(np.sqrt(np.mean(clip.astype(np.float64) ** 2)) * 2, 0, 1)),
        "f0_track": track,
        "features": {
            "log_attack": log_attack_time(clip, sr),
            "centroid_mean": cmean,
            "centroid_slope": cslope,
            "f0_slope_cents_s": f0_slope_cents_s(track_rel),
            "polyphony": 1,
            "hnr": hnr_db(clip, sr, f0_med),
        },
        "label": "",
        "vibrato": vib,
    }


def build_labeled_set(sr=SR):
    """Build an event set with known ground truth + a concatenated timeline (for taking grains/curves for the .tplib)."""
    specs = []  # (clip, true_label, t_on): t_on is a placeholder, laid out later
    # single-event classes
    for f0 in (147, 165, 196, 220, 247, 294):
        specs.append((gen_pluck(f0), "pluck"))
    for f0 in (165, 196, 247):
        specs.append((gen_sustain(f0), "sustain"))
    for f0 in (175, 220, 262):  # vibrato -> still sustain
        specs.append((gen_sustain(f0, vib=True), "sustain"))
    for f0, s in ((165, 4), (196, 3), (220, 5), (147, 4), (262, 3), (175, 4)):
        specs.append((gen_slide(f0, s), "slide"))

    events, true, timeline = [], [], []
    cur_t, eid = 0.0, 0
    cur_samp = 0

    def _place(clip, label):
        nonlocal cur_t, eid, cur_samp
        ev = _mk_event(eid, clip, sr, cur_t)
        ev_true = label
        events.append(ev)
        true.append(ev_true)
        timeline.append((cur_samp, clip))
        eid += 1
        cur_t += len(clip) / sr + 0.15
        cur_samp = int(cur_t * sr)

    for clip, lab in specs:
        _place(clip, lab)

    # legato pair: previous note (sustain) + an immediately following legato target (small interval, soft attack, 10ms gap)
    for f0a, f0b in ((196, 220), (220, 247), (165, 175)):
        a = gen_sustain(f0a, dur=0.5)
        _place(a, "sustain")
        cur_t -= 0.15 - 0.01  # shrink the gap to ~10ms so the legato joins
        cur_samp = int(cur_t * sr)
        b = gen_legato_target(f0b, dur=0.5)
        _place(b, "legato")

    # strum group: 3 notes staggered by 25ms (onset≥3 in the same window)
    for base in (130, 165):
        chord = [
            gen_pluck(base, dur=0.6),
            gen_pluck(base * 2 ** (4 / 12), dur=0.6),
            gen_pluck(base * 2 ** (7 / 12), dur=0.6),
        ]
        for off, c in zip((0.0, 0.025, 0.05), chord):
            ev = _mk_event(eid, c, sr, cur_t + off)
            events.append(ev)
            true.append("strum")
            timeline.append((int((cur_t + off) * sr), c))
            eid += 1
        cur_t += 0.6 + 0.15
        cur_samp = int(cur_t * sr)

    # recompute polyphony (based on the final time layout)
    for ev in events:
        ev["features"]["polyphony"] = polyphony_at(ev, events)

    # concatenate timeline audio (for taking grains/curves when building the library)
    total = max(s + len(c) for s, c in timeline) + sr
    y = np.zeros(total, dtype=np.float32)
    for s, c in timeline:
        y[s : s + len(c)] += c
    return events, true, normalize(y), sr


# --------------------------------------------------------------------------- #
def _confusion(true, pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


def main():
    print("Building the known-articulation event set ...")
    events, true, y, sr = build_labeled_set()
    labels = ["pluck", "slide", "legato", "strum", "sustain"]

    # rule tree
    classify_events([dict(e) for e in events])  # warm (does not modify the original)
    classified = classify_events(events)
    pred = [e["label"] for e in classified]
    acc = float(np.mean([t == p for t, p in zip(true, pred)]))
    M = _confusion(true, pred, labels)

    # vibrato detection (sustain subcase)
    vib_true = sum(1 for e, t in zip(events, true) if t == "sustain" and e.get("vibrato"))
    print(f"\n  total events: {len(events)}  rule tree accuracy: {acc * 100:.1f}%  (criterion ≥80%)")
    print("  confusion matrix (rows=truth, cols=predicted):  " + " ".join(f"{l[:4]:>5}" for l in labels))
    for i, l in enumerate(labels):
        print(f"    {l:8s} " + " ".join(f"{M[i, j]:5d}" for j in range(len(labels))))
    print(f"  vibrato detected (sustain with vibrato field): {vib_true} cases")

    # random forest ablation (train/test split)
    from sklearn.model_selection import train_test_split

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            events, true, test_size=0.35, random_state=0, stratify=true
        )
        clf = train_forest(Xtr, ytr)
        fpred = forest_predict(clf, Xte)
        facc = float(np.mean([a == b for a, b in zip(yte, fpred)]))
        print(
            f"\n  forest ablation: test accuracy {facc * 100:.1f}% "
            f"(train {len(Xtr)} / test {len(Xte)}): rules vs forest comparison"
        )
    except Exception as exc:
        facc = float("nan")
        print(f"  forest ablation skipped: {exc}")

    # FR-7: build .tplib + lossless round trip
    print("\n  Building the .tplib articulation library ...")
    sustain_clip = gen_sustain(196, dur=1.5)
    snap = analyze_timbre(sustain_clip, sr=sr, name="stage5_guitar")
    lib = build_library(
        classified, y, sr, sustain_snapshot=snap, instrument_guess="guitar", source="stage5_synth"
    )
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    p = save_tplib(DEMO_DIR / "stage5.tplib", lib)
    lib2 = load_tplib(p)
    lossless = _tplib_equal(lib, lib2)
    print(
        f"    Library contents: pluck×{len(lib['pluck'])} slide×{len(lib['slide'])} "
        f"legato×{len(lib['legato'])} strum×{len(lib['strum'])} "
        f"sustain={'yes' if lib['sustain'] else 'none'}"
    )
    print(f"    .tplib -> {p.name}  lossless round trip: {'PASS' if lossless else 'FAIL'}")

    ok = acc >= 0.80 and lossless
    print("\n---------------- Stage 5 acceptance summary ----------------")
    print(f"  [{'OK ' if acc >= 0.8 else 'FAIL'}] rule tree classification accuracy {acc * 100:.1f}% (≥80%)")
    print(f"  [{'OK ' if lossless else 'FAIL'}] lossless .tplib round trip")
    print("  Note: the self-recorded/IDMT macro F1 dual criteria (rules≥0.7 / MERT≥0.8) are formally evaluated in stage 8")
    return 0 if ok else 1


def _tplib_equal(a, b):
    """Compare two libraries array by array/scalar by scalar (lossless round-trip check)."""
    try:
        for cat in ("pluck", "slide", "legato", "strum"):
            if len(a[cat]) != len(b[cat]):
                return False
            for x, y_ in zip(a[cat], b[cat]):
                for k, v in x.items():
                    w = y_[k]
                    if isinstance(v, np.ndarray):
                        if not np.array_equal(v.astype(np.float32), np.asarray(w, np.float32)):
                            return False
                    elif isinstance(v, float):
                        if abs(v - float(w)) > 1e-6:
                            return False
                    elif v != w:
                        return False
        if (a["sustain"] is None) != (b["sustain"] is None):
            return False
        return not (
            a["sustain"] is not None
            and not np.array_equal(
                a["sustain"]["H_env"].astype(np.float32), b["sustain"]["H_env"].astype(np.float32)
            )
        )
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
