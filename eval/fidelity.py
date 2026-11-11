"""Cross-pitch timbre fidelity evaluation: FR-13 (hard metric for timbre fidelity).

Quantitative evidence for the core claim: **Quill's formant-preserving resynthesis is more faithful to the real instrument than naive sampler
transposition far from the reference pitch.** Method:

  Build a TimbreSnapshot from a real recording at pitch A -> resynthesise to pitch B
    Quill   : formant-preserving additive resynthesis (render_offline / synth.render)
    Sampler : librosa.effects.pitch_shift (phase-vocoder transposition, no formant correction) = typical sampler baseline
  Each is compared with the **real recording at pitch B** by spectral-envelope distance (L1 of the mean log magnitude spectrum over the steady segment, robust to timing).
  Quill distance < Sampler distance = formant preservation is more faithful (especially for formant instruments).

Input is a set of (pitch_midi, audio) multi-pitch samples of the same instrument; currently supports the user's voice at three pitches / IDMT guitar /
NSynth (once downloaded).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.io import load  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from synth.render import render  # noqa: E402

SR = 44100


# --------------------------------------------------------------------------- #
# Spectral-envelope distance (L1 of the mean log magnitude spectrum over the steady segment, robust to phase/timing)
# --------------------------------------------------------------------------- #
def _steady_logspec(y, sr=SR, n_fft=2048, fmax=8000.0):
    seg = y[len(y) // 5 : -len(y) // 5] if len(y) > 10 else y
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    S = []
    for i in range(0, len(seg) - n_fft, n_fft // 2):
        S.append(np.abs(np.fft.rfft(seg[i : i + n_fft] * np.hanning(n_fft))))
    mag = np.mean(S, axis=0) if S else np.abs(np.fft.rfft(seg[:n_fft] * np.hanning(n_fft)))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    mag = mag[f <= fmax]
    mag = mag / (mag.sum() + 1e-12)  # normalise (remove overall loudness, compare shape)
    return np.log(mag + 1e-6)


def spectral_distance(y1, y2, sr=SR):
    a, b = _steady_logspec(y1, sr), _steady_logspec(y2, sr)
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _sampler_pitch_shift(y, semitones, sr=SR):
    """Sampler baseline: phase-vocoder transposition (librosa), formants not preserved."""
    import librosa

    if abs(semitones) < 1e-3:
        return np.asarray(y, dtype=np.float32)
    return librosa.effects.pitch_shift(
        np.asarray(y, dtype=np.float32), sr=sr, n_steps=float(semitones)
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cross-pitch fidelity of one instrument
# --------------------------------------------------------------------------- #
def evaluate_instrument(samples, name="instrument", sr=SR, quasi=False, max_semis=None):
    """samples: [(pitch_midi, audio), ...] real multi-pitch recordings of the same instrument.

    The median pitch is the reference for the snapshot; **each sample's true f0 is measured with PESTO** (independent of annotated/estimated pitch);
    Quill renders to the target's true f0 and the sampler transposes by the true f0 ratio; both are compared with the target's real recording by spectral distance.
    max_semis limits the target transposition (a musically reasonable range, so unrealistic 3-octave extremes do not drag both down).
    """
    from analysis.harmonic import f0_scalar, track_f0

    enriched = []
    for pitch, audio in samples:
        y = np.asarray(audio, np.float32)
        f0v = f0_scalar(track_f0(y, sr=sr)[0])
        if np.isfinite(f0v) and f0v > 0:
            enriched.append((f0v, pitch, y))
    enriched.sort(key=lambda s: s[0])
    if len(enriched) < 2:
        return [], None
    ref_i = len(enriched) // 2
    f0_ref_real, ref_pitch, ref_audio = enriched[ref_i]
    snap = analyze_timbre(ref_audio, sr=sr, name=name, quasi_harmonic=quasi)
    f0_ref = snap["f0_ref"]

    rows = []
    for i, (f0_tgt_real, _pitch, real) in enumerate(enriched):
        if i == ref_i:
            continue
        semis = 12.0 * np.log2(f0_tgt_real / f0_ref_real)
        if max_semis is not None and abs(semis) > max_semis:
            continue
        f0_tgt = f0_ref * (f0_tgt_real / f0_ref_real)  # snapshot reference f0 × true ratio
        n = len(real)
        quill = render(snap, f0_tgt, length=n, mode="formant", add_attack=False)
        sampler = _sampler_pitch_shift(ref_audio, semis, sr)
        dq = spectral_distance(quill, real, sr)
        ds = spectral_distance(sampler, real, sr)
        rows.append(
            {"semis": int(round(semis)), "quill": dq, "sampler": ds, "quill_better": dq < ds}
        )
    return rows, ref_pitch


def _report(rows, name):
    if not rows:
        print(f"  [{name}] not enough samples (need ≥2 pitches)")
        return None
    q = np.mean([r["quill"] for r in rows])
    s = np.mean([r["sampler"] for r in rows])
    win = np.mean([r["quill_better"] for r in rows])
    print(f"\n=== {name} ({len(rows)} target pitches) ===")
    print(f"  {'shift (st)':>10} {'Quill dist':>10} {'sampler dist':>10}  winner")
    for r in sorted(rows, key=lambda x: x["semis"]):
        print(
            f"  {r['semis']:>+10d} {r['quill']:>10.4f} {r['sampler']:>10.4f}  "
            f"{'Quill' if r['quill_better'] else 'sampler'}"
        )
    print(
        f"  mean: Quill {q:.4f} vs sampler {s:.4f}  "
        f"-> Quill {'more faithful' if q < s else 'not better'} (lower is closer to the real recording); "
        f"Quill win rate {win * 100:.0f}%"
    )
    return {"name": name, "quill": q, "sampler": s, "win_rate": win}


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
def voice_samples():
    """User's voice at three pitches (strong formants -> Quill's advantage should be clear)."""
    base = Path(__file__).resolve().parents[1] / "assets" / "recordings" / "tones"
    out = []
    midi = {"low": 50, "mid": 60, "high": 69}  # estimated register (only as a reference order; true f0 comes from analyze)
    for reg in ("low", "mid", "high"):
        for ext in (".m4a", ".wav", ".mp3"):
            p = base / f"voice_tone_{reg}{ext}"
            if p.exists():
                y, _ = load(str(p), sr=SR)
                out.append((midi[reg], y))
                break
    return out


def idmt_samples(root, guitar_subdir=None, max_notes=12):
    """Multi-pitch single notes from the same IDMT guitar (weak formants -> Quill's advantage is modest)."""
    import glob
    import os

    from analysis.datasets import load_idmt_smt_guitar

    evs = load_idmt_smt_guitar(root)
    # pick single-note files from one subdir (files often contain several notes; take pluck/normal ones)
    from collections import defaultdict

    byf = defaultdict(list)
    for e in evs:
        byf[e["_source_file"]].append(e)
    out = []
    seen_pitch = set()
    for f, es in byf.items():
        if len(es) != 1:  # take single-note files only, clean
            continue
        e = es[0]
        if e["pitch_midi"] in seen_pitch:
            continue
        stem = os.path.splitext(f)[0]
        hits = glob.glob(os.path.join(root, "**", stem + ".wav"), recursive=True)
        if not hits:
            continue
        y, _ = load(hits[0], sr=SR)
        out.append((int(e["pitch_midi"]), y))
        seen_pitch.add(e["pitch_midi"])
        if len(out) >= max_notes:
            break
    return out


# --------------------------------------------------------------------------- #
# Multi-instrument breadth data sources (TinySOL real orchestral recordings / NSynth benchmark)
# --------------------------------------------------------------------------- #
_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _note_to_midi(s):
    """Note name -> MIDI: 'A4'->69, 'C#5', 'Bb3', supports #/♯ b/♭."""
    s = s.strip()
    name = s[0].upper()
    i, acc = 1, 0
    while i < len(s) and s[i] in "#b♯♭":
        acc += 1 if s[i] in "#♯" else -1
        i += 1
    return 12 * (int(s[i:]) + 1) + _NOTE[name] + acc


def _spread(samples, max_n):
    """Evenly pick max_n samples (including both ends) from samples sorted by pitch, to bound run time."""
    if len(samples) <= max_n:
        return samples
    idx = np.linspace(0, len(samples) - 1, max_n).round().astype(int)
    return [samples[i] for i in sorted(set(idx))]


def tinysol_groups(root, technique="ord", dynamics="mf", max_pitch=9, min_pitch=3):
    """TinySOL -> [(TinySOL/Family/Instrument, [(midi,audio),...]), ...].

    Real layout <Family>/<Instrument>/<technique>/<Instr-tech-Pitch-Dyn-...>.wav
    (e.g. Brass/Bass_Tuba/ordinario/BTb-ord-F#1-pp-N-N.wav). Family/Instrument are taken from
    the 4th/3rd path segments from the end (robust to nested root directories); the file name is parsed for tech/pitch/dyn, fixed to
    ordinario + a single dynamics level to keep the timbre consistent.
    """
    import glob
    import os
    from collections import defaultdict

    root = str(root)
    groups = defaultdict(list)
    for wav in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        comp = wav.split(os.sep)
        if len(comp) < 4:
            continue
        fam, instr = comp[-4], comp[-3]  # …/Family/Instrument/technique/file.wav
        base = os.path.basename(wav)[:-4]
        parts = base.split("-")
        if len(parts) < 4:
            continue
        tech, pitch, dyn = parts[1], parts[2], parts[3]
        if tech != technique or dyn != dynamics:
            continue
        try:
            midi = _note_to_midi(pitch)
        except Exception:
            continue
        groups[(fam, instr)].append((midi, wav))
    out = []
    for (fam, instr), items in sorted(groups.items()):
        items = sorted(set(items))
        if len(items) < min_pitch:
            continue
        items = _spread(items, max_pitch)
        loaded = [(m, load(p, sr=SR)[0]) for m, p in items]
        out.append((f"TinySOL/{fam}/{instr}", loaded))
    return out


def nsynth_groups(root, per_family=1, max_pitch=9, min_pitch=3, families=None, velocity=None):
    """NSynth -> [(NSynth/family, [(midi,audio),...]), ...].

    File name <family>_<source>_<instrID>-<pitch>-<velocity>.wav, pitch is MIDI;
    grouped by instrument (family_source_instrID), per_family instruments taken per family.
    """
    import glob
    import os
    from collections import defaultdict

    root = str(root)
    audio_dir = os.path.join(root, "audio")
    files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not files:
        files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    by_instr = defaultdict(list)  # instr_key -> [(midi, vel, path)]
    fam_of = {}
    for wav in files:
        base = os.path.basename(wav)[:-4]
        if "-" not in base:
            continue
        head, pitch, vel = base.rsplit("-", 2)
        fam = head.split("_")[0]
        if families and fam not in families:
            continue
        try:
            by_instr[head].append((int(pitch), int(vel), wav))
            fam_of[head] = fam
        except ValueError:
            continue
    # per family, pick the per_family instruments with the widest pitch span, velocity fixed
    fam_instrs = defaultdict(list)
    for head, items in by_instr.items():
        fam_instrs[fam_of[head]].append((head, items))
    out = []
    for fam, instrs in sorted(fam_instrs.items()):
        instrs.sort(key=lambda hi: -len({m for m, v, p in hi[1]}))
        for _head, items in instrs[:per_family]:
            vels = sorted({v for m, v, p in items})
            vel = velocity if velocity in vels else vels[len(vels) // 2]
            ps = sorted({(m, p) for m, v, p in items if v == vel})
            if len(ps) < min_pitch:
                continue
            ps = _spread(ps, max_pitch)
            loaded = [(m, load(p, sr=SR)[0]) for m, p in ps]
            out.append((f"NSynth/{fam}", loaded))
    return out


def _run_breadth(groups, quasi=False, max_semis=18.0):
    """Run a batch of (name, samples) groups and return per-group + per-family results aggregated by family prefix."""
    from collections import defaultdict

    fam_rows = defaultdict(list)
    per_group = []
    for name, samples in groups:
        rows, _ = evaluate_instrument(samples, name=name, quasi=quasi, max_semis=max_semis)
        if not rows:
            continue
        q = float(np.mean([r["quill"] for r in rows]))
        s = float(np.mean([r["sampler"] for r in rows]))
        win = float(np.mean([r["quill_better"] for r in rows]))
        per_group.append({"name": name, "quill": q, "sampler": s, "win_rate": win, "n": len(rows)})
        fam = "/".join(name.split("/")[:2])
        fam_rows[fam].extend(rows)
        print(
            f"  {name:32s} Quill {q:.3f} vs sampler {s:.3f}  "
            f"win rate {win * 100:3.0f}%  ({'✓' if q < s else '×'})"
        )
    fam_summary = []
    for fam, rows in sorted(fam_rows.items()):
        q = float(np.mean([r["quill"] for r in rows]))
        s = float(np.mean([r["sampler"] for r in rows]))
        win = float(np.mean([r["quill_better"] for r in rows]))
        fam_summary.append({"name": fam, "quill": q, "sampler": s, "win_rate": win, "n": len(rows)})
    return per_group, fam_summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-pitch timbre fidelity vs sampler baseline")
    ap.add_argument("--idmt", default=None, help="IDMT root directory (adds guitar)")
    ap.add_argument("--tinysol", default=None, help="TinySOL root directory (orchestral breadth)")
    ap.add_argument("--nsynth", default=None, help="NSynth root directory (benchmark breadth)")
    ap.add_argument("--per-family", type=int, default=1, help="instruments per NSynth family")
    args = ap.parse_args(argv)

    summary = []
    vs = voice_samples()
    if len(vs) >= 2:
        rows, _ = evaluate_instrument(vs, name="voice (voice, strong formants)")
        r = _report(rows, "voice (voice, strong formants)")
        if r:
            summary.append(r)
    else:
        print("(fewer than 2 voice samples, skipped; needs tones/voice_tone_low/mid/high)")

    if args.idmt:
        gs = idmt_samples(args.idmt)
        if len(gs) >= 2:
            rows, _ = evaluate_instrument(gs, name="guitar (guitar, weak formants)")
            r = _report(rows, "guitar (guitar, weak formants)")
            if r:
                summary.append(r)

    # ---- Multi-instrument breadth ----
    fam_all = []
    if args.tinysol:
        print("\n========== TinySOL orchestral breadth (real recordings, strong formants) ==========")
        groups = tinysol_groups(args.tinysol)
        print(f"  ({len(groups)} instruments)")
        _, fam = _run_breadth(groups)
        fam_all += fam
    if args.nsynth:
        print("\n========== NSynth benchmark breadth ==========")
        groups = nsynth_groups(args.nsynth, per_family=args.per_family)
        print(f"  ({len(groups)} instruments)")
        _, fam = _run_breadth(groups)
        fam_all += fam

    if summary:
        print("\n---------------- Cross-pitch fidelity summary (user material) ----------------")
        for r in summary:
            print(
                f"  {r['name']:24s} Quill {r['quill']:.4f} vs sampler {r['sampler']:.4f}  "
                f"win rate {r['win_rate'] * 100:.0f}%  ({'✓ Quill more faithful' if r['quill'] < r['sampler'] else '×'})"
            )
        print("  Criterion: lower spectral distance is closer to the real recording; Quill's advantage should be larger on formant instruments.")

    if fam_all:
        print("\n---------------- Multi-instrument breadth summary (by family) ----------------")
        n_win = sum(f["quill"] < f["sampler"] for f in fam_all)
        for f in fam_all:
            print(
                f"  {f['name']:22s} Quill {f['quill']:.3f} vs sampler {f['sampler']:.3f}  "
                f"win rate {f['win_rate'] * 100:3.0f}%  n={f['n']:<3d} "
                f"({'✓ Quill more faithful' if f['quill'] < f['sampler'] else '×'})"
            )
        gq = np.mean([f["quill"] for f in fam_all])
        gs = np.mean([f["sampler"] for f in fam_all])
        print(
            f"  ── overall Quill more faithful in {n_win}/{len(fam_all)} families; "
            f"macro average Quill {gq:.3f} vs sampler {gs:.3f} ──"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
