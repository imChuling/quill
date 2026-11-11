"""Fréchet Audio Distance (FAD) evaluation: the standard distribution-level metric for timbre fidelity.

E2 v2 design (2026-11):
  3 arms (reference / Quill / sampler) × 2 instruments (glass / guitar)
  + split-half floor + bootstrap CI

  reference  = segments cut from real recordings (≥1 s)
  Quill      = style library → control_from_behavior → DDSP render
  sampler    = single-source real note → per-note pitch-shift + gating

  VGGish is the primary metric; MFCC uses frame-level statistics. Both instruments are in-sample.

Usage:
  conda run -n mt python eval/fad.py --e2
  conda run -n mt python eval/fad.py --e2 --instrument glass
  conda run -n mt python eval/fad.py --ref-dir X --eval-dir Y
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUILL = Path(__file__).resolve().parents[1]
SR = 16000
FPS = 250
MIN_DUR_S = 1.0


# ─── Common utilities ─────────────────────────────────────────────────────────────── #

def _ensure_16k_mono(src_dir: Path, dst_dir: Path, sr_out: int = SR):
    """Resample every .wav in src_dir to 16kHz mono and write it to dst_dir."""
    import librosa

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.glob("*.wav")):
        y, _ = librosa.load(f, sr=sr_out, mono=True)
        sf.write(str(dst_dir / f.name), y, sr_out)
        count += 1
    return count


def compute_fad(ref_dir: str | Path, eval_dir: str | Path,
                model: str = "vggish") -> float:
    """VGGish FAD between two directories (lower is better). Returns -1.0 on failure."""
    from frechet_audio_distance import FrechetAudioDistance

    fad = FrechetAudioDistance(model_name=model, sample_rate=SR,
                              use_pca=False, use_activation=False,
                              verbose=False, audio_load_worker=1)
    return float(fad.score(str(ref_dir), str(eval_dir)))


# ─── Frame-level MFCC ────────────────────────────────────────────────────────────── #

def _collect_mfcc_frames_per_file(wav_dir: Path) -> list[np.ndarray]:
    """Each ≥1 s wav → MFCC frame matrix (T_i, 20); returns a list, one entry per file."""
    import librosa

    result = []
    for f in sorted(wav_dir.glob("*.wav")):
        y, _ = librosa.load(f, sr=SR, mono=True)
        if len(y) < int(MIN_DUR_S * SR):
            continue
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
        result.append(mfcc.T)
    return result


def _frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """Fréchet distance between two multivariate Gaussians."""
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def _mfcc_fad(dir_a: Path, dir_b: Path) -> float | None:
    """Frame-level MFCC Fréchet distance."""
    fa = _collect_mfcc_frames_per_file(dir_a)
    fb = _collect_mfcc_frames_per_file(dir_b)
    if not fa or not fb:
        return None
    a = np.concatenate(fa)
    b = np.concatenate(fb)
    return _frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                             b.mean(0), np.cov(b, rowvar=False))


# ─── E2: Glass ─────────────────────────────────────────────────────────────── #

def _cut_glass_reference(out_dir: Path) -> int:
    """Cut glass_perf_studio.wav into segments by style library t_on/dur; keep only ≥1 s."""
    import librosa

    wav_path = QUILL / "assets/glass/glass_perf_studio.wav"
    npz_path = QUILL / "checkpoints/glass_style.npz"
    if not wav_path.exists() or not npz_path.exists():
        print("  SKIP glass reference: missing files")
        return 0

    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    d = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(d["meta"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, m in enumerate(meta):
        if m["dur"] < MIN_DUR_S:
            continue
        s0 = int(m["t_on"] * SR)
        s1 = int((m["t_on"] + m["dur"]) * SR)
        seg = y[s0:s1]
        if len(seg) < int(MIN_DUR_S * SR):
            continue
        sf.write(str(out_dir / f"glass_ref_{i:02d}.wav"), seg, SR)
        count += 1
    return count


def _render_quill_glass(out_dir: Path) -> int:
    """Render the glass style library with ddsp_glass.pt, sustain envelope, ≥1 s."""
    import torch

    from neural.neural_take import ConditionedNN
    from neural.perform import control_from_behavior, render

    ckpt = QUILL / "checkpoints/ddsp_glass.pt"
    lib_path = QUILL / "checkpoints/glass_style.npz"
    if not ckpt.exists() or not lib_path.exists():
        print("  SKIP Quill glass: missing checkpoint or library")
        return 0

    d = np.load(str(lib_path), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    nn = ConditionedNN(d["features"], d["trajectories"])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    count = 0
    for i, m in enumerate(meta):
        if m["dur"] < MIN_DUR_S:
            continue
        notes = [{"midi": m["midi"], "dur": m["dur"], "art": m.get("art", 0)}]
        with torch.no_grad():
            f0, loud = control_from_behavior(
                notes, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0,
            )
            y = render(str(ckpt), f0, loud)
        if len(y) < int(MIN_DUR_S * SR):
            continue
        sf.write(str(out_dir / f"glass_quill_{i:02d}.wav"), y, SR)
        count += 1
    return count


def _build_sampler_glass(meta: list[dict], out_dir: Path) -> int:
    """Glass sampler: timbre_train single note → pitch-shift to each target pitch, ≥1 s."""
    import librosa

    tt_dir = QUILL / "assets/glass/timbre_train"
    if not tt_dir.exists():
        print(f"  SKIP sampler glass: no timbre_train at {tt_dir}")
        return 0
    src_files = sorted(tt_dir.glob("*.wav"))
    if not src_files:
        return 0

    src_path = src_files[len(src_files) // 2]
    y_src, _ = librosa.load(str(src_path), sr=SR, mono=True)
    f0_raw = librosa.pyin(y_src, fmin=80, fmax=4000, sr=SR)[0]
    valid_f0 = f0_raw[~np.isnan(f0_raw)]
    if len(valid_f0) == 0:
        return 0
    src_midi = 12 * np.log2(np.median(valid_f0) / 440.0) + 69
    print(f"  sampler source: {src_path.name}, midi≈{src_midi:.1f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, m in enumerate(meta):
        if m["dur"] < MIN_DUR_S:
            continue
        shift_st = m["midi"] - src_midi
        dur_samp = int(m["dur"] * SR)
        seg = y_src[:dur_samp]
        if len(seg) < int(MIN_DUR_S * SR):
            seg = y_src[: int(MIN_DUR_S * SR)]
        shifted = librosa.effects.pitch_shift(
            seg.astype(np.float32), sr=SR, n_steps=float(shift_st)
        )
        peak = np.abs(shifted).max()
        if peak > 0:
            shifted = shifted / peak * 0.9
        sf.write(str(out_dir / f"glass_sampler_{i:02d}.wav"),
                 shifted.astype(np.float32), SR)
        count += 1
    return count


# ─── E2: Guitar helpers ───────────────────────────────────────────────────── #

def _guitar_phrases(meta: list[dict], seg_dur: float = 3.0,
                    max_phrases: int = 200):
    """Take the first phrase of consecutive notes that reaches ≥seg_dur in each track (1 phrase/track).

    Yields (track_id, 0, [{'midi', 'dur', 'art'}]).
    Quill and sampler share this list so their note sequences are identical.
    """
    by_track: dict[str, list[dict]] = {}
    for m in meta:
        by_track.setdefault(m["track_id"], []).append(m)
    for notes in by_track.values():
        notes.sort(key=lambda x: x["note_idx"])

    count = 0
    for track_id in sorted(by_track):
        phrase: list[dict] = []
        phrase_dur = 0.0
        for m in by_track[track_id]:
            dur = min(m["dur"], 2.0)
            phrase.append({"midi": m["midi"], "dur": dur,
                           "art": m.get("art", 0)})
            phrase_dur += dur
            if phrase_dur >= seg_dur:
                yield track_id, 0, phrase
                count += 1
                break
        else:
            if phrase and phrase_dur >= MIN_DUR_S:
                yield track_id, 0, phrase
                count += 1
        if count >= max_phrases:
            return


def _extract_guitar_source_note(meta: list[dict],
                                guitarset_dir: Path) -> tuple:
    """Cut one isolated sustained note from GuitarSet as the sampler source.

    Pick a note with dur ≥ 1.5 s and note_idx < 3 (near the start of the track),
    and verify its actual pitch with pYIN. Returns (audio_array, midi_float).
    """
    import librosa

    candidates = [m for m in meta if m["dur"] >= 1.5 and m["note_idx"] < 3]
    if not candidates:
        candidates = sorted(meta, key=lambda m: m["dur"], reverse=True)[:20]
    candidates.sort(key=lambda m: m["dur"], reverse=True)
    src = candidates[0]

    wav_path = guitarset_dir / f"{src['track_id']}_mic.wav"
    if not wav_path.exists():
        return None, None
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)

    track_notes = sorted(
        [m for m in meta if m["track_id"] == src["track_id"]],
        key=lambda m: m["note_idx"],
    )
    t_approx = sum(n["dur"] for n in track_notes
                   if n["note_idx"] < src["note_idx"])
    s0 = int(t_approx * SR)
    s1 = s0 + int(src["dur"] * SR)
    seg = y[max(0, s0): min(s1, len(y))]
    if len(seg) < SR:
        seg = y[: int(1.5 * SR)]

    f0_raw = librosa.pyin(seg, fmin=60, fmax=1000, sr=SR)[0]
    valid = f0_raw[~np.isnan(f0_raw)]
    src_midi = (12 * np.log2(np.median(valid) / 440.0) + 69
                if len(valid) > 0 else src["midi"])
    print(f"  sampler source: {src['track_id']} idx={src['note_idx']} "
          f"dur={src['dur']:.2f}s midi≈{src_midi:.1f}")
    return seg, src_midi


# ─── E2: Guitar ───────────────────────────────────────────────────────────── #

def _cut_guitar_reference(out_dir: Path, max_segs: int = 200,
                          seg_dur: float = 3.0,
                          guitarset_dir: Path | None = None) -> int:
    """Cut 1 random segment of seg_dur seconds from each track, covering all tracks evenly."""
    import librosa

    gs_dir = guitarset_dir or Path.home() / "Desktop/MT/datasets/GuitarSet_mic"
    npz_path = QUILL / "checkpoints/guitarset_trajectories.npz"
    if not gs_dir.exists() or not npz_path.exists():
        print("  SKIP guitar reference: missing GuitarSet_mic or npz")
        return 0

    d = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    tracks = sorted(set(m["track_id"] for m in meta))

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    count = 0
    for track_id in tracks:
        wav_path = gs_dir / f"{track_id}_mic.wav"
        if not wav_path.exists():
            continue
        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        dur_samp = int(seg_dur * SR)
        if len(y) < dur_samp:
            continue
        start = int(rng.integers(0, max(1, len(y) - dur_samp)))
        seg = y[start: start + dur_samp]
        sf.write(str(out_dir / f"guitar_ref_{track_id}.wav"), seg, SR)
        count += 1
        if count >= max_segs:
            break
    return count


def _render_quill_guitar(out_dir: Path,
                         phrases: list[tuple]) -> int:
    """Render guitarset style library phrases from the phrases list."""
    import torch

    from neural.neural_take import ConditionedNN
    from neural.perform import control_from_behavior, render

    ckpt = QUILL / "checkpoints/ddsp_guitarset.pt"
    lib_path = QUILL / "checkpoints/guitarset_trajectories.npz"
    if not ckpt.exists() or not lib_path.exists():
        print("  SKIP Quill guitar: missing checkpoint or library")
        return 0

    d = np.load(str(lib_path), allow_pickle=True)
    nn = ConditionedNN(d["features"], d["trajectories"])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    count = 0
    for track_id, seg_idx, notes in phrases:
        with torch.no_grad():
            f0, loud = control_from_behavior(
                notes, nn, pitch_glide=True,
                envelope="pluck", traj_gain=1.0,
            )
            y = render(str(ckpt), f0, loud)
        if len(y) < int(MIN_DUR_S * SR):
            continue
        sf.write(
            str(out_dir / f"guitar_quill_{track_id}_{seg_idx:03d}.wav"),
            y, SR,
        )
        count += 1
    return count


def _build_sampler_guitar(out_dir: Path,
                          phrases: list[tuple],
                          src_audio: np.ndarray,
                          src_midi: float) -> int:
    """Per-note pitch-shift of the single source + gating, joined into phrases with the same structure as Quill."""
    import librosa

    out_dir.mkdir(parents=True, exist_ok=True)
    fade = int(0.005 * SR)  # 5 ms fade
    count = 0
    for track_id, seg_idx, notes in phrases:
        segments = []
        for note in notes:
            dur_samp = int(note["dur"] * SR)
            if dur_samp <= len(src_audio):
                seg = src_audio[:dur_samp].copy()
            else:
                seg = np.pad(src_audio, (0, dur_samp - len(src_audio)))
            shifted = librosa.effects.pitch_shift(
                seg.astype(np.float32), sr=SR,
                n_steps=float(note["midi"] - src_midi),
            )
            if len(shifted) > 2 * fade:
                shifted[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                shifted[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            segments.append(shifted)
        phrase = np.concatenate(segments)
        peak = np.abs(phrase).max()
        if peak > 0:
            phrase = phrase / peak * 0.9
        if len(phrase) < int(MIN_DUR_S * SR):
            continue
        sf.write(
            str(out_dir / f"guitar_sampler_{track_id}_{seg_idx:03d}.wav"),
            phrase.astype(np.float32), SR,
        )
        count += 1
    return count


# ─── Split-half floor ────────────────────────────────────────────────────── #

def _split_half_fad(wav_dir: Path, metric: str = "mfcc") -> float | None:
    """FAD between two halves of the reference set, used as the split-half floor."""
    files = sorted(wav_dir.glob("*.wav"))
    if len(files) < 4:
        return None
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(files))
    mid = len(files) // 2
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a_dir, b_dir = tmp / "a", tmp / "b"
        a_dir.mkdir()
        b_dir.mkdir()
        for i in idx[:mid]:
            f = files[i]
            sf.write(str(a_dir / f.name), *sf.read(str(f)))
        for i in idx[mid:]:
            f = files[i]
            sf.write(str(b_dir / f.name), *sf.read(str(f)))
        if metric == "vggish":
            return compute_fad(a_dir, b_dir)
        return _mfcc_fad(a_dir, b_dir)


# ─── Bootstrap ───────────────────────────────────────────────────────────── #

def _bootstrap_comparison(ref_dir: Path, quill_dir: Path,
                          sampler_dir: Path,
                          n_boot: int = 1000,
                          seed: int = 42) -> dict | None:
    """File-level resampling + frame-level MFCC FAD bootstrap."""
    ref = _collect_mfcc_frames_per_file(ref_dir)
    quill = _collect_mfcc_frames_per_file(quill_dir)
    sampler = _collect_mfcc_frames_per_file(sampler_dir)
    if not ref or not quill or not sampler:
        return None

    rng = np.random.default_rng(seed)
    q_wins = 0
    q_fads: list[float] = []
    s_fads: list[float] = []

    for _ in range(n_boot):
        ir = rng.choice(len(ref), len(ref), replace=True)
        iq = rng.choice(len(quill), len(quill), replace=True)
        is_ = rng.choice(len(sampler), len(sampler), replace=True)

        rp = np.concatenate([ref[i] for i in ir])
        qp = np.concatenate([quill[i] for i in iq])
        sp = np.concatenate([sampler[i] for i in is_])

        mu_r, cov_r = rp.mean(0), np.cov(rp, rowvar=False)
        mu_q, cov_q = qp.mean(0), np.cov(qp, rowvar=False)
        mu_s, cov_s = sp.mean(0), np.cov(sp, rowvar=False)

        fq = _frechet_distance(mu_r, cov_r, mu_q, cov_q)
        fs = _frechet_distance(mu_r, cov_r, mu_s, cov_s)
        q_fads.append(fq)
        s_fads.append(fs)
        if fq < fs:
            q_wins += 1

    q_arr = np.array(q_fads)
    s_arr = np.array(s_fads)
    return {
        "quill_win_rate": round(q_wins / n_boot, 3),
        "quill_ci": [round(float(np.percentile(q_arr, 2.5)), 1),
                     round(float(np.percentile(q_arr, 97.5)), 1)],
        "sampler_ci": [round(float(np.percentile(s_arr, 2.5)), 1),
                       round(float(np.percentile(s_arr, 97.5)), 1)],
    }


# ─── E2 main ───────────────────────────────────────────────────────────── #

def run_e2(out_dir: Path, instrument: str = "all",
           guitarset_dir: Path | None = None):
    """E2 FAD: 3 arms × instruments. VGGish is the primary metric, frame-level MFCC is auxiliary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    instruments = ["glass", "guitar"] if instrument == "all" else [instrument]

    for inst in instruments:
        print(f"\n{'='*60}")
        print(f"  E2 FAD: {inst}")
        print(f"{'='*60}")

        ref_dir = out_dir / inst / "reference"
        quill_dir = out_dir / inst / "quill"
        sampler_dir = out_dir / inst / "sampler"

        # ── 1 reference ──────────────────────────────────────────────
        print("\n[1/5] Cutting reference segments...")
        if inst == "glass":
            n_ref = _cut_glass_reference(ref_dir)
        else:
            n_ref = _cut_guitar_reference(ref_dir,
                                          guitarset_dir=guitarset_dir)
        print(f"  reference: {n_ref} segments")

        # ── 2 Quill renders ──────────────────────────────────────────
        print(f"\n[2/5] Rendering Quill {inst}...")
        if inst == "glass":
            n_quill = _render_quill_glass(quill_dir)
        else:
            npz_path = QUILL / "checkpoints/guitarset_trajectories.npz"
            d_lib = np.load(str(npz_path), allow_pickle=True)
            meta_guitar = json.loads(str(d_lib["meta"]))
            phrases = list(_guitar_phrases(meta_guitar,
                                           max_phrases=n_ref))
            n_quill = _render_quill_guitar(quill_dir, phrases)
        print(f"  Quill: {n_quill} renders")

        # ── 3 sampler baseline ───────────────────────────────────────
        print("\n[3/5] Building sampler baseline...")
        if inst == "glass":
            npz_path = QUILL / "checkpoints/glass_style.npz"
            if npz_path.exists():
                d_lib = np.load(str(npz_path), allow_pickle=True)
                meta_glass = json.loads(str(d_lib["meta"]))
                n_sampler = _build_sampler_glass(meta_glass, sampler_dir)
            else:
                n_sampler = 0
        else:
            gs_dir = (guitarset_dir
                      or Path.home() / "Desktop/MT/datasets/GuitarSet_mic")
            src_audio, src_midi = _extract_guitar_source_note(
                meta_guitar, gs_dir)
            if src_audio is not None:
                n_sampler = _build_sampler_guitar(
                    sampler_dir, phrases, src_audio, src_midi)
            else:
                n_sampler = 0
        print(f"  sampler: {n_sampler} segments")

        if min(n_ref, n_quill) < 2:
            print(f"\n  Not enough audio for {inst} "
                  f"(ref={n_ref}, quill={n_quill})")
            continue

        # ── 4 FAD ────────────────────────────────────────────────────
        print("\n[4/5] Computing FAD...")

        # VGGish (primary metric)
        try:
            fad_vgg_q = compute_fad(ref_dir, quill_dir)
            if fad_vgg_q >= 0:
                results[f"{inst}_vggish_fad_quill"] = round(fad_vgg_q, 2)
                print(f"  VGGish Quill:   {fad_vgg_q:.2f}")
            if n_sampler >= 2:
                fad_vgg_s = compute_fad(ref_dir, sampler_dir)
                if fad_vgg_s >= 0:
                    results[f"{inst}_vggish_fad_sampler"] = round(fad_vgg_s, 2)
                    print(f"  VGGish Sampler: {fad_vgg_s:.2f}")
            vgg_floor = _split_half_fad(ref_dir, metric="vggish")
            if vgg_floor is not None and vgg_floor >= 0:
                results[f"{inst}_vggish_fad_floor"] = round(vgg_floor, 2)
                print(f"  VGGish floor:   {vgg_floor:.2f}")
        except Exception as e:
            print(f"  VGGish failed: {e}")

        # MFCC (frame-level)
        fad_mfcc_q = _mfcc_fad(ref_dir, quill_dir)
        if fad_mfcc_q is not None:
            results[f"{inst}_mfcc_fad_quill"] = round(fad_mfcc_q, 2)
            print(f"  MFCC Quill:     {fad_mfcc_q:.2f}")
        if n_sampler >= 2:
            fad_mfcc_s = _mfcc_fad(ref_dir, sampler_dir)
            if fad_mfcc_s is not None:
                results[f"{inst}_mfcc_fad_sampler"] = round(fad_mfcc_s, 2)
                print(f"  MFCC Sampler:   {fad_mfcc_s:.2f}")
        mfcc_floor = _split_half_fad(ref_dir, metric="mfcc")
        if mfcc_floor is not None:
            results[f"{inst}_mfcc_fad_floor"] = round(mfcc_floor, 2)
            print(f"  MFCC floor:     {mfcc_floor:.2f}")

        # ── 5 bootstrap ──────────────────────────────────────────────
        if n_sampler >= 2:
            print("\n[5/5] Bootstrap (MFCC frame-level)...")
            boot = _bootstrap_comparison(ref_dir, quill_dir, sampler_dir)
            if boot:
                results[f"{inst}_bootstrap_mfcc"] = boot
                print(f"  Quill win rate: {boot['quill_win_rate']:.1%}")
                print(f"  Quill CI: {boot['quill_ci']}")
                print(f"  Sampler CI: {boot['sampler_ci']}")

    out_json = out_dir / "e2_fad_results.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults -> {out_json}")
    print(json.dumps(results, indent=2))
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="FAD evaluation for Quill")
    parser.add_argument("--ref-dir", type=Path)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--model", default="vggish",
                        choices=["vggish", "pann", "clap", "encodec"])
    parser.add_argument("--e2", action="store_true")
    parser.add_argument("--instrument", default="all",
                        choices=["all", "glass", "guitar"])
    parser.add_argument("--out-dir", type=Path, default=Path("eval/fad_output"))
    parser.add_argument("--guitarset-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.e2:
        run_e2(args.out_dir, instrument=args.instrument,
               guitarset_dir=args.guitarset_dir)
    elif args.ref_dir and args.eval_dir:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            r16, e16 = tmp / "ref_16k", tmp / "eval_16k"
            nr = _ensure_16k_mono(args.ref_dir, r16)
            ne = _ensure_16k_mono(args.eval_dir, e16)
            print(f"Loaded {nr} reference, {ne} eval files")
            score = compute_fad(r16, e16, model=args.model)
            print(f"\nFAD ({args.model}): {score:.2f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
