"""FAD (Fréchet Audio Distance) evaluation: the standard distribution-level metric for timbre fidelity.

FAD, proposed by Kilgour et al. 2019, is the standard comparable metric in audio generation work such as DDSP/RAVE/AFTER/TokenSynth.
This script computes the FAD between Quill resynthesis and real recordings, and compares it with the sampler baseline.

Usage:
  python eval/fad.py                          # use the built-in ear_test data
  python eval/fad.py --ref-dir path/to/ref --eval-dir path/to/eval
  python eval/fad.py --generate               # generate evaluation pairs automatically from snapshots

Dependency: frechet_audio_distance (pip install frechet_audio_distance)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000


def _ensure_16k_mono(src_dir: Path, dst_dir: Path, sr_out: int = SR):
    """Resample every .wav in src_dir to 16kHz mono and write it to dst_dir."""
    import librosa

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.glob("*.wav")):
        y, sr_in = librosa.load(f, sr=sr_out, mono=True)
        out = dst_dir / f.name
        sf.write(str(out), y, sr_out)
        count += 1
    return count


def compute_fad(ref_dir: str | Path, eval_dir: str | Path,
                model: str = "vggish") -> float:
    """Compute the FAD between two directories. Returns a float (lower is better)."""
    from frechet_audio_distance import FrechetAudioDistance

    fad = FrechetAudioDistance(model_name=model, sample_rate=SR,
                              use_pca=False, use_activation=False,
                              verbose=False, audio_load_worker=1)
    score = fad.score(str(ref_dir), str(eval_dir))
    return float(score)


def generate_pairs(out_base: Path, snap_paths: list[Path] | None = None,
                   n_pitches: int = 8, dur_s: float = 2.0):
    """Generate FAD evaluation pairs from snapshots: Quill resynthesis vs sampler transposition.

    Each snapshot is rendered at n_pitches different pitches (evenly spread within ±1 octave),
    written to the out_base/{quill,sampler}/ directories.
    """
    import librosa

    from snapshot import load_snapshot
    from synth.render import render

    ref_dir = out_base / "reference"
    quill_dir = out_base / "quill"
    sampler_dir = out_base / "sampler"
    for d in [ref_dir, quill_dir, sampler_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if snap_paths is None:
        snap_paths = list(Path("checkpoints").glob("*.npz"))
        snap_paths += list(Path("assets").rglob("snapshot*.npz"))

    for sp in snap_paths:
        snap = load_snapshot(str(sp))
        f0_ref = snap["f0_ref"]
        name = sp.stem
        n = int(dur_s * 44100)

        semitones = np.linspace(-12, 12, n_pitches)
        for i, st in enumerate(semitones):
            f0_tgt = f0_ref * 2.0 ** (st / 12.0)
            tag = f"{name}_p{i}"

            quill_y = render(snap, f0_tgt, length=n, mode="formant", add_attack=True)
            sf.write(str(quill_dir / f"{tag}.wav"), quill_y, 44100)

            ref_y = render(snap, f0_ref, length=n, mode="formant", add_attack=True)
            shifted = librosa.effects.pitch_shift(
                ref_y.astype(np.float32), sr=44100, n_steps=float(st)
            )
            sf.write(str(sampler_dir / f"{tag}.wav"), shifted, 44100)
            sf.write(str(ref_dir / f"{tag}.wav"), ref_y, 44100)

    return ref_dir, quill_dir, sampler_dir


def run_builtin():
    """Compute E2 FAD with ear_test audio: Quill vs real recordings, sampler vs real recordings.

    Corrected reference: each system vs real recordings; the distance between the two systems is no longer computed.
    quill/sampler come from the ear_test audio; real recordings come from the originals in the resynth_check and ab directories.
    """
    base = Path(__file__).resolve().parents[1]
    ear_ab = base / "assets/ear_tests/ab_vs_sampler"
    ear_st = base / "assets/ear_tests/structural"
    ear_resynth = base / "assets/ear_tests/resynth_check"
    midi_ref = base / "checkpoints/midi_ddsp_ref"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_dir = tmp / "reference"
        q_dir = tmp / "quill"
        s_dir = tmp / "sampler"
        ref_dir.mkdir()
        q_dir.mkdir()
        s_dir.mkdir()

        # Quill renders
        if ear_ab.exists():
            for src in ear_ab.glob("*quill*.wav"):
                shutil.copy2(src, q_dir / src.name)
        if ear_st.exists():
            for src in ear_st.glob("*quill*.wav"):
                shutil.copy2(src, q_dir / src.name)

        # Sampler baseline renders
        if ear_ab.exists():
            for src in ear_ab.glob("*sampler*.wav"):
                shutil.copy2(src, s_dir / src.name)
        if ear_st.exists():
            for src in ear_st.glob("*sampler*.wav"):
                shutil.copy2(src, s_dir / src.name)

        # Reference real recordings
        if ear_resynth.exists():
            for src in ear_resynth.glob("*SOURCE*original*.wav"):
                shutil.copy2(src, ref_dir / src.name)
        if midi_ref.exists():
            gt = midi_ref / "ground-truth.wav"
            if gt.exists():
                shutil.copy2(gt, ref_dir / gt.name)
            fs = midi_ref / "fluidsynth.wav"
            if fs.exists():
                shutil.copy2(fs, s_dir / f"mididdsp_ref_{fs.name}")

        nref = len(list(ref_dir.glob("*.wav")))
        nq = len(list(q_dir.glob("*.wav")))
        ns = len(list(s_dir.glob("*.wav")))
        print(f"files: {nref} reference, {nq} quill, {ns} sampler")

        if nref < 2 or nq < 2 or ns < 2:
            print("Not enough audio files for three-way FAD (need >=2 each)")
            print("Falling back to MFCC-based Fréchet distance...")
            _run_mfcc_fad(ref_dir, q_dir, s_dir)
            return

        r16 = tmp / "ref_16k"
        q16 = tmp / "quill_16k"
        s16 = tmp / "sampler_16k"
        _ensure_16k_mono(ref_dir, r16)
        _ensure_16k_mono(q_dir, q16)
        _ensure_16k_mono(s_dir, s16)

        fad_quill = compute_fad(r16, q16)
        fad_sampler = compute_fad(r16, s16)
        print(f"\nFAD (Quill resynth vs real):   {fad_quill:.2f}")
        print(f"FAD (Sampler baseline vs real): {fad_sampler:.2f}")
        if fad_quill < fad_sampler:
            print(f"  Quill closer to real by {fad_sampler - fad_quill:.2f}")
        else:
            print(f"  Sampler closer to real by {fad_quill - fad_sampler:.2f}")


def _mfcc_stats(wav_dir: Path):
    """Compute mean and covariance of MFCC features across all WAVs in a dir."""
    import librosa
    all_feats = []
    for f in sorted(wav_dir.glob("*.wav")):
        y, sr = librosa.load(f, sr=SR, mono=True)
        if len(y) < 2048:
            continue
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
        all_feats.append(mfcc.mean(axis=1))
    if len(all_feats) < 2:
        return None, None
    feats = np.stack(all_feats)
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def _frechet_distance(mu1, sigma1, mu2, sigma2):
    """Fréchet distance between two multivariate Gaussians."""
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def _run_mfcc_fad(ref_dir, quill_dir, sampler_dir):
    """MFCC-based Fréchet distance (fallback when frechet_audio_distance is not installed)."""
    mu_r, cov_r = _mfcc_stats(ref_dir)
    mu_q, cov_q = _mfcc_stats(quill_dir)
    mu_s, cov_s = _mfcc_stats(sampler_dir)

    if mu_r is None:
        print("Not enough reference audio for MFCC FAD")
        return
    results = {}
    if mu_q is not None:
        fd_q = _frechet_distance(mu_r, cov_r, mu_q, cov_q)
        results["mfcc_fad_quill_vs_real"] = round(fd_q, 2)
        print(f"\nMFCC-FAD (Quill vs real):   {fd_q:.2f}")
    if mu_s is not None:
        fd_s = _frechet_distance(mu_r, cov_r, mu_s, cov_s)
        results["mfcc_fad_sampler_vs_real"] = round(fd_s, 2)
        print(f"MFCC-FAD (Sampler vs real): {fd_s:.2f}")
    return results


def run_e2(out_dir: Path, model: str = "vggish"):
    """E2: corrected FAD reference using DDSP resynthesis.

    Collect three groups from checkpoints/midi_ddsp_ref/ and the ear_test audio:
    real recordings / Quill resynthesis / sampler baseline, and compute the FAD of each vs real recordings.
    """
    import json as json_mod

    base = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = out_dir / "reference"
    quill_dir = out_dir / "quill"
    sampler_dir = out_dir / "sampler"
    for d in [ref_dir, quill_dir, sampler_dir]:
        d.mkdir(exist_ok=True)

    # Collect reference, quill, and sampler audio
    midi_ref = base / "checkpoints/midi_ddsp_ref"
    ear_ab = base / "assets/ear_tests/ab_vs_sampler"
    ear_st = base / "assets/ear_tests/structural"
    ear_resynth = base / "assets/ear_tests/resynth_check"

    collected = {"reference": 0, "quill": 0, "sampler": 0}

    if midi_ref.exists():
        gt = midi_ref / "ground-truth.wav"
        if gt.exists():
            shutil.copy2(gt, ref_dir / "mididdsp_ground_truth.wav")
            collected["reference"] += 1
        for name in ["midi-ddsp.wav", "ddsp_inference.wav", "midi2params.wav"]:
            src = midi_ref / name
            if src.exists():
                shutil.copy2(src, quill_dir / f"mididdsp_ref_{name}")
                collected["quill"] += 1
        fs = midi_ref / "fluidsynth.wav"
        if fs.exists():
            shutil.copy2(fs, sampler_dir / "mididdsp_ref_fluidsynth.wav")
            collected["sampler"] += 1
        ab = midi_ref / "ableton.wav"
        if ab.exists():
            shutil.copy2(ab, sampler_dir / "mididdsp_ref_ableton.wav")
            collected["sampler"] += 1

    if ear_resynth.exists():
        for src in ear_resynth.glob("*SOURCE*.wav"):
            shutil.copy2(src, ref_dir / src.name)
            collected["reference"] += 1
        for src in ear_resynth.glob("*RESYNTH*.wav"):
            shutil.copy2(src, quill_dir / src.name)
            collected["quill"] += 1

    if ear_ab.exists():
        for src in ear_ab.glob("*quill*.wav"):
            shutil.copy2(src, quill_dir / src.name)
            collected["quill"] += 1
        for src in ear_ab.glob("*sampler*.wav"):
            shutil.copy2(src, sampler_dir / src.name)
            collected["sampler"] += 1

    if ear_st.exists():
        for src in ear_st.glob("*quill*.wav"):
            shutil.copy2(src, quill_dir / src.name)
            collected["quill"] += 1
        for src in ear_st.glob("*sampler*.wav"):
            shutil.copy2(src, sampler_dir / src.name)
            collected["sampler"] += 1

    print(f"Collected: {collected}")

    results = {}

    # MFCC-based FAD (always available)
    mfcc_results = _run_mfcc_fad(ref_dir, quill_dir, sampler_dir)
    if mfcc_results:
        results.update(mfcc_results)

    # VGGish FAD (if frechet_audio_distance installed)
    try:
        with tempfile.TemporaryDirectory() as tmp16:
            tmp16 = Path(tmp16)
            r16 = tmp16 / "ref_16k"
            q16 = tmp16 / "quill_16k"
            s16 = tmp16 / "sampler_16k"
            nr = _ensure_16k_mono(ref_dir, r16)
            nq = _ensure_16k_mono(quill_dir, q16)
            ns = _ensure_16k_mono(sampler_dir, s16)
            if min(nr, nq, ns) >= 2:
                fad_q = compute_fad(r16, q16, model=model)
                fad_s = compute_fad(r16, s16, model=model)
                results[f"fad_{model}_quill_vs_real"] = round(fad_q, 2)
                results[f"fad_{model}_sampler_vs_real"] = round(fad_s, 2)
                print(f"\nFAD-{model} (Quill vs real):   {fad_q:.2f}")
                print(f"FAD-{model} (Sampler vs real): {fad_s:.2f}")
    except ImportError:
        print("\nfrechet_audio_distance not installed, using MFCC-FAD only")

    out_json = out_dir / "e2_fad_results.json"
    out_json.write_text(json_mod.dumps(results, indent=2) + "\n")
    print(f"\nResults -> {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="FAD evaluation for Quill")
    parser.add_argument("--ref-dir", type=Path, help="reference audio directory")
    parser.add_argument("--eval-dir", type=Path, help="evaluation audio directory")
    parser.add_argument("--model", default="vggish", choices=["vggish", "pann", "clap", "encodec"])
    parser.add_argument("--generate", action="store_true",
                        help="generate eval pairs from checkpoints")
    parser.add_argument("--e2", action="store_true",
                        help="E2: corrected FAD (each system vs real recordings)")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/fad_output"))
    parser.add_argument("--builtin", action="store_true",
                        help="use built-in ear_test audio (default if no args)")
    args = parser.parse_args()

    if args.e2:
        run_e2(args.out_dir, model=args.model)
    elif args.generate:
        print("Generating evaluation pairs from checkpoints...")
        ref_dir, quill_dir, sampler_dir = generate_pairs(args.out_dir)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            r16 = tmp / "ref_16k"
            q16 = tmp / "quill_16k"
            s16 = tmp / "sampler_16k"
            nr = _ensure_16k_mono(ref_dir, r16)
            nq = _ensure_16k_mono(quill_dir, q16)
            ns = _ensure_16k_mono(sampler_dir, s16)
            print(f"Generated {nr} ref, {nq} quill, {ns} sampler files")

            fad_quill = compute_fad(r16, q16, model=args.model)
            fad_sampler = compute_fad(r16, s16, model=args.model)
            print(f"\n{'Model':>10}: {args.model}")
            print(f"{'FAD Quill':>10}: {fad_quill:.2f}")
            print(f"{'FAD Sampler':>10}: {fad_sampler:.2f}")
            if fad_quill < fad_sampler:
                print(f"  Quill wins by {fad_sampler - fad_quill:.2f}")
            else:
                print(f"  Sampler wins by {fad_quill - fad_sampler:.2f}")

    elif args.ref_dir and args.eval_dir:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            r16 = tmp / "ref_16k"
            e16 = tmp / "eval_16k"
            nr = _ensure_16k_mono(args.ref_dir, r16)
            ne = _ensure_16k_mono(args.eval_dir, e16)
            print(f"Loaded {nr} reference, {ne} eval files")
            score = compute_fad(r16, e16, model=args.model)
            print(f"\nFAD ({args.model}): {score:.2f}")

    else:
        run_builtin()


if __name__ == "__main__":
    main()
