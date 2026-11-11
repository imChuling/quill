"""E1: MIDI-DDSP cross-comparison — evaluate four conditions × four melodies.

Reads pre-rendered wav files (not raw synth params) and computes:
  - Timbre signature: MFCC (same as E3)
  - Behavior signature: per-note vib_energy + drift + total_rms (same as E3),
    reported both as aggregate L2 and per-feature

Conditions:
  QB×QD  Quill behavior → Quill DDSP                   (rendered here, mt env)
  MB×QD  MIDI-DDSP behavior → Quill DDSP               (rendered here, mt env)
  QB×PG  Quill f0+amps injected into MIDI-DDSP
         processor group (spectral envelope unchanged)  (from mididdsp_render.py)
  MB×MD  MIDI-DDSP full stack                           (from mididdsp_render.py)

Note: QB×PG is NOT "Quill behavior × MIDI-DDSP decoder" — the MIDI-DDSP
processor group does not re-infer harmonic_distribution / noise_magnitudes
from the injected controls. The spectral envelope stays from the original
MIDI-DDSP synthesis.

Prerequisites:
  1. conda run -n mt python tools/e1_export_quill_controls.py [--no-slide]
  2. conda run -n mididdsp python tools/mididdsp_render.py \\
         --quill-controls checkpoints/eval/e1_quill_controls/ [--no-slide]
  3. conda run -n mt python tools/eval_mididdsp_cross.py [--no-slide]

Usage:
    conda run -n mt python tools/eval_mididdsp_cross.py --no-slide
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
OUT = CKPT / "eval" / "e1_cross"
EXPORTS = CKPT / "eval" / "mididdsp_exports"
RENDERS = CKPT / "eval" / "e1_renders"

SR = 16000
FPS = 250

MELODIES = {
    "lyrical": [
        (62, 0.8, 1), (65, 0.4, 0), (67, 1.2, 1), (65, 0.4, 0),
        (69, 0.4, 2), (72, 1.6, 1), (67, 0.8, 0), (62, 1.6, 1),
    ],
    "scale_run": [
        (62, 0.2, 0), (64, 0.2, 0), (65, 0.2, 0), (67, 0.2, 0),
        (69, 0.2, 0), (71, 0.2, 0), (72, 0.2, 0), (74, 0.2, 0),
        (76, 0.4, 0), (74, 0.4, 0), (72, 1.6, 1),
    ],
    "leaps": [
        (57, 0.4, 0), (64, 0.4, 0), (62, 0.4, 2), (69, 0.8, 1),
        (65, 0.4, 0), (72, 0.4, 2), (71, 0.8, 0), (64, 1.2, 1),
    ],
    "folk": [
        (69, 0.4, 0), (71, 0.2, 0), (72, 0.4, 0), (69, 0.4, 0),
        (67, 0.8, 1), (64, 0.4, 0), (67, 0.4, 2), (69, 1.2, 1),
        (67, 0.4, 0), (64, 0.4, 0), (62, 1.6, 1),
    ],
}


def strip_slides(melodies):
    """Replace art=2 (slide) with art=0 (plain)."""
    return {k: [(m, d, 0 if a == 2 else a) for m, d, a in v]
            for k, v in melodies.items()}


# ---- Sanity check ----

def voiced_ratio(audio):
    import librosa
    f0, _, _ = librosa.pyin(audio, fmin=80, fmax=2000, sr=SR,
                            frame_length=1024, hop_length=160)
    return float(np.sum(~np.isnan(f0)) / max(len(f0), 1))


# ---- Metrics (same as E3: tools/eval_l0_attribution.py) ----

def analyze(audio, bounds):
    """Per-note behavior features. Identical to eval_l0_attribution.analyze()."""
    import librosa
    hop = 160
    f0, _, _ = librosa.pyin(
        audio, fmin=170, fmax=1700, sr=SR, frame_length=1024, hop_length=hop
    )
    t_f = np.arange(len(f0)) * hop / SR
    good = ~np.isnan(f0)
    if good.sum() < 4:
        return [{"vib_energy": 0, "rate": 0, "drift": 0, "total_rms": 0}] * len(bounds)
    f0i = np.interp(t_f, t_f[good], f0[good])
    out = []
    TRIM = 5
    F_RATE = 100
    for t0, t1 in bounds:
        if (t1 - t0) < 0.3:
            out.append({"vib_energy": 0.0, "rate": 0.0, "drift": 0.0, "total_rms": 0.0})
            continue
        i0, i1 = int(t0 * F_RATE) + TRIM, int(t1 * F_RATE) - TRIM
        if i1 <= i0:
            i0, i1 = int(t0 * F_RATE), int(t1 * F_RATE)
        seg = f0i[i0:i1]
        if len(seg) < 8:
            out.append({"vib_energy": 0.0, "rate": 0.0, "drift": 0.0, "total_rms": 0.0})
            continue
        cents = 1200 * np.log2(seg / (np.median(seg) + 1e-9) + 1e-12)
        win = np.hanning(len(cents))
        spec = np.abs(np.fft.rfft((cents - cents.mean()) * win))
        freqs = np.fft.rfftfreq(len(cents), d=1.0 / F_RATE)
        vib_band = (freqs >= 3.5) & (freqs <= 9.0)
        N = max(len(cents), 1)
        vib_energy = float(np.sqrt((spec[vib_band] ** 2).sum() / N ** 2))
        if vib_band.any() and spec[vib_band].max() > 0:
            rate = float(freqs[vib_band][int(np.argmax(spec[vib_band]))])
        else:
            rate = 0.0
        q = max(2, len(cents) // 4)
        drift = float(np.median(cents[-q:]) - np.median(cents[:q]))
        total_rms = float(np.sqrt((cents ** 2).mean()))
        out.append({"vib_energy": vib_energy, "rate": rate, "drift": drift, "total_rms": total_rms})
    return out


def behavior_sig(ana):
    """Per-note behavior signature vector. Same as eval_l0_attribution.behavior_sig()."""
    return np.array([[a["vib_energy"], a["drift"], a["total_rms"]] for a in ana]).ravel()


def behavior_sig_per_feature(ana):
    """Per-feature signature arrays for decomposed reporting."""
    arr = np.array([[a["vib_energy"], a["drift"], a["total_rms"]] for a in ana])
    return {
        "vib_energy": arr[:, 0],
        "drift": arr[:, 1],
        "total_rms": arr[:, 2],
    }


def timbre_sig(audio):
    """MFCC-based timbre signature. Same as eval_l0_attribution.timbre_sig()."""
    import librosa
    mf = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)[1:]
    return mf.mean(axis=1)


# ---- Quill rendering (QB×QD and MB×QD) ----

def load_quill_renderer():
    from neural.ddsp import DDSP
    from neural.neural_take import ConditionedNN

    ck = CKPT / "ddsp_urmp_vn_12k.pt"
    tr = CKPT / "urmp_vn_trajectories.npz"
    if not ck.exists() or not tr.exists():
        raise FileNotFoundError(f"need {ck} and {tr}")
    m = DDSP(dropout=0.0)
    m.load_state_dict(torch.load(str(ck), map_location="cpu", weights_only=True))
    m.eval()
    d = np.load(str(tr))
    nn = ConditionedNN(d["features"], d["trajectories"])
    return m, nn


def quill_behavior_controls(score, nn):
    from neural.perform import control_from_behavior
    notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
    with torch.no_grad():
        f0, loud = control_from_behavior(
            notes, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0,
        )
    return f0.numpy()[0, :, 0], loud.numpy()[0, :, 0]


def quill_ddsp_render(f0_hz, loudness, ddsp_model):
    T = len(f0_hz)
    f0_t = torch.from_numpy(f0_hz.astype(np.float32)).reshape(1, T, 1)
    lo_t = torch.from_numpy(loudness.astype(np.float32)).reshape(1, T, 1)
    with torch.no_grad():
        y = ddsp_model(f0_t, lo_t, noise_gain=0.6, noise_gate_db=-1.0).numpy()[0]
    peak = max(float(np.max(np.abs(y))), 1e-9)
    return (y / peak * 0.9).astype(np.float32)


def mididdsp_f0_loudness_for_quill(melody_name):
    """Load MIDI-DDSP's f0 + loudness for MB×QD condition.

    npz amplitudes are in dB. Z-score them directly — no abs/log
    (which would invert the sign: -41 dB louder than -80 dB).
    """
    path = EXPORTS / f"{melody_name}.npz"
    d = np.load(str(path), allow_pickle=False)
    f0 = np.squeeze(d["f0_hz"]).astype(np.float32)
    amps_dB = np.squeeze(d["amplitudes"]).astype(np.float32)
    loud = (amps_dB - amps_dB.mean()) / max(float(amps_dB.std()), 1e-6)
    return f0, loud


def quill_note_frames(score, fps=250, cf_frames=5):
    """Per-note (start_frame, end_frame) with Quill's crossfade overlap."""
    frames = []
    t = 0
    prev_n = None
    for _midi, dur, _art in score:
        n = max(4, int(dur * fps))
        if prev_n is not None:
            t -= min(cf_frames, prev_n // 2, n // 2)
        frames.append((t, t + n))
        t += n
        prev_n = n
    return frames


def midi_note_frames(npz_path):
    """Per-note (onset_frame, offset_frame) from MIDI-DDSP npz note_table."""
    d = np.load(str(npz_path), allow_pickle=True)
    nt = d["note_table"]
    nk = list(d["note_keys"])
    oi = nk.index("onset")
    fi = nk.index("offset")
    return [(int(row[oi]), int(row[fi])) for row in nt]


def midi_note_bounds_sec(npz_path):
    """Note bounds in seconds from MIDI-DDSP note table (for analysis)."""
    return [(s / FPS, e / FPS) for s, e in midi_note_frames(npz_path)]


def resample_per_note(signal, src_frames, dst_frames, total_dst_len):
    """Resample a 1-D signal per-note from src timing to dst timing."""
    assert len(src_frames) == len(dst_frames), \
        f"note count mismatch: {len(src_frames)} vs {len(dst_frames)}"
    out = np.zeros(total_dst_len, dtype=np.float32)
    for (s0, s1), (d0, d1) in zip(src_frames, dst_frames):
        src_seg = signal[s0:min(s1, len(signal))]
        dst_len = d1 - d0
        if len(src_seg) == 0 or dst_len == 0:
            continue
        if len(src_seg) == dst_len:
            out[d0:d1] = src_seg
        else:
            x_old = np.linspace(0, 1, len(src_seg))
            x_new = np.linspace(0, 1, dst_len)
            out[d0:d1] = np.interp(x_new, x_old, src_seg)
    return out


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-slide", action="store_true",
                        help="Replace slide (art=2) with plain (art=0)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    melodies = strip_slides(MELODIES) if args.no_slide else MELODIES
    if args.no_slide:
        print("--no-slide: art=2 → art=0")

    QUILL_CTRL = CKPT / "eval" / "e1_quill_controls"

    ddsp, nn = load_quill_renderer()
    print("Quill renderer loaded")

    FEAT_NAMES = ["vib_energy", "drift", "total_rms"]

    def _sig_dist(sigs, a, b):
        return float(np.linalg.norm(sigs[a] - sigs[b]))

    def _sig_dist_feat(sigs_feat, a, b, feat_names):
        return {fn: float(np.linalg.norm(sigs_feat[a][fn] - sigs_feat[b][fn]))
                for fn in feat_names}

    all_results = {}

    for mname, score in melodies.items():
        print(f"\n=== {mname} ({len(score)} notes) ===")
        npz_path = EXPORTS / f"{mname}.npz"
        bounds = midi_note_bounds_sec(npz_path)
        m_frames = midi_note_frames(npz_path)
        q_frames = quill_note_frames(score)
        T_midi = m_frames[-1][1]
        print(f"  MIDI bounds: {len(bounds)} notes, last offset={bounds[-1][1]:.2f}s")

        # Read Quill controls from .npy (not recompute)
        f0_path = QUILL_CTRL / f"{mname}_f0.npy"
        loud_path = QUILL_CTRL / f"{mname}_loudness.npy"
        if f0_path.exists() and loud_path.exists():
            f0_q = np.load(str(f0_path))
            lo_q = np.load(str(loud_path))
        else:
            print("  Quill .npy not found, computing from score...")
            f0_q, lo_q = quill_behavior_controls(score, nn)

        # Per-note resample Quill controls to MIDI note grid
        f0_q_midi = resample_per_note(f0_q, q_frames, m_frames, T_midi)
        lo_q_midi = resample_per_note(lo_q, q_frames, m_frames, T_midi)
        print(f"  Quill controls: {len(f0_q)} -> {T_midi} frames (per-note resampled)")

        # QB×QD: Quill behavior × Quill DDSP (MIDI timing)
        y_qbqd = quill_ddsp_render(f0_q_midi, lo_q_midi, ddsp)
        sf.write(str(OUT / f"{mname}_QB_QD.wav"), y_qbqd, SR)
        vr = voiced_ratio(y_qbqd)
        print(f"  QB×QD: {len(y_qbqd)/SR:.1f}s, voiced={vr:.0%}")
        if vr < 0.5:
            print("  ERROR: QB×QD voiced ratio too low!")
            sys.exit(1)

        # MB×QD: MIDI-DDSP behavior → Quill DDSP (truncate to note content)
        f0_md, lo_md = mididdsp_f0_loudness_for_quill(mname)
        f0_md = f0_md[:T_midi]
        lo_md = lo_md[:T_midi]
        y_mbqd = quill_ddsp_render(f0_md, lo_md, ddsp)
        sf.write(str(OUT / f"{mname}_MB_QD.wav"), y_mbqd, SR)
        vr = voiced_ratio(y_mbqd)
        print(f"  MB×QD: {len(y_mbqd)/SR:.1f}s, voiced={vr:.0%}")
        if vr < 0.5:
            print("  ERROR: MB×QD voiced ratio too low!")
            sys.exit(1)

        # QB×PG and MB×MD: from tools/mididdsp_render.py output
        # QB×PG = Quill f0+amps injected into MIDI-DDSP processor group
        #         (harmonic_distribution / noise_magnitudes unchanged)
        qbpg_path = RENDERS / f"{mname}_QB_MD.wav"
        mbmd_path = RENDERS / f"{mname}_MB_MD.wav"

        if not qbpg_path.exists() or not mbmd_path.exists():
            print(f"  SKIP: {RENDERS}/ missing QB_MD or MB_MD wav files.")
            print("  Run: conda run -n mididdsp python tools/mididdsp_render.py "
                  "--quill-controls checkpoints/eval/e1_quill_controls/")
            continue

        y_qbpg, _ = sf.read(str(qbpg_path), dtype="float32")
        y_mbmd, _ = sf.read(str(mbmd_path), dtype="float32")

        for tag, y in [("QB×PG", y_qbpg), ("MB×MD", y_mbmd)]:
            vr = voiced_ratio(y)
            print(f"  {tag}: {len(y)/SR:.1f}s, voiced={vr:.0%}")
            if vr < 0.5:
                print(f"  ERROR: {tag} voiced ratio {vr:.0%} < 50%, aborting!")
                sys.exit(1)

        # Compute signatures (all conditions use MIDI bounds)
        audios = {
            "QB_QD": y_qbqd,
            "MB_QD": y_mbqd,
            "QB_PG": y_qbpg,
            "MB_MD": y_mbmd,
        }

        bsigs, tsigs, bsigs_feat = {}, {}, {}
        for cond, y in audios.items():
            ana = analyze(y, bounds)
            bsigs[cond] = behavior_sig(ana)
            bsigs_feat[cond] = behavior_sig_per_feature(ana)
            tsigs[cond] = timbre_sig(y)

        # --- Timbre ---
        # Quill decoder: switch behavior, timbre should stay
        t_quill_switch_beh = _sig_dist(tsigs, "QB_QD", "MB_QD")
        # MIDI-DDSP PG: switch behavior (but spectral envelope is unchanged)
        t_pg_switch_beh = _sig_dist(tsigs, "QB_PG", "MB_MD")
        # Fixed behavior, different synthesis path
        t_cross_1 = _sig_dist(tsigs, "QB_QD", "QB_PG")
        t_cross_2 = _sig_dist(tsigs, "MB_QD", "MB_MD")

        # --- Behavior (aggregate L2) ---
        b_same_q = _sig_dist(bsigs, "QB_QD", "QB_PG")
        b_same_m = _sig_dist(bsigs, "MB_QD", "MB_MD")
        b_cross_1 = _sig_dist(bsigs, "QB_QD", "MB_QD")
        b_cross_2 = _sig_dist(bsigs, "QB_PG", "MB_MD")

        # --- Behavior per-feature ---
        bf_cross_1 = _sig_dist_feat(bsigs_feat, "QB_QD", "MB_QD", FEAT_NAMES)
        bf_cross_2 = _sig_dist_feat(bsigs_feat, "QB_PG", "MB_MD", FEAT_NAMES)
        bf_same_q = _sig_dist_feat(bsigs_feat, "QB_QD", "QB_PG", FEAT_NAMES)
        bf_same_m = _sig_dist_feat(bsigs_feat, "MB_QD", "MB_MD", FEAT_NAMES)

        melody_result = {
            "timbre": {
                "quill_switch_behavior": t_quill_switch_beh,
                "pg_switch_behavior": t_pg_switch_beh,
                "cross_synth_1": t_cross_1,
                "cross_synth_2": t_cross_2,
            },
            "behavior_L2": {
                "same_quill": b_same_q,
                "same_mididdsp": b_same_m,
                "cross_1": b_cross_1,
                "cross_2": b_cross_2,
                "same_mean": (b_same_q + b_same_m) / 2,
                "cross_mean": (b_cross_1 + b_cross_2) / 2,
            },
            "behavior_per_feature": {
                fn: {
                    "same_quill": bf_same_q[fn],
                    "same_mididdsp": bf_same_m[fn],
                    "cross_1": bf_cross_1[fn],
                    "cross_2": bf_cross_2[fn],
                    "same_mean": (bf_same_q[fn] + bf_same_m[fn]) / 2,
                    "cross_mean": (bf_cross_1[fn] + bf_cross_2[fn]) / 2,
                }
                for fn in FEAT_NAMES
            },
        }

        all_results[mname] = melody_result

        b_ratio = melody_result["behavior_L2"]["cross_mean"] / \
            max(melody_result["behavior_L2"]["same_mean"], 1e-9)
        print(f"  Timbre:   Quill switch_beh={t_quill_switch_beh:.1f}, "
              f"cross_synth=[{t_cross_1:.1f}, {t_cross_2:.1f}]")
        print(f"  Behavior (L2): same={melody_result['behavior_L2']['same_mean']:.1f}, "
              f"cross={melody_result['behavior_L2']['cross_mean']:.1f}, ratio={b_ratio:.1f}×")
        print("  Per-feature cross/same:")
        for fn in FEAT_NAMES:
            bf = melody_result["behavior_per_feature"][fn]
            r = bf["cross_mean"] / max(bf["same_mean"], 1e-9)
            print(f"    {fn:12s}: same={bf['same_mean']:.2f}, "
                  f"cross={bf['cross_mean']:.2f}, ratio={r:.1f}×")

    # Summary
    if all_results:
        print("\n=== Summary ===")

        # Timbre: Quill decoder robustness
        t_q_all = np.mean([r["timbre"]["quill_switch_behavior"]
                           for r in all_results.values()])
        t_c1_all = np.mean([r["timbre"]["cross_synth_1"]
                            for r in all_results.values()])
        t_c2_all = np.mean([r["timbre"]["cross_synth_2"]
                            for r in all_results.values()])
        print("Timbre (MFCC L2):")
        print(f"  Quill decoder, switch behavior:    {t_q_all:.1f}")
        print(f"  Fixed Quill beh, cross synth:      {t_c1_all:.1f}")
        print(f"  Fixed MIDI beh, cross synth:       {t_c2_all:.1f}")

        # Behavior L2
        b_same_all = np.mean([r["behavior_L2"]["same_mean"]
                              for r in all_results.values()])
        b_cross_all = np.mean([r["behavior_L2"]["cross_mean"]
                               for r in all_results.values()])
        b_ratio_all = b_cross_all / max(b_same_all, 1e-9)
        print("\nBehavior (L2):")
        print(f"  same={b_same_all:.1f}, cross={b_cross_all:.1f}, "
              f"ratio={b_ratio_all:.1f}×")

        # Per-feature
        print("\nPer-feature breakdown:")
        feat_summary = {}
        for fn in FEAT_NAMES:
            s = np.mean([r["behavior_per_feature"][fn]["same_mean"]
                         for r in all_results.values()])
            c = np.mean([r["behavior_per_feature"][fn]["cross_mean"]
                         for r in all_results.values()])
            r = c / max(s, 1e-9)
            feat_summary[fn] = {"same": round(float(s), 2),
                                "cross": round(float(c), 2),
                                "ratio": round(float(r), 2)}
            print(f"  {fn:12s}: same={s:.2f}, cross={c:.2f}, ratio={r:.1f}×")

        summary = {
            "timbre_quill_switch_beh": round(float(t_q_all), 2),
            "timbre_cross_synth_1": round(float(t_c1_all), 2),
            "timbre_cross_synth_2": round(float(t_c2_all), 2),
            "behavior_L2_same": round(float(b_same_all), 2),
            "behavior_L2_cross": round(float(b_cross_all), 2),
            "behavior_L2_ratio": round(float(b_ratio_all), 2),
            "behavior_per_feature": feat_summary,
            "no_slide": args.no_slide,
        }

        out_json = {"melodies": all_results, "summary": summary}
        out_path = OUT / "e1_cross_results.json"
        with open(str(out_path), "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
