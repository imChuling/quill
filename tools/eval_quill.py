"""Quantitative evaluation: turn the three core claims into numbers.

  1. Controllability: articulation condition -> fraction of output audio that actually exhibits that articulation (control fidelity matrix)
     After rendering, f0 is re-extracted with pyin and vibrato (3.5-9Hz micro oscillation) and slide (net drift) are measured per note.
  2. Decoupling: 2×2 transfer matrix; f0 micro-dynamics distance should cluster by "behaviour source",
     MFCC timbre distance should cluster by "timbre source": orthogonal axes = quantitative proof of decoupling.
  3. Data efficiency: control fidelity curve as the behaviour library grows from 50 to full size.

Run (locally): cd ~/Desktop/MT/project/quill && conda run -n mt python tools/eval_quill.py
Outputs: rendered audio under checkpoints/eval/ + eval_results.json + terminal tables
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
OUT = CKPT / "eval"

SR = 16000
FPS = 250

# ---- Test melody set: 4 phrases in different styles (midi, dur_s, art) ---- #
MELODIES = {
    "lyrical": [  # slow, long notes, mostly vibrato + one slide
        (62, 0.8, 1),
        (65, 0.4, 0),
        (67, 1.2, 1),
        (65, 0.4, 0),
        (69, 0.4, 2),
        (72, 1.6, 1),
        (67, 0.8, 0),
        (62, 1.6, 1),
    ],
    "scale_run": [  # fast stepwise motion, all plain, long vibrato on the last note
        (62, 0.2, 0),
        (64, 0.2, 0),
        (65, 0.2, 0),
        (67, 0.2, 0),
        (69, 0.2, 0),
        (71, 0.2, 0),
        (72, 0.2, 0),
        (74, 0.2, 0),
        (76, 0.4, 0),
        (74, 0.4, 0),
        (72, 1.6, 1),
    ],
    "leaps": [  # leaps + two slides
        (57, 0.4, 0),
        (64, 0.4, 0),
        (62, 0.4, 2),
        (69, 0.8, 1),
        (65, 0.4, 0),
        (72, 0.4, 2),
        (71, 0.8, 0),
        (64, 1.2, 1),
    ],
    "folk": [  # folk phrasing, mixed
        (69, 0.4, 0),
        (71, 0.2, 0),
        (72, 0.4, 0),
        (69, 0.4, 0),
        (67, 0.8, 1),
        (64, 0.4, 0),
        (67, 0.4, 2),
        (69, 1.2, 1),
        (67, 0.4, 0),
        (64, 0.4, 0),
        (62, 1.6, 1),
    ],
}


def strip_art(score):
    return [(m, d, 0) for (m, d, _a) in score]


# ---- Renderer ---- #
class Renderer:
    def __init__(self):
        from neural.ddsp import DDSP
        from neural.neural_take import ConditionedNN

        specs = [
            ("violin", "ddsp_urmp_vn_12k.pt", "urmp_vn_trajectories.npz", "sustain"),
            ("trumpet", "ddsp_urmp_tpt_12k.pt", "urmp_tpt_trajectories.npz", "sustain"),
            ("guitar", "ddsp_guitarset.pt", "guitarset_trajectories.npz", "pluck"),
        ]
        self.models, self.nns, self.envelopes = {}, {}, {}
        for name, ck, tr, env in specs:
            if not (CKPT / ck).exists():
                continue
            m = DDSP(dropout=0.0)
            m.load_state_dict(torch.load(str(CKPT / ck), map_location="cpu", weights_only=True))
            m.eval()
            self.models[name] = m
            d = np.load(str(CKPT / tr))
            self.nns[name] = ConditionedNN(d["features"], d["trajectories"])
            self.envelopes[name] = env
        d = np.load(str(CKPT / "urmp_vn_trajectories.npz"))
        self.vn_feats, self.vn_trajs = d["features"], d["trajectories"]

    def render(self, score, timbre="violin", behavior=None, source="v0", nn_override=None):
        """source: v0 (retrieval) / v1 (generative) / flat (no behaviour). Returns (audio, note_bounds_sec)."""
        from neural.perform import control_from_behavior

        behavior = behavior or timbre
        notes = [{"midi": m, "dur": d, "art": a} for (m, d, a) in score]

        if source == "flat":
            segs = []
            for m_, d_, _a in score:
                T = int(d_ * FPS)
                f0 = torch.full((1, T, 1), 440.0 * 2 ** ((m_ - 69) / 12.0))
                lo = torch.full((1, T, 1), 0.0)
                with torch.no_grad():
                    segs.append(
                        self.models[timbre](f0, lo, noise_gain=0.6, noise_gate_db=-1.0).numpy()[0]
                    )
            y = np.concatenate(segs)
            y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
            bounds, t = [], 0.0
            for _m, d_, _a in score:
                bounds.append((t, t + d_))
                t += d_
            return y, bounds
        else:
            beh_model = nn_override or self.nns[behavior]
            f0, loud = control_from_behavior(
                notes,
                beh_model,
                pitch_glide=False,
                envelope=self.envelopes[behavior],
                traj_gain=1.0,
            )
            with torch.no_grad():
                y = self.models[timbre](f0, loud, noise_gain=0.6, noise_gate_db=-1.0).numpy()[0]

        y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
        CF_FRAMES = 5
        bounds, t_sec = [], 0.0
        prev_n = None
        for _m, d_, _a in score:
            n_ = max(4, int(d_ * FPS))
            if prev_n is not None:
                t_sec -= min(CF_FRAMES, prev_n // 2, n_ // 2) / FPS
            bounds.append((t_sec, t_sec + d_))
            t_sec += d_
            prev_n = n_
        return y, bounds


# ---- Per-note f0 analysis ---- #
def analyze(audio, bounds):
    """pyin f0 extraction -> per note {micro_rms(cents), rate(Hz), drift(cents), rng(cents)}"""
    import librosa

    hop = 160  # 100 fps
    f0, _vfl, _vpr = librosa.pyin(
        audio, fmin=170, fmax=1700, sr=SR, frame_length=1024, hop_length=hop
    )
    t_f = np.arange(len(f0)) * hop / SR
    good = ~np.isnan(f0)
    if good.sum() < 4:
        return [{"micro_rms": 0, "rate": 0, "drift": 0, "rng": 0}] * len(bounds)
    f0i = np.interp(t_f, t_f[good], f0[good])

    out = []
    TRIM = 5  # drop 50ms at each end (5 frames @100fps), filtering f0 jumps in note transitions
    F_RATE = 100  # pyin frame rate
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

        # frequency-domain vibrato detection: Hann-windowed FFT -> 3.5-9Hz band energy
        win = np.hanning(len(cents))
        spec = np.abs(np.fft.rfft((cents - cents.mean()) * win))
        freqs = np.fft.rfftfreq(len(cents), d=1.0 / F_RATE)
        vib_band = (freqs >= 3.5) & (freqs <= 9.0)
        N = max(len(cents), 1)
        vib_energy = float(np.sqrt((spec[vib_band] ** 2).sum() / N**2))
        # dominant frequency (within the vib band only)
        if vib_band.any() and spec[vib_band].max() > 0:
            rate = float(freqs[vib_band][int(np.argmax(spec[vib_band]))])
        else:
            rate = 0.0
        # net drift: median of the first 1/4 -> median of the last 1/4
        q = max(2, len(cents) // 4)
        drift = float(np.median(cents[-q:]) - np.median(cents[:q]))
        total_rms = float(np.sqrt((cents**2).mean()))
        out.append({"vib_energy": vib_energy, "rate": rate, "drift": drift, "total_rms": total_rms})
    return out


def is_vibrato(a):
    return a["vib_energy"] > 3.0 and 3.5 <= a["rate"] <= 9.5


def is_slide(a):
    return abs(a["drift"]) > 60.0


# ---- Experiment 1: control fidelity (control-signal-level analysis) ---- #
def analyze_control(f0_np, score):
    """Analyse the f0 control signal (250Hz) directly before DDSP, unaffected by DDSP audio quality.
    Uses the middle 60% of each note, avoiding the crossfade overlap and the start/end transitions."""
    FPS_CTRL = 250
    CF = 5  # crossfade frames of control_from_behavior (20ms@250Hz); overlaps accumulate and must be aligned
    out = []
    t_frame = 0
    prev_T = None
    for _m, dur, art in score:
        T = max(4, int(dur * FPS_CTRL))
        if prev_T is not None:
            t_frame -= min(CF, prev_T // 2, T // 2)  # overlap in sync with control_from_behavior
        # vibrato/plain take the middle 60% (avoiding crossfade + attack/release);
        # a slide happens at the note head (slide-in over the first 40%), so the head only skips the crossfade (~24ms).
        margin = max(12, T // 5)
        i0 = t_frame + (6 if art == 2 else margin)
        i1 = t_frame + T - margin
        t_frame += T
        prev_T = T
        if i1 <= i0 or i1 > len(f0_np) or dur < 0.3:
            out.append(
                {"vib_energy": np.nan, "total_rms": np.nan, "drift": 0.0, "dur": dur, "art": art}
            )
            continue
        seg = f0_np[i0:i1]
        cents = 1200 * np.log2(seg / (np.median(seg) + 1e-9) + 1e-12)
        win = np.hanning(len(cents))
        spec = np.abs(np.fft.rfft((cents - cents.mean()) * win))
        freqs = np.fft.rfftfreq(len(cents), d=1.0 / FPS_CTRL)
        vib_band = (freqs >= 3.5) & (freqs <= 9.0)
        Nc = max(len(cents), 1)
        vib_e = float(np.sqrt((spec[vib_band] ** 2).sum() / Nc**2))
        total_rms = float(np.sqrt((cents**2).mean()))
        q = max(2, len(cents) // 4)
        # start from the first 4 frames of the window (not the median of the first 1/4): a fast slide finishes within 80ms, and a long-window median would swallow it
        qh = max(2, min(4, len(cents) // 8))
        drift = float(np.median(cents[-q:]) - np.median(cents[:qh]))
        out.append(
            {"vib_energy": vib_e, "total_rms": total_rms, "drift": drift, "dur": dur, "art": art}
        )
    return out


class RandSameLabel:
    """Ablation baseline: label filtering ✓ context retrieval ✗: random trajectory from the same-label pool."""

    def __init__(self, nn, seed=42):
        self.nn = nn
        self.rng = np.random.default_rng(seed)

    def predict(self, feat):
        feat = np.asarray(feat, dtype=np.float32)
        a = int(round(float(feat[6]))) if len(feat) >= 7 else 0
        pool = self.nn._pools.get(a)
        if pool is None or len(pool) == 0:
            pool = np.arange(len(self.nn.base))
        return self.nn.trajectories[int(self.rng.choice(pool))]


class NNNoLabel:
    """Ablation baseline: label filtering ✗ context retrieval ✓: whole-library 6-D nearest neighbour, labels ignored."""

    def __init__(self, nn):
        self.nn = nn

    def predict(self, feat):
        q = np.asarray(feat, dtype=np.float32)[:6] / self.nn._std
        dist = np.linalg.norm(self.nn._norm - q, axis=1)
        return self.nn.trajectories[int(np.argmin(dist))]


def exp1_controllability(R):
    print("\n=== Experiment 1: articulation control fidelity (control-signal level, 4 melodies) ===")
    from neural.perform import control_from_behavior

    results = {}
    ablations = {"nn_nolabel": NNNoLabel(R.nns["violin"])}
    # the v1 (VAE)/v2 (parametric) baseline models were removed in a code cleanup; their historical numbers are kept in
    # checkpoints/eval/eval_results.json (the unpaired comparison in Sec.2 of the paper cites that archive)
    for source in ["v0", "nn_nolabel"]:
        vib_es, plain_es, slide_drifts = [], [], []
        slide_ok, slide_n = 0, 0
        for mname, score in MELODIES.items():
            notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
            beh_model = ablations.get(source, R.nns["violin"])
            f0, loud = control_from_behavior(
                notes, beh_model, pitch_glide=False, envelope="sustain", traj_gain=1.0
            )
            f0_np = f0.numpy()[0, :, 0]
            # also render and save audio (for ear tests; ablation baselines only do control-level analysis)
            if source == "v0":
                audio, bounds = R.render(score, source=source)
                sf.write(str(OUT / f"{mname}_{source}.wav"), audio, SR)
            ana = analyze_control(f0_np, score)
            for a in ana:
                if np.isnan(a["vib_energy"]):
                    continue
                if a["art"] == 1:
                    vib_es.append(a["vib_energy"])
                elif a["art"] == 2:
                    slide_drifts.append(abs(a["drift"]))
                    slide_n += 1
                    slide_ok += abs(a["drift"]) > 50
                else:
                    plain_es.append(a["vib_energy"])

        vib_m = float(np.median(vib_es)) if vib_es else 0
        plain_m = float(np.median(plain_es)) if plain_es else 0
        contrast = vib_m / max(plain_m, 0.1)
        r = {
            "vib_median": vib_m,
            "plain_median": plain_m,
            "contrast": contrast,
            "slide_ok": slide_ok,
            "slide_n": slide_n,
            "n_vib": len(vib_es),
            "n_plain": len(plain_es),
        }
        results[source] = r
        print(
            f"  {source:>10s}  vib median {vib_m:.0f}c  plain median {plain_m:.0f}c  "
            f"contrast {contrast:.1f}×   "
            f"slide hits {slide_ok}/{slide_n}"
        )

    # rand_label ablation: 100 random seeds, report the distribution instead of a single point (single-seed conclusions are unreliable)
    from neural.perform import control_from_behavior

    rand_contrasts = []
    for seed in range(100):
        rl = RandSameLabel(R.nns["violin"], seed=seed)
        vib_es, plain_es = [], []
        for _mname, score in MELODIES.items():
            notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
            f0, _ = control_from_behavior(
                notes, rl, pitch_glide=False, envelope="sustain", traj_gain=1.0
            )
            for a in analyze_control(f0.numpy()[0, :, 0], score):
                if np.isnan(a["vib_energy"]):
                    continue
                if a["art"] == 1:
                    vib_es.append(a["vib_energy"])
                elif a["art"] == 0:
                    plain_es.append(a["vib_energy"])
        vm = float(np.median(vib_es)) if vib_es else 0
        pm = float(np.median(plain_es)) if plain_es else 0
        rand_contrasts.append(vm / max(pm, 0.1))
    rc = np.array(rand_contrasts)
    results["rand_label_multiseed"] = {
        "n_seeds": len(rc),
        "median": round(float(np.median(rc)), 2),
        "p2.5": round(float(np.percentile(rc, 2.5)), 2),
        "p97.5": round(float(np.percentile(rc, 97.5)), 2),
        "frac_gt1": round(float((rc > 1).mean()), 2),
    }
    print(
        f"  rand_label (100 seeds)  median {np.median(rc):.2f}×  "
        f"95% range [{np.percentile(rc, 2.5):.2f}, {np.percentile(rc, 97.5):.2f}]  "
        f"{int((rc > 1).sum())}/100 seeds >1"
    )

    # flat baseline
    for mname, score in MELODIES.items():
        audio, bounds = R.render(score, source="flat")
        sf.write(str(OUT / f"{mname}_flat.wav"), audio, SR)
    print("  flat  (no behaviour trajectories, all metrics=0, audio saved)")
    return results


# ---- Experiment 2: decoupling (2×2 transfer matrix) ---- #
def behavior_sig(ana):
    """Behaviour signature = per note [vib_energy, drift, total_rms]"""
    return np.array([[a["vib_energy"], a["drift"], a["total_rms"]] for a in ana]).ravel()


def timbre_sig(audio):
    import librosa

    mf = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)[1:]  # drop c0 (energy)
    return mf.mean(axis=1)


def exp2_decoupling(R):
    insts = list(R.models.keys())  # violin / guitar / cello (only those with a checkpoint)
    n = len(insts)
    print(f"\n=== Experiment 2: behaviour × timbre decoupling ({n}×{n}, lyrical+folk pooled) ===")
    bsig, tsig = {}, {}
    for mname in ["lyrical", "folk"]:
        score = MELODIES[mname]
        for beh in insts:
            for tim in insts:
                audio, bounds = R.render(score, timbre=tim, behavior=beh)
                sf.write(str(OUT / f"x_{mname}_{beh[:2]}B_{tim[:2]}T.wav"), audio, SR)
                key = (beh, tim)
                bsig[key] = bsig.get(key, []) + [behavior_sig(analyze(audio, bounds))]
                tsig[key] = tsig.get(key, []) + [timbre_sig(audio)]

    def dist(sig, k1, k2):
        return float(np.mean([np.linalg.norm(a - b) for a, b in zip(sig[k1], sig[k2])]))

    # for each cross-instrument combination (behB, timT) run a double test:
    #   behaviour distance (to the same-behaviour diagonal bB×bT) < behaviour distance (to the same-timbre diagonal tB×tT) -> behaviour follows the behaviour source
    #   timbre distance (to the same-timbre diagonal tB×tT) < timbre distance (to the same-behaviour diagonal bB×bT) -> timbre follows the timbre source
    results, ok = {}, 0
    total = 0
    for beh in insts:
        for tim in insts:
            if beh == tim:
                continue
            x = (beh, tim)
            b_beh = dist(bsig, x, (beh, beh))
            b_tim = dist(bsig, x, (tim, tim))
            t_beh = dist(tsig, x, (beh, beh))
            t_tim = dist(tsig, x, (tim, tim))
            c1, c2 = b_beh < b_tim, t_tim < t_beh
            ok += c1 + c2
            total += 2
            results[f"{beh[:2]}B_{tim[:2]}T"] = {
                "b_beh": b_beh,
                "b_tim": b_tim,
                "t_beh": t_beh,
                "t_tim": t_tim,
                "behavior_follows": bool(c1),
                "timbre_follows": bool(c2),
            }
            print(
                f"  {beh:>6s} behaviour × {tim:>6s} timbre:  "
                f"behaviour follows behaviour source {'✓' if c1 else '✗'} ({b_beh:.0f} vs {b_tim:.0f})   "
                f"timbre follows timbre source {'✓' if c2 else '✗'} ({t_tim:.0f} vs {t_beh:.0f})"
            )
    print(f"  passed {ok}/{total} tests")
    results["passed"] = f"{ok}/{total}"
    return results


# ---- Experiment 3: behaviour library size curve ---- #
def exp3_datacurve(R):
    """Control-signal-level analysis (same criterion as experiment 1): unaffected by DDSP audio quality/pyin re-extraction noise."""
    from neural.neural_take import ConditionedNN
    from neural.perform import control_from_behavior

    print("\n=== Experiment 3: behaviour library size -> control fidelity (control-signal level) ===")
    sizes = [50, 150, 500, 1500, 7582]
    n_seeds = 8  # subsampling is noisy (retrieval hits are random with a small library), average over several seeds
    curve = []
    for n in sizes:
        c_list, s_list, v_list, p_list = [], [], [], []
        for seed in range(n_seeds):
            order = np.random.default_rng(42 + seed).permutation(len(R.vn_feats))
            idx = order[: min(n, len(order))]
            nn = ConditionedNN(R.vn_feats[idx], R.vn_trajs[idx])
            vib_es, plain_es, slide_ok, slide_n = [], [], 0, 0
            for _mname, score in MELODIES.items():
                notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
                f0, _loud = control_from_behavior(
                    notes, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0
                )
                for a in analyze_control(f0.numpy()[0, :, 0], score):
                    if np.isnan(a["vib_energy"]):
                        continue
                    if a["art"] == 1:
                        vib_es.append(a["vib_energy"])
                    elif a["art"] == 2:
                        slide_n += 1
                        slide_ok += abs(a["drift"]) > 50
                    else:
                        plain_es.append(a["vib_energy"])
            vib_m = float(np.median(vib_es)) if vib_es else 0
            plain_m = float(np.median(plain_es)) if plain_es else 0
            c_list.append(vib_m / max(plain_m, 0.1))
            s_list.append(slide_ok / max(slide_n, 1))
            v_list.append(vib_m)
            p_list.append(plain_m)
        row = {
            "n": int(min(n, len(R.vn_feats))),
            "vib_median": float(np.mean(v_list)),
            "plain_median": float(np.mean(p_list)),
            "contrast": float(np.mean(c_list)),
            "contrast_std": float(np.std(c_list) / np.sqrt(len(c_list))),
            "slide_hit": float(np.mean(s_list)),
        }
        curve.append(row)
        print(
            f"  {row['n']:>5d} notes  contrast {row['contrast']:.1f}×±{row['contrast_std']:.1f}  "
            f"slide hits {row['slide_hit']:.0%}"
        )
    return curve


# ---- Experiment 4: audio-level articulation validation (pYIN re-extraction) ---- #
def exp4_audio_controllability(R):
    """Run pYIN on the rendered WAVs and measure vibrato band energy per note,
    verifying that the control-signal-level contrast still holds at the audio end."""
    print("\n=== Experiment 4: audio-level articulation validation (pYIN re-extraction) ===")
    results = {}
    for source in ["v0", "flat"]:
        vib_es, plain_es = [], []
        for _mname, score in MELODIES.items():
            audio, bounds = R.render(score, source=source)
            ana = analyze(audio, bounds)
            for a, (_m, _d, art) in zip(ana, score):
                if a["vib_energy"] == 0.0 and a["total_rms"] == 0.0:
                    continue
                if art == 1:
                    vib_es.append(a["vib_energy"])
                elif art == 0:
                    plain_es.append(a["vib_energy"])

        vib_m = float(np.median(vib_es)) if vib_es else 0
        plain_m = float(np.median(plain_es)) if plain_es else 0
        contrast = vib_m / max(plain_m, 0.1)
        results[source] = {
            "vib_median": round(vib_m, 1),
            "plain_median": round(plain_m, 1),
            "contrast": round(contrast, 1),
            "n_vib": len(vib_es),
            "n_plain": len(plain_es),
        }
        print(
            f"  {source:>4s}  vib median {vib_m:.1f}c  plain median {plain_m:.1f}c  "
            f"contrast {contrast:.1f}×  (n_vib={len(vib_es)}, n_plain={len(plain_es)})"
        )
    return results


# ---- Experiment 5: paired counterfactual (same note, swapped label) ---- #
def _bootstrap_median_ci(arr, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(42)
    medians = np.array(
        [float(np.median(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    )
    lo = float(np.percentile(medians, 100 * alpha / 2))
    hi = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    return lo, hi


def exp5_paired_counterfactual(R):
    """For every note of every melody, render once with a vibrato and once with a plain label,
    and compare vib_energy pairwise -> removes duration/pitch/position confounds.
    Uses log-ratio + epsilon to keep all predefined eligible notes (no outcome-dependent exclusion);
    on the audio side the torch seed is reset before each melody pair to guarantee the same DDSP noise realisation."""
    from neural.perform import control_from_behavior

    print("\n=== Experiment 5: paired counterfactual (same note vibrato vs plain, control-signal level + audio level) ===")

    EPS = 1e-4
    ctrl_logr, audio_logr = [], []
    ctrl_gt1, audio_gt1 = 0, 0
    ctrl_by_mel, audio_by_mel = {}, {}
    for mi, (mname, score) in enumerate(MELODIES.items()):
        score_vib = [(m, d, 1) for (m, d, _a) in score]
        score_pln = [(m, d, 0) for (m, d, _a) in score]
        notes_vib = [{"midi": m, "dur": d, "art": a} for m, d, a in score_vib]
        notes_pln = [{"midi": m, "dur": d, "art": a} for m, d, a in score_pln]
        nn = R.nns["violin"]

        f0_v, _ = control_from_behavior(
            notes_vib, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0
        )
        f0_p, _ = control_from_behavior(
            notes_pln, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0
        )
        ana_v = analyze_control(f0_v.numpy()[0, :, 0], score_vib)
        ana_p = analyze_control(f0_p.numpy()[0, :, 0], score_pln)
        _mel = []
        for av, ap in zip(ana_v, ana_p):
            if np.isnan(av["vib_energy"]) or np.isnan(ap["vib_energy"]):
                continue
            lr = np.log2((av["vib_energy"] + EPS) / (ap["vib_energy"] + EPS))
            ctrl_logr.append(lr)
            _mel.append(lr)
            if lr > 0:
                ctrl_gt1 += 1
        ctrl_by_mel[mname] = _mel

        pair_seed = 1000 + mi
        torch.manual_seed(pair_seed)
        audio_v, bounds_v = R.render(score_vib, source="v0")
        torch.manual_seed(pair_seed)
        audio_p, bounds_p = R.render(score_pln, source="v0")
        aana_v = analyze(audio_v, bounds_v)
        aana_p = analyze(audio_p, bounds_p)
        _mel = []
        for av, ap, (_m, d_, _a) in zip(aana_v, aana_p, score):
            if d_ < 0.3:
                continue
            lr = np.log2((av["vib_energy"] + EPS) / (ap["vib_energy"] + EPS))
            audio_logr.append(lr)
            _mel.append(lr)
            if lr > 0:
                audio_gt1 += 1
        audio_by_mel[mname] = _mel

    ctrl_arr = np.array(ctrl_logr)
    audio_arr = np.array(audio_logr)
    ctrl_med = float(2 ** np.median(ctrl_arr))
    audio_med = float(2 ** np.median(audio_arr))
    ctrl_ci = tuple(float(2**x) for x in _bootstrap_median_ci(ctrl_arr))
    audio_ci = tuple(float(2**x) for x in _bootstrap_median_ci(audio_arr))

    print(
        f"  control signal: paired median ratio {ctrl_med:.1f}× bootstrap 95% CI [{ctrl_ci[0]:.1f}, {ctrl_ci[1]:.1f}]  "
        f"{ctrl_gt1}/{len(ctrl_logr)} ratio>1"
    )
    print(
        f"  audio:   paired median ratio {audio_med:.1f}× bootstrap 95% CI [{audio_ci[0]:.1f}, {audio_ci[1]:.1f}]  "
        f"{audio_gt1}/{len(audio_logr)} ratio>1"
    )

    results = {
        "control_median_ratio": round(ctrl_med, 2),
        "control_bootstrap_ci95": [round(ctrl_ci[0], 2), round(ctrl_ci[1], 2)],
        "control_n": len(ctrl_logr),
        "control_gt1": ctrl_gt1,
        "audio_median_ratio": round(audio_med, 2),
        "audio_bootstrap_ci95": [round(audio_ci[0], 2), round(audio_ci[1], 2)],
        "audio_n": len(audio_logr),
        "audio_gt1": audio_gt1,
        # per-melody median ratio: notes share melodic context (the item CI has pseudo-replication),
        # and there are only 4 clusters, so per-melody values are reported directly for readers to judge robustness
        "control_per_melody": {
            m: round(float(2 ** np.median(v)), 2) for m, v in ctrl_by_mel.items() if v
        },
        "audio_per_melody": {
            m: round(float(2 ** np.median(v)), 2) for m, v in audio_by_mel.items() if v
        },
    }
    print("  per melody (control):", results["control_per_melody"])
    print("  per melody (audio):", results["audio_per_melody"])
    return results


# ---- Experiment 6: post-processing attribution ablation (paired counterfactual × three post-processing modes) ---- #
def exp6_processing_ablation(R):
    """Answers "does the contrast come from retrieval or from hand-written filtering": the same paired counterfactual protocol,
    comparing label_processing = none (retrieval as is) / uniform (uniform processing) / full (complete)."""
    from neural.perform import control_from_behavior

    print("\n=== Experiment 6: post-processing attribution (paired counterfactual, control-signal level) ===")
    results = {}
    for mode in ["none", "uniform", "full"]:
        logr, gt1 = [], 0
        by_melody = {}
        for mname, score in MELODIES.items():
            score_vib = [(m, d, 1) for (m, d, _a) in score]
            score_pln = [(m, d, 0) for (m, d, _a) in score]
            notes_vib = [{"midi": m, "dur": d, "art": a} for m, d, a in score_vib]
            notes_pln = [{"midi": m, "dur": d, "art": a} for m, d, a in score_pln]
            nn = R.nns["violin"]
            f0_v, _ = control_from_behavior(
                notes_vib,
                nn,
                pitch_glide=False,
                envelope="sustain",
                traj_gain=1.0,
                label_processing=mode,
            )
            f0_p, _ = control_from_behavior(
                notes_pln,
                nn,
                pitch_glide=False,
                envelope="sustain",
                traj_gain=1.0,
                label_processing=mode,
            )
            ana_v = analyze_control(f0_v.numpy()[0, :, 0], score_vib)
            ana_p = analyze_control(f0_p.numpy()[0, :, 0], score_pln)
            EPS = 1e-4
            mel_lrs = []
            for av, ap in zip(ana_v, ana_p):
                if np.isnan(av["vib_energy"]) or np.isnan(ap["vib_energy"]):
                    continue
                lr = np.log2((av["vib_energy"] + EPS) / (ap["vib_energy"] + EPS))
                logr.append(lr)
                mel_lrs.append(lr)
                if lr > 0:
                    gt1 += 1
            by_melody[mname] = mel_lrs
        arr = np.array(logr)
        med = float(2 ** np.median(arr))
        ci = tuple(float(2**x) for x in _bootstrap_median_ci(arr))
        # melody level: paired median ratio per melody (notes share melodic context, the item CI has pseudo-replication,
        # 4 clusters are not enough for a stable cluster bootstrap -> report per-melody values directly)
        per_mel = {m: round(float(2 ** np.median(v)), 2) for m, v in by_melody.items() if v}
        results[mode] = {
            "paired_median_ratio": round(med, 2),
            "item_ci95": [round(ci[0], 2), round(ci[1], 2)],
            "n": len(logr),
            "gt1": gt1,
            "per_melody_median": per_mel,
        }
        pm_str = " ".join(f"{m}:{v:.1f}×" for m, v in per_mel.items())
        print(
            f"  {mode:>8s}: paired median ratio {med:.1f}× item-CI [{ci[0]:.1f}, {ci[1]:.1f}]  "
            f"{gt1}/{len(logr)} ratio>1  | per melody {pm_str}"
        )
    return results


# ---- Experiment 7: glass×violin 2×2 formal attribution ---- #
def exp7_glass_2x2(R):
    """glass/violin two-asset 2×2: the behaviour signature follows the behaviour source, the timbre signature follows the timbre source.
    All cells are shifted up an octave (glass's natural range), and the envelope is uniformly sustain for comparability."""
    from neural.ddsp import DDSP
    from neural.neural_take import ConditionedNN
    from neural.perform import control_from_behavior

    ck = CKPT / "ddsp_glass.pt"
    st = CKPT / "glass_style.npz"
    if not ck.exists() or not st.exists():
        print("\n=== Experiment 7: skipped (glass checkpoint missing) ===")
        return {"skipped": True}
    print("\n=== Experiment 7: glass×violin 2×2 attribution ===")
    glass = DDSP(dropout=0.0)
    glass.load_state_dict(torch.load(str(ck), map_location="cpu", weights_only=True))
    glass.eval()
    d_g = np.load(str(st))
    nn_g = ConditionedNN(d_g["features"], d_g["trajectories"])
    models = {"glass": glass, "violin": R.models["violin"]}
    nns = {"glass": nn_g, "violin": R.nns["violin"]}

    TR = 12
    bsig, tsig = {}, {}
    for mname in ["lyrical", "folk"]:
        # slide notes become plain: the glass library has no slide pool, ConditionedNN would silently fall back to the whole library,
        # and slides are mostly rebuilt by hand-written rules -> keeping them would contaminate the attribution of the "behaviour swap"
        score = [(m + TR, d, 0 if a == 2 else a) for (m, d, a) in MELODIES[mname]]
        for beh in ["glass", "violin"]:
            for tim in ["glass", "violin"]:
                notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
                torch.manual_seed(42)
                f0, loud = control_from_behavior(
                    notes, nns[beh], pitch_glide=False, envelope="sustain", traj_gain=1.0
                )
                with torch.no_grad():
                    y = models[tim](f0, loud, noise_gain=0.6, noise_gate_db=-1.0).numpy()[0]
                y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
                CF_FRAMES = 5
                bounds, t_sec, prev_n = [], 0.0, None
                for _m, d_, _a in score:
                    n_ = max(4, int(d_ * FPS))
                    if prev_n is not None:
                        t_sec -= min(CF_FRAMES, prev_n // 2, n_ // 2) / FPS
                    bounds.append((t_sec, t_sec + d_))
                    t_sec += d_
                    prev_n = n_
                sf.write(str(OUT / f"g22_{mname}_{beh[:2]}B_{tim[:2]}T.wav"), y, SR)
                key = (beh, tim)
                bsig[key] = bsig.get(key, []) + [behavior_sig(analyze(y, bounds))]
                tsig[key] = tsig.get(key, []) + [timbre_sig(y)]

    def dist(sig, k1, k2):
        return float(np.mean([np.linalg.norm(a - b) for a, b in zip(sig[k1], sig[k2])]))

    results, ok, total = {}, 0, 0
    for beh, tim in [("glass", "violin"), ("violin", "glass")]:
        x = (beh, tim)
        b_beh = dist(bsig, x, (beh, beh))
        b_tim = dist(bsig, x, (tim, tim))
        t_beh = dist(tsig, x, (beh, beh))
        t_tim = dist(tsig, x, (tim, tim))
        c1, c2 = b_beh < b_tim, t_tim < t_beh
        ok += c1 + c2
        total += 2
        results[f"{beh[:2]}B_{tim[:2]}T"] = {
            "b_beh": b_beh,
            "b_tim": b_tim,
            "t_beh": t_beh,
            "t_tim": t_tim,
            "behavior_follows": bool(c1),
            "timbre_follows": bool(c2),
        }
        print(
            f"  {beh:>6s} behaviour × {tim:>6s} timbre:  "
            f"behaviour follows {'✓' if c1 else '✗'} ({b_beh:.0f} vs {b_tim:.0f})   "
            f"timbre follows {'✓' if c2 else '✗'} ({t_tim:.0f} vs {t_beh:.0f})"
        )
    print(f"  passed {ok}/{total}")
    results["passed"] = f"{ok}/{total}"
    return results


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    R = Renderer()
    res = {
        "controllability": exp1_controllability(R),
        "decoupling": exp2_decoupling(R),
        "data_curve": exp3_datacurve(R),
        "audio_controllability": exp4_audio_controllability(R),
        "paired_counterfactual": exp5_paired_counterfactual(R),
        "processing_ablation": exp6_processing_ablation(R),
        "glass_2x2": exp7_glass_2x2(R),
    }
    with open(str(OUT / "eval_results.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUT / 'eval_results.json'}; rendered audio in {OUT}/")


if __name__ == "__main__":
    main()
