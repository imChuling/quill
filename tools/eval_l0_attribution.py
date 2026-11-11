"""E3: decoder-independent evidence: source attribution for the L0 capture timbre.

Uses glass_studio (L0 capture DDSP) as a 4th timbre and runs the same protocol as exp2/exp7,
a 2×2 behaviour × timbre attribution matrix (two melodies × pairs). Verifies that behaviour/timbre decoupling still holds on a DDSP decoder
outside the training set.

Run: cd ~/Desktop/MT/project/quill && python tools/eval_l0_attribution.py
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

MELODIES = {
    "lyrical": [
        (62, 0.8, 1), (65, 0.4, 0), (67, 1.2, 1), (65, 0.4, 0),
        (69, 0.4, 2), (72, 1.6, 1), (67, 0.8, 0), (62, 1.6, 1),
    ],
    "folk": [
        (69, 0.4, 0), (71, 0.2, 0), (72, 0.4, 0), (69, 0.4, 0),
        (67, 0.8, 1), (64, 0.4, 0), (67, 0.4, 2), (69, 1.2, 1),
        (67, 0.4, 0), (64, 0.4, 0), (62, 1.6, 1),
    ],
}


def analyze(audio, bounds):
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
    return np.array([[a["vib_energy"], a["drift"], a["total_rms"]] for a in ana]).ravel()


def timbre_sig(audio):
    import librosa
    mf = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)[1:]
    return mf.mean(axis=1)


def render(model, nn, score, envelope="sustain"):
    from neural.perform import control_from_behavior
    notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
    f0, loud = control_from_behavior(
        notes, nn, pitch_glide=False, envelope=envelope, traj_gain=1.0,
    )
    with torch.no_grad():
        y = model(f0, loud, noise_gain=0.6, noise_gate_db=-1.0).numpy()[0]
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
    return y, bounds


def main():
    from neural.ddsp import DDSP
    from neural.neural_take import ConditionedNN

    np.random.seed(42)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    l0_ck = QUILL / "assets" / "user_ddsp" / "glass_studio.pt"
    vn_ck = CKPT / "ddsp_urmp_vn_12k.pt"
    vn_lib = CKPT / "urmp_vn_trajectories.npz"
    gl_lib = CKPT / "glass_style.npz"

    for p in [l0_ck, vn_ck, vn_lib, gl_lib]:
        if not p.exists():
            print(f"SKIP: {p} not found")
            return

    print("Loading models...")
    vn_ddsp = DDSP(dropout=0.0)
    vn_ddsp.load_state_dict(torch.load(str(vn_ck), map_location="cpu", weights_only=True))
    vn_ddsp.eval()

    l0_ddsp = DDSP(dropout=0.0)
    l0_ddsp.load_state_dict(torch.load(str(l0_ck), map_location="cpu", weights_only=True))
    l0_ddsp.eval()

    d_vn = np.load(str(vn_lib), allow_pickle=False)
    nn_vn = ConditionedNN(d_vn["features"], d_vn["trajectories"])

    d_gl = np.load(str(gl_lib), allow_pickle=False)
    nn_gl = ConditionedNN(d_gl["features"], d_gl["trajectories"])

    models = {"violin": vn_ddsp, "glass_l0": l0_ddsp}
    nns = {"violin": nn_vn, "glass": nn_gl}

    TR = 12
    bsig, tsig = {}, {}

    print("\n=== E3: L0 capture decoder-independent evidence (glass_l0×violin 2×2) ===")
    for mname in ["lyrical", "folk"]:
        score = [(m + TR, d, 0 if a == 2 else a) for (m, d, a) in MELODIES[mname]]
        for beh in ["glass", "violin"]:
            for tim in ["glass_l0", "violin"]:
                torch.manual_seed(42)
                y, bounds = render(models[tim], nns[beh], score)
                tag = f"l0_{mname}_{beh[:2]}B_{tim.replace('glass_l0','gl0')}T"
                sf.write(str(OUT / f"{tag}.wav"), y, SR)
                key = (beh, tim)
                bsig[key] = bsig.get(key, []) + [behavior_sig(analyze(y, bounds))]
                tsig[key] = tsig.get(key, []) + [timbre_sig(y)]

    def dist(sig, k1, k2):
        return float(np.mean([np.linalg.norm(a - b) for a, b in zip(sig[k1], sig[k2])]))

    results, ok, total = {}, 0, 0
    for beh, tim in [("glass", "violin"), ("violin", "glass_l0")]:
        x = (beh, tim)
        diag_beh = (beh, {"glass": "glass_l0", "violin": "violin"}[beh])
        diag_tim = ({"glass_l0": "glass", "violin": "violin"}[tim], tim)
        b_beh = dist(bsig, x, diag_beh)
        b_tim = dist(bsig, x, diag_tim)
        t_beh = dist(tsig, x, diag_beh)
        t_tim = dist(tsig, x, diag_tim)
        c1, c2 = b_beh < b_tim, t_tim < t_beh
        ok += c1 + c2
        total += 2
        label = f"{beh}B_{tim}T"
        results[label] = {
            "b_beh": round(b_beh, 2),
            "b_tim": round(b_tim, 2),
            "t_beh": round(t_beh, 2),
            "t_tim": round(t_tim, 2),
            "behavior_follows": bool(c1),
            "timbre_follows": bool(c2),
        }
        print(
            f"  {beh:>6s} behavior × {tim:>8s} timbre:  "
            f"beh follows {'Y' if c1 else 'N'} ({b_beh:.0f} vs {b_tim:.0f})   "
            f"tim follows {'Y' if c2 else 'N'} ({t_tim:.0f} vs {t_beh:.0f})"
        )
    results["passed"] = f"{ok}/{total}"
    print(f"  Passed {ok}/{total}")

    out_path = OUT / "l0_attribution.json"
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
