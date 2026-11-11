"""Structural difference tests: an honest check responding to "no different from a sampler" (2026-08-01).

Only tests what a sampler **structurally cannot do** (the information is not in the sample, a script cannot add it), not what it "has not done":

T1 · single-source chord: play a 6-note chord from one recording.
     sampler = the same sample transposed 6 times and summed -> the voices' noise layers are **fully correlated** -> comb filtering/phase cancellation;
     Quill  = 6 voices each with dispersed initial phases + independent noise read pointers -> sounds like an ensemble.
     Metrics: noise cross-correlation between voices; comb-notch depth of the chord spectrum.

T2 · long sustain from a short recording: a 2-second source has to play a 15-second note.
     sampler = must loop (crossfade) -> audible periodicity; Quill = resynthesis, no loop point.
     Metric: periodic autocorrelation peaks of the envelope/spectrum (loop traces).

T3 · behaviour transfer: samplers have no such concept (see stimuli_2x2 for the comparison).

Outputs: objective metrics + audio pairs (assets/ear_tests/structural/).
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

from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from synth.render import render  # noqa: E402

SR = 44100
CHORD = [55, 59, 62, 67, 71, 74]  # 6 notes, including octaves and fifths (most likely to expose phase problems)


def rms_norm(y, r=0.08):
    v = np.sqrt((np.asarray(y, np.float64) ** 2).mean())
    return (np.asarray(y, np.float64) * (r / max(v, 1e-9))).astype(np.float32)


def sampler_chord(snap, dur=3.0):
    """Sampler chord: the same sample transposed N times and summed (this is physically what a sampler does)."""
    import librosa

    f0_ref = float(snap["f0_ref"])
    base = render(snap, f0_ref, length=int(dur * SR), mode="formant", add_attack=True)
    ref_midi = 69 + 12 * np.log2(f0_ref / 440.0)
    out = np.zeros(int(dur * SR), np.float64)
    voices = []
    for m in CHORD:
        v = librosa.effects.pitch_shift(base.astype(np.float32), sr=SR, n_steps=float(m - ref_midi))
        v = v[: len(out)]
        voices.append(v)
        out[: len(v)] += v
    return out, voices


def quill_chord(snap, dur=3.0):
    """Quill chord: 6 engine voices (dispersed initial phases + their own noise read pointers)."""
    inst = build_instrument(snap, noise_level=0.02)
    eng = Engine(inst, eval_mode=False)  # eval_mode=False -> randomised noise start points
    eng.reset()
    evs = []
    for m in CHORD:
        evs.append((0.02, "on", m, 100, 0.0, -1))
        evs.append((dur - 0.3, "off", m, 0, 0.0, -1))
    return eng.render_offline(evs, dur).astype(np.float64)


def comb_depth(y):
    """Comb-notch depth: valley/peak excursion (dB) of the detrended spectral envelope. Heavier phase cancellation gives larger values."""
    S = np.abs(np.fft.rfft(y[: 1 << 16] * np.hanning(min(len(y), 1 << 16))))
    f = np.fft.rfftfreq(min(len(y), 1 << 16), 1 / SR)
    m = (f > 200) & (f < 6000)
    lg = 20 * np.log10(np.maximum(S[m], 1e-9))
    k = 101  # trend smoothing
    trend = np.convolve(lg, np.ones(k) / k, mode="same")
    d = lg[k:-k] - trend[k:-k]
    return float(np.percentile(d, 95) - np.percentile(d, 5))


def loop_periodicity(y, min_lag_s=0.5):
    """Loop trace: the maximum autocorrelation peak of the envelope at lags >0.5s (0 = no periodicity, larger = more like looping)."""
    hop = 441
    n = len(y) // hop * hop
    e = np.sqrt((y[:n].reshape(-1, hop) ** 2).mean(1) + 1e-12)
    e = e[int(0.5 * SR / hop) :]  # drop the attack
    e = e - e.mean()
    if len(e) < 40:
        return 0.0
    ac = np.correlate(e, e, mode="full")[len(e) - 1 :]
    ac /= max(ac[0], 1e-12)
    lag0 = int(min_lag_s * SR / hop)
    return float(ac[lag0:].max()) if len(ac) > lag0 + 2 else 0.0


def sampler_sustain(snap, src_len=2.0, out_len=15.0, xfade=0.25):
    """Sampler long sustain: loop (crossfade) a 2-second source to 15 seconds: the only thing a sampler can do."""
    f0_ref = float(snap["f0_ref"])
    src = render(snap, f0_ref, length=int(src_len * SR), mode="formant", add_attack=True).astype(
        np.float64
    )
    xf = int(xfade * SR)
    loop = src[int(0.3 * SR) :]  # loop region (skipping the attack)
    out = list(src[: int(0.3 * SR)])
    body = loop.copy()
    while len(out) < int(out_len * SR):
        if len(out) > xf:  # equal-power crossfade (standard sampler loop)
            tail = np.array(out[-xf:])
            head = body[:xf]
            w = np.linspace(0, 1, xf)
            out[-xf:] = list(tail * np.cos(w * np.pi / 2) + head * np.sin(w * np.pi / 2))
            out += list(body[xf:])
        else:
            out += list(body)
    return np.array(out[: int(out_len * SR)])


def quill_sustain(snap, out_len=15.0):
    """Quill long sustain: resynthesise to any length, no loop point."""
    inst = build_instrument(snap, noise_level=0.02)
    eng = Engine(inst, eval_mode=False)
    eng.reset()
    m = int(round(69 + 12 * np.log2(float(snap["f0_ref"]) / 440.0)))
    evs = [(0.02, "on", m, 100, 0.0, -1), (out_len - 0.5, "off", m, 0, 0.0, -1)]
    return eng.render_offline(evs, out_len).astype(np.float64)


def main():
    import soundfile as sf

    out = ROOT / "assets/ear_tests/structural"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, path in [
        ("glass", "assets/snapshots/wineglass.npz"),
        ("guitar", "assets/snapshots/guitar_rich.npz"),
    ]:
        snap = load_snapshot(str(ROOT / path))

        # T1 chord
        s_ch, s_voices = sampler_chord(snap)
        q_ch = quill_chord(snap)
        # noise correlation between voices (sampler: same sample -> highly correlated)
        cors = []
        for i in range(len(s_voices)):
            for j in range(i + 1, len(s_voices)):
                a, b = s_voices[i][:SR], s_voices[j][:SR]
                cors.append(abs(np.corrcoef(a, b)[0, 1]))
        sf.write(str(out / f"{name}_chord_sampler.wav"), rms_norm(s_ch), SR)
        sf.write(str(out / f"{name}_chord_quill.wav"), rms_norm(q_ch), SR)

        # T2 long sustain
        s_su = sampler_sustain(snap)
        q_su = quill_sustain(snap)
        sf.write(str(out / f"{name}_sustain15s_sampler.wav"), rms_norm(s_su), SR)
        sf.write(str(out / f"{name}_sustain15s_quill.wav"), rms_norm(q_su), SR)

        res[name] = {
            "chord_comb_depth_dB": {
                "sampler": round(comb_depth(s_ch), 1),
                "quill": round(comb_depth(q_ch), 1),
            },
            "chord_voice_corr_mean_sampler": round(float(np.mean(cors)), 3),
            "sustain_loop_periodicity": {
                "sampler": round(loop_periodicity(s_su), 3),
                "quill": round(loop_periodicity(q_su), 3),
            },
        }
        print(f"{name}:")
        print(
            f"  T1 chord comb depth (dB, smaller is cleaner): sampler={res[name]['chord_comb_depth_dB']['sampler']}"
            f"  quill={res[name]['chord_comb_depth_dB']['quill']}"
        )
        print(
            f"     sampler voice correlation={res[name]['chord_voice_corr_mean_sampler']} (higher = more like copies of one sound)"
        )
        print(
            f"  T2 15s sustain loop trace (0 = no loop): sampler={res[name]['sustain_loop_periodicity']['sampler']}"
            f"  quill={res[name]['sustain_loop_periodicity']['quill']}"
        )
    (ROOT / "results/structural_tests.json").write_text(json.dumps(res, indent=1))
    print(f"\naudio -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------- #
# T4 · parameter-space morphing (structurally impossible for a sampler: layering ≠ morphing)
# --------------------------------------------------------------------- #
def morph_snapshots(sa: dict, sb: dict, w: float = 0.5) -> dict:
    """Spectral parameter interpolation: yields an **instrument that never existed** (not two sounds layered).
    Aligned to A's component frequency axis, log-domain interpolation of H_steady/N_env; f0_ref is the geometric midpoint."""
    out = dict(sa)
    Ha = np.asarray(sa["H_env"], np.float64)
    Hb = np.asarray(sb["H_env"], np.float64)
    Ka = Ha.shape[0]
    fa = float(sa["f0_ref"])
    fb = float(sb["f0_ref"])
    pfa = sa.get("partial_freqs")
    xa = (
        np.asarray(pfa, np.float64)[:Ka]
        if pfa is not None and np.asarray(pfa).size
        else np.arange(1, Ka + 1) * fa
    )
    Kb = Hb.shape[0]
    # B's envelope is resampled to A's axis by **relative harmonic index** (pitch-independent timbre-shape interpolation)
    ta = np.arange(1, Ka + 1, dtype=float)
    tb = np.arange(1, Kb + 1, dtype=float)
    T = min(Ha.shape[1], Hb.shape[1])
    Hm = np.empty((Ka, T))
    for t in range(T):
        hb_on_a = np.interp(ta, tb, np.maximum(Hb[:, t], 1e-9))
        Hm[:, t] = np.exp((1 - w) * np.log(np.maximum(Ha[:Ka, t], 1e-9)) + w * np.log(hb_on_a))
    out["H_env"] = Hm.astype(np.float32)
    out["f0_ref"] = float(fa ** (1 - w) * fb**w)
    out["partial_freqs"] = (xa * (out["f0_ref"] / fa)).astype(np.float32)
    Na = np.asarray(sa.get("N_env"), np.float64)
    Nb = np.asarray(sb.get("N_env"), np.float64)
    if Na.size and Nb.size:
        Tn = min(Na.shape[1], Nb.shape[1])
        B = min(Na.shape[0], Nb.shape[0])
        out["N_env"] = np.exp(
            (1 - w) * np.log(np.maximum(Na[:B, :Tn], 1e-9))
            + w * np.log(np.maximum(Nb[:B, :Tn], 1e-9))
        ).astype(np.float32)
    return out


def peak_set(y, n=12):
    """Set of spectral peak positions (Hz): decides between "two peak sets coexisting" (layering) and "one set of intermediate peaks" (morphing)."""
    S = np.abs(np.fft.rfft(y[: 1 << 16] * np.hanning(min(len(y), 1 << 16))))
    f = np.fft.rfftfreq(min(len(y), 1 << 16), 1 / SR)
    m = (f > 100) & (f < 5000)
    S, f = S[m], f[m]
    idx = np.argsort(-S)[:400]
    picked = []
    for i in sorted(idx, key=lambda k: -S[k]):
        if all(abs(f[i] - p) > 25 for p in picked):
            picked.append(f[i])
        if len(picked) >= n:
            break
    return np.array(sorted(picked))


def morph_test():
    import soundfile as sf

    out = ROOT / "assets/ear_tests/structural"
    sa = load_snapshot(str(ROOT / "assets/snapshots/wineglass.npz"))
    sb = load_snapshot(str(ROOT / "assets/snapshots/guitar_rich.npz"))
    midi = 62
    f0 = 440 * 2 ** ((midi - 69) / 12)
    n = int(3.0 * SR)
    ya = render(sa, f0, length=n, mode="formant", add_attack=True)
    yb = render(sb, f0, length=n, mode="formant", add_attack=True)
    layer = rms_norm(
        0.5 * (ya / max(np.abs(ya).max(), 1e-9) + yb / max(np.abs(yb).max(), 1e-9))
    )  # sampler: layering
    sm = morph_snapshots(sa, sb, 0.5)
    ymorph = rms_norm(render(sm, f0, length=n, mode="formant", add_attack=True))
    sf.write(str(out / "morph_sampler_layered.wav"), layer, SR)
    sf.write(str(out / "morph_quill_50_50.wav"), ymorph, SR)
    for w, tag in ((0.25, "25"), (0.75, "75")):
        sw = morph_snapshots(sa, sb, w)
        sf.write(
            str(out / f"morph_quill_{tag}.wav"),
            rms_norm(render(sw, f0, length=n, mode="formant", add_attack=True)),
            SR,
        )
    pa, pb = peak_set(ya), peak_set(yb)
    pl, pm = peak_set(layer), peak_set(ymorph)

    def frac_from(p, src, tol=30):
        return float(np.mean([any(abs(x - s) < tol for s in src) for x in p])) if len(p) else 0.0

    r = {
        "layered_peaks_from_A": frac_from(pl, pa),
        "layered_peaks_from_B": frac_from(pl, pb),
        "morph_peaks_from_A": frac_from(pm, pa),
        "morph_peaks_from_B": frac_from(pm, pb),
    }
    print("\nT4 parameter morphing vs layering (spectral peak attribution):")
    print(
        f"  layering (what a sampler can do): from A {r['layered_peaks_from_A']:.0%}  from B {r['layered_peaks_from_B']:.0%}"
        f"  -> total {r['layered_peaks_from_A'] + r['layered_peaks_from_B']:.0%} (>100% = two peak sets coexist)"
    )
    print(
        f"  parameter morphing (Quill only): from A {r['morph_peaks_from_A']:.0%}  from B {r['morph_peaks_from_B']:.0%}"
        f"  -> total {r['morph_peaks_from_A'] + r['morph_peaks_from_B']:.0%} (low = new intermediate peaks, a new instrument)"
    )
    return r


if __name__ != "__main__":
    pass
