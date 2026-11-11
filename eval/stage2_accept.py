"""Stage 2 acceptance script: FR-2 complete + FR-3.

Exit criterion: three examples (voice / guitar single note / wine glass) audible at ±12 semitones, with formants preserved.

For each example:
  audio -> analyze_timbre -> TimbreSnapshot(.npz) -> render(-12 / 0 / +12)
  and also render the +12 "carried over by index" wrong version (chipmunk) as an archived reference (report material).

Outputs:
  assets/snapshots/<name>.npz                     timbre snapshot
  assets/demos/<name>_orig.wav                    original
  assets/demos/<name>_re{0,up12,down12}.wav       formant-preserving resynthesis (±12)
  assets/demos/<name>_up12_index.wav              carried-over-by-index wrong version (reference)
  assets/demos/<name>_fr3.png                     formant preservation vs carry-over spectrum comparison

Usage:
  python -m eval.stage2_accept                 # built-in synthetic material
  python -m eval.stage2_accept guitar.wav      # real single note (harmonic source by default)
  python -m eval.stage2_accept --quasi bowl.wav  # quasi-harmonic source (wine glass/bowl)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.io import normalize  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from eval.stage1_accept import synth_wineglass  # noqa: E402
from snapshot import save_snapshot  # noqa: E402
from synth.render import render  # noqa: E402

SR = 44100
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "assets" / "demos"
SNAP_DIR = ROOT / "assets" / "snapshots"


def synth_voice_steady(dur=2.0, sr=SR):
    """Voice-like **steady** long note (two formants, no vibrato).

    Per the material capture protocol (A5.5) timbre snapshots are only taken from isolated sustained notes: a steady f0 is a prerequisite for a clean residual/
    noise decomposition (vibrato drifts harmonic frequencies within the analysis window and pollutes the residual, a known limitation).
    """
    t = np.arange(int(sr * dur)) / sr
    f0 = 174.6  # F3
    y = np.zeros_like(t)
    for k in range(1, 41):
        fk = k * f0
        if fk >= sr / 2:
            break
        form = np.exp(-(((fk - 700) / 350) ** 2)) + 0.6 * np.exp(-(((fk - 1800) / 600) ** 2)) + 0.03
        y += form * np.cos(2 * np.pi * fk * t)
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (dur - t) / 0.1)
    return normalize((y * env).astype(np.float32)), "voice"


def synth_guitar(dur=2.0, sr=SR):
    """Guitar-like single note: exponential harmonic decay (faster for higher harmonics) + pluck noise at the attack."""
    t = np.arange(int(sr * dur)) / sr
    f0 = 146.83  # D3
    y = np.zeros_like(t)
    for k in range(1, 36):
        fk = k * f0
        if fk >= sr / 2:
            break
        tau = 1.2 / (1.0 + 0.5 * k)
        y += (1.0 / k) * np.exp(-t / tau) * np.sin(2 * np.pi * fk * t + 0.2 * k)
    # pluck transient: ~6ms broadband noise burst
    rng = np.random.default_rng(0)
    pluck = rng.standard_normal(len(t)) * np.exp(-t / 0.006)
    y = y + 0.3 * pluck
    return normalize(y.astype(np.float32)), "guitar"


def _spectral_envelope_peak(y, sr, lo=200.0, hi=3000.0, smooth_hz=300.0):
    """Rough spectral-envelope peak frequency (Hz): smooth the **linear power** spectrum and take the peak in [lo,hi].

    Smoothing linear power (not log magnitude): the log domain is dominated by the deep valleys and density between harmonics,
    giving a wrong "envelope peak"; smoothing linear power follows the harmonic energy distribution directly and locates formants reliably.
    """
    seg = y[len(y) // 8 : -len(y) // 8] if len(y) > 16 else y
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2  # power spectrum
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    df = f[1] - f[0]
    win = max(1, int(smooth_hz / df))
    kern = np.ones(win) / win
    sm = np.convolve(S, kern, mode="same")
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return float("nan")
    idx = np.where(band)[0]
    return float(f[idx[np.argmax(sm[idx])]])


def _mag_db(y, sr):
    seg = y[len(y) // 8 : -len(y) // 8] if len(y) > 16 else y
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    return f, 20 * np.log10(S / (S.max() + 1e-12) + 1e-12)


def _smoothed_env_db(y, sr, sigma_hz=250.0):
    """Gaussian smoothing of the linear power spectrum -> dB spectral-envelope curve (shows formant positions clearly)."""
    seg = y[len(y) // 8 : -len(y) // 8] if len(y) > 16 else y
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    df = f[1] - f[0]
    sigma_bins = max(1.0, sigma_hz / df)
    half = int(4 * sigma_bins)
    x = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kern /= kern.sum()
    sm = np.convolve(S, kern, mode="same")
    return f, 10 * np.log10(sm / (sm.max() + 1e-12) + 1e-12)


def _save_fr3_plot(name, orig, re0, up_form, up_idx, sr, formant_hz):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # thin line: raw line spectrum (faint); thick line: smoothed spectral envelope (tells the story)
    f, s = _mag_db(up_form, sr)
    ax.plot(f, s, lw=0.5, alpha=0.25, color="C1")
    f, s = _mag_db(up_idx, sr)
    ax.plot(f, s, lw=0.5, alpha=0.25, color="C2")
    for sig, lab, col in (
        (orig, "Original envelope (f0_ref)", "C0"),
        (up_form, "+12 formant-preserving env", "C1"),
        (up_idx, "+12 by-index (chipmunk) env", "C2"),
    ):
        f, e = _smoothed_env_db(sig, sr)
        ax.plot(f, e, lw=2.0, color=col, label=lab)
    if np.isfinite(formant_hz):
        ax.axvline(formant_hz, color="k", ls=":", lw=1.2, label=f"orig formant ~{formant_hz:.0f}Hz")
    ax.set_xlim(0, 5000)
    ax.set_ylim(-60, 3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title(f"{name}  FR-3: formant preserved vs shifted (+12 semitones)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DEMO_DIR / f"{name}_fr3.png", dpi=110)
    plt.close(fig)


def process(y, sr, name, quasi=False):
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    y = normalize(np.asarray(y, dtype=np.float32))

    snap = analyze_timbre(y, sr=sr, name=name, quasi_harmonic=quasi)
    snap_path = save_snapshot(SNAP_DIR / f"{name}.npz", snap)

    f0r = snap["f0_ref"]
    length = len(y)
    # formant-preserving rendering: -12 / 0 / +12 semitones
    re0 = render(snap, f0r, length=length, mode="formant")
    up = render(snap, f0r * 2.0, length=length, mode="formant")  # +12
    down = render(snap, f0r / 2.0, length=length, mode="formant")  # -12
    up_idx = render(snap, f0r * 2.0, length=length, mode="index")  # wrong reference version

    outs = {
        "orig": y,
        "re0": re0,
        "up12": up,
        "down12": down,
        "up12_index": up_idx,
    }
    for tag, sig in outs.items():
        sf.write(DEMO_DIR / f"{name}_{tag}.wav", normalize(sig), sr)

    # formant preservation measure: spectral-envelope peak (should barely move at ±12)
    formant_orig = _spectral_envelope_peak(y, sr)
    formant_up_form = _spectral_envelope_peak(up, sr)
    formant_up_idx = _spectral_envelope_peak(up_idx, sr)
    _save_fr3_plot(name, y, re0, up, up_idx, sr, formant_orig)

    # resynthesis fidelity (same f0): spectral-envelope peak drift
    drift_re0 = abs(_spectral_envelope_peak(re0, sr) - formant_orig)

    print(f"\n=== {name} ===")
    print(f"  f0_ref               : {f0r:8.2f} Hz")
    print(f"  vibrato              : {snap['vibrato']}")
    print(
        f"  partial_freqs        : "
        f"{'measured first 4=' + str(np.round(snap['partial_freqs'][:4], 1).tolist()) if snap['partial_freqs'] is not None else 'None (strictly harmonic)'}"
    )
    print(f"  envelope peak, original         : {formant_orig:8.1f} Hz")
    print(
        f"  envelope peak, +12 formant-preserving: {formant_up_form:8.1f} Hz  "
        f"(drift {abs(formant_up_form - formant_orig):.0f}Hz, should be small)"
    )
    print(
        f"  envelope peak, +12 by index   : {formant_up_idx:8.1f} Hz  "
        f"(drift {abs(formant_up_idx - formant_orig):.0f}Hz, should ≈ double)"
    )
    print(f"  same-f0 resynthesis peak drift     : {drift_re0:8.1f} Hz")
    print(
        f"  snapshot -> {snap_path.name}  ({snap['H_env'].shape[1]} H frames, "
        f"{snap['N_env'].shape[1]} N frames)"
    )
    return {
        "name": name,
        "f0": f0r,
        "formant_orig": formant_orig,
        "drift_re0": drift_re0,
        "drift_form": abs(formant_up_form - formant_orig),
        "drift_idx": abs(formant_up_idx - formant_orig),
        "clean": bool(np.all(np.isfinite(up)) and np.max(np.abs(up)) < 10.0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill stage 2 acceptance (FR-2 complete + FR-3)")
    ap.add_argument("wavs", nargs="*", help="real single-note wav")
    ap.add_argument(
        "--quasi", action="store_true", help="quasi-harmonic source (wine glass/bowl): widen the peak search window and measure partial_freqs"
    )
    args = ap.parse_args(argv)

    jobs = []
    if args.wavs:
        from analysis.io import load

        for w in args.wavs:
            y, _ = load(w, sr=SR)
            jobs.append((y, Path(w).stem, args.quasi))
    else:
        print("No recordings given, using built-in synthetic single notes (voice/guitar/wine glass).")
        jobs.append((synth_voice_steady()[0], "voice", False))
        jobs.append((synth_guitar()[0], "guitar", False))
        jobs.append((synth_wineglass()[0], "wineglass", True))

    results = [process(y, SR, nm, q) for (y, nm, q) in jobs]

    print("\n---------------- Stage 2 acceptance summary ----------------")
    print(
        "Criteria: (1) same-f0 resynthesis fidelity (peak drift<150Hz) (2) clean render (finite, bounded) (3) formant preservation no worse than carry-over by index"
    )
    print(
        "(note: guitar/wine glass have monotonically rolling-off spectra with no explicit formants, so the two modes are naturally close; "
        "the voice has a ~700Hz formant, where the FR-3 advantage is clear)"
    )
    ok = True
    for r in results:
        faithful = r["drift_re0"] < 150.0
        not_worse = r["drift_form"] <= r["drift_idx"] + 30.0
        ok_row = faithful and not_worse and r["clean"]
        ok = ok and ok_row
        flag = "OK " if ok_row else "WARN"
        note = "  <- formant preservation clearly better" if r["drift_form"] < 0.7 * r["drift_idx"] else ""
        print(
            f"  [{flag}] {r['name']:10s} f0={r['f0']:7.1f}Hz  "
            f"same-f0 drift={r['drift_re0']:4.0f}Hz  "
            f"+12: formant={r['drift_form']:4.0f}Hz / index={r['drift_idx']:4.0f}Hz{note}"
        )
    print(f"Outputs: {DEMO_DIR}  |  snapshots: {SNAP_DIR}")
    print(
        "Blind listening at ±12 semitones: *_down12 / *_re0 / *_up12 should have a consistent timbre (no chipmunk); "
        "compare *_up12_index to hear the chipmunk effect."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
