"""2×2 stimuli generator: the first consumer of .tplib v2 (sprint notebook §7.1; technical master document §4.2).

Behaviour transfer rules (§4.2, all applied to **residual** trajectories, the source's absolute values are not carried over):
  pitch (cents Δ)      -> per-frame f0: f0_note · 2^(Δp(τ)/1200)      (vibrato/slides/hammer-ons all kept)
  energy (ln-RMS Δ)    -> per-frame amplitude: exp(Δe(τ))                    (breathing/fluctuation)
  brightness (oct Δ)   -> per-frame tilt: gain_k ×(f_k/1kHz)^Δb(τ)      (brightness motion)
  noise (not used in v1)    -> recorded in the manifest, to be wired up on the consumer side later

Synthesis path = snapshot steady envelope (formant-preserving g_k) + resynth per-frame f0/H: the same math as L0;
with no behaviour trajectories (all zeros) it degrades to a static render = the control group, naturally α=0.

Output: the same phrase × {Voice, Glass timbre} × {Vocal, Guitar behaviour} 4 cells + 2 static controls,
RMS-matched, written to assets/ear_tests/stimuli_2x2_v1/.
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

from analysis.tplib import event_trajectories  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.additive import resynth  # noqa: E402

SR = 44100
HOP = 512
# shared phrase (midi, t_on, dur_s): five notes, with long/short contrast and a large leap
PHRASE = [(60, 0.00, 0.55), (62, 0.60, 0.55), (64, 1.20, 0.90), (67, 2.20, 0.55), (72, 2.80, 1.40)]
TAIL = 0.6


def _steady_gains(snap: dict, f0_note: float, K: int) -> np.ndarray:
    """Formant preservation: sample the steady envelope at the absolute frequencies k·f0_note (same semantics as voice._compute_gain_k)."""
    H = np.asarray(snap["H_env"], np.float64)
    T = H.shape[1]
    lo, hi = int(T * 0.25), max(int(T * 0.75), int(T * 0.25) + 1)
    Hs = np.median(H[:, lo:hi], axis=1)
    pf = snap.get("partial_freqs")
    xp = (
        np.asarray(pf, np.float64)[: len(Hs)]
        if pf is not None and np.asarray(pf).size
        else np.arange(1, len(Hs) + 1) * float(snap["f0_ref"])
    )
    tgt = np.arange(1, K + 1) * f0_note
    g = np.interp(tgt, xp, Hs, left=Hs[0], right=Hs[-1])
    hi_f = xp[-1]
    above = tgt > hi_f
    if np.any(above) and Hs[-1] > 0:
        octv = np.log2(np.maximum(tgt[above], hi_f) / hi_f)
        g[above] = Hs[-1] * 10 ** (-12.0 * octv / 20.0)
    return g


def synth_note(snap: dict, midi: int, dur_s: float, traj: dict | None) -> np.ndarray:
    """One note: additive resynthesis driven by behaviour trajectories. traj=None -> static control (only 8ms edge fades)."""
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    n = int(dur_s * SR)
    T = max(3, n // HOP)
    K = int(min(60, (SR / 2 - 800) / f0))
    g = _steady_gains(snap, f0, K)

    tau = np.linspace(0, 1, T)

    def tr(key, scale=1.0):
        if traj is None:
            return np.zeros(T)
        pts = np.asarray(traj[key]["pts"], np.float64)
        return np.interp(tau, np.linspace(0, 1, len(pts)), pts) * scale

    dp = tr("pitch")  # cents
    de = tr("energy")  # ln RMS
    db = np.clip(tr("brightness"), -1.0, 1.0)  # oct

    f0_t = f0 * 2 ** (dp / 1200.0)
    amp_t = np.exp(de)
    fk = np.arange(1, K + 1)[:, None] * f0  # [K,1] nominal frequencies
    tilt = (fk / 1000.0) ** db[None, :]  # [K,T] brightness tilt
    H = g[:, None] * amp_t[None, :] * tilt
    # global start/end protection (static control too): 8ms in + 60ms out
    env = np.ones(T)
    a = max(1, int(0.008 * SR / HOP))
    r = max(2, int(0.06 * SR / HOP))
    env[:a] *= np.linspace(0, 1, a)
    env[-r:] *= np.linspace(1, 0, r) ** 1.5
    H *= env[None, :]
    return resynth(H.astype(np.float32), f0_t, sr=SR, hop=HOP, length=n)


def behavior_events(wav_path: Path, max_events=8) -> list:
    """Behaviour source -> four trajectories per event. Try onset segmentation first, fall back to the whole segment as one event on failure."""
    try:
        import soundfile as sf

        y, sr = sf.read(str(wav_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if sr != SR:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    except Exception:  # compressed formats such as m4a go through librosa/audioread
        import librosa

        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        y = y.astype(np.float32)
    spans = []
    try:
        from analysis.segment import onset_f0_events

        evs = onset_f0_events(y, sr=SR)
        spans = [
            (e["t_on"], e["t_off"]) for e in evs if e.get("t_off", 0) - e.get("t_on", 0) > 0.15
        ][:max_events]
    except Exception:
        pass
    if not spans:
        spans = [(0.0, len(y) / SR)]
    out = []
    for t0, t1 in spans:
        tj = event_trajectories(y, SR, t0, t1)
        if tj["energy"]["confidence"] > 0:
            out.append(tj)
    return out or [event_trajectories(y, SR, 0.0, len(y) / SR)]


def render_cell(snap: dict, events: list | None) -> np.ndarray:
    """**Continuous-line** phrase synthesis (A1, 2026-08-01): synthesising each note separately and summing = the sampler's failure mode
    (amplitude returns to zero between notes, each note restarts). Here the whole phrase is built as one continuous f0/H timeline,
    f0 glides within 60ms between adjacent notes and energy does not return to zero: the same semantics as the engine's legato."""
    total = int((PHRASE[-1][1] + PHRASE[-1][2] + TAIL) * SR)
    T = max(8, total // HOP)
    t_fr = np.arange(T) * HOP / SR
    K_max = 60

    # 1) nominal f0 timeline: held within a note, 60ms log-domain glide between notes (= legato connection)
    f0_nom = np.zeros(T)
    glide = 0.06
    for i, (midi, t_on, dur) in enumerate(PHRASE):
        f_here = 440.0 * 2 ** ((midi - 69) / 12)
        if i == 0:
            f0_nom[t_fr < t_on + dur] = f_here
            continue
        prev_f = 440.0 * 2 ** ((PHRASE[i - 1][0] - 69) / 12)
        m_gl = (t_fr >= t_on) & (t_fr < t_on + glide)
        u = np.clip((t_fr[m_gl] - t_on) / glide, 0, 1)
        f0_nom[m_gl] = prev_f * (f_here / prev_f) ** u  # linear in log domain = perceptually linear
        f0_nom[(t_fr >= t_on + glide) & (t_fr < t_on + dur)] = f_here
    f0_nom[f0_nom <= 0] = f0_nom[f0_nom > 0][-1] if np.any(f0_nom > 0) else 440.0

    # 2) lay behaviour trajectories onto the timeline (one per note, taken from the event the note belongs to)
    dp = np.zeros(T)
    de = np.zeros(T)
    db = np.zeros(T)
    for i, (_midi, t_on, dur) in enumerate(PHRASE):
        tj = events[i % len(events)] if events else None
        m = (t_fr >= t_on) & (t_fr < t_on + dur)
        if not m.any() or tj is None:
            continue
        tau = np.linspace(0, 1, int(m.sum()))
        for key, dst in (("pitch", dp), ("energy", de), ("brightness", db)):
            pts = np.asarray(tj[key]["pts"], np.float64)
            dst[m] = np.interp(tau, np.linspace(0, 1, len(pts)), pts)
    db = np.clip(db, -1.0, 1.0)

    # 3) synthesise the whole line at once (formant preservation: steady gains taken per frame at the current nominal pitch)
    f0_t = f0_nom * 2 ** (dp / 1200.0)
    amp_t = np.exp(de)
    H = np.zeros((K_max, T))
    for i, (_midi, t_on, _dur) in enumerate(PHRASE):  # take g per segment (pitch is constant within a note)
        # coverage extends to the **start of the next note** (the score has gaps between notes, but legato playing is connected:
        # leaving the gap at zero = per-note retrigger = sampler behaviour. Here the sound continues across the gap)
        t_next = PHRASE[i + 1][1] if i + 1 < len(PHRASE) else 1e9
        m = (t_fr >= t_on) & (t_fr < t_next)
        if not m.any():
            continue
        f_here = 440.0 * 2 ** ((midi - 69) / 12)
        K = int(min(K_max, (SR / 2 - 800) / f_here))
        g = _steady_gains(snap, f_here, K)
        fk = np.arange(1, K + 1)[:, None] * f_here
        H[:K, m] = g[:, None] * (fk / 1000.0) ** db[None, m]
    H *= amp_t[None, :]
    # whole-phrase start/end (not per note!) + natural decay at the end
    env = np.ones(T)
    a = max(1, int(0.01 * SR / HOP))
    env[:a] = np.linspace(0, 1, a)
    tail_fr = int(TAIL * SR / HOP)
    env[-tail_fr:] = np.linspace(1, 0, tail_fr) ** 2.0
    H *= env[None, :]

    y = resynth(H.astype(np.float32), f0_t, sr=SR, hop=HOP, length=total).astype(np.float64)
    r = np.sqrt((y**2).mean())
    return (y * (0.08 / max(r, 1e-9))).astype(np.float32)


def main():
    """2×2 (positioning revision 2026-08-01: voice dropped) = {Glass, Guitar timbre} × {Glass, Guitar behaviour}.
    The glass behaviour source (rim-rubbing performance) awaits the user's recording; for now the guitar behaviour column + two static controls are produced."""
    import soundfile as sf

    timbres = {
        "glass": ROOT / "assets/snapshots/wineglass.npz",
        "guitar": ROOT / "assets/snapshots/guitar.npz",
    }
    behaviors = {"guitar": ROOT / "assets/recordings/performance/guitar_perf_01.m4a"}
    glass_perf = ROOT / "assets/recordings/performance/glass_perf_01.wav"
    if glass_perf.exists():  # automatically enters the 2×2 once the user has recorded it
        behaviors["glass"] = glass_perf
    out = ROOT / "assets/ear_tests/stimuli_2x2_v1"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"phrase": PHRASE, "cells": {}}
    beh_events = {}
    for bname, bpath in behaviors.items():
        beh_events[bname] = behavior_events(bpath)
        print(f"behavior[{bname}]: {len(beh_events[bname])} events  ({bpath.name})")
    for tname, tpath in timbres.items():
        snap = load_snapshot(str(tpath))
        y = render_cell(snap, None)
        sf.write(str(out / f"{tname}_static.wav"), y, SR)
        manifest["cells"][f"{tname}_static"] = {"timbre": tname, "behavior": None}
        for bname in behaviors:
            y = render_cell(snap, beh_events[bname])
            sf.write(str(out / f"{tname}_x_{bname}.wav"), y, SR)
            manifest["cells"][f"{tname}_x_{bname}"] = {"timbre": tname, "behavior": bname}
            print(f"  {tname} × {bname} ✓")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest['cells'])} files → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
