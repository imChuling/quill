"""Playable instrument: timbre snapshot -> precomputed structure for the real-time engine.

Compresses the time-varying analysis snapshot into the steady data the real-time voice needs:
  H_steady[K]   steady harmonic amplitudes (median over the sustain segment)
  ratio_k[K]    frequency ratio per component (strict harmonic = k; quasi-harmonic = partial_freqs/f0_ref)
  oct_k[K]      log2(ratio_k), brightness tilt gain is computed per octave
  attack_grain  attack residual grain (played back on note-on)
  atk_traj      attack spectral trajectory; spec_traj/amp_traj sustain spectral shape and loudness trajectories; amp_env natural amplitude envelope
  noise_table   noise texture prerendered with a fixed seed; real time only does table lookup, results are reproducible

Depends only on numpy. Hot swap: the Engine holds a reference to the instrument; a worker thread builds the new
Instrument and rebinds the reference directly (reference assignment is atomic in CPython); the old one is GC'd.
"""

from __future__ import annotations

import numpy as np

from synth.noise import synth_noise


class Instrument:
    def __init__(
        self,
        H_steady,
        ratio_k,
        attack_grain,
        noise_table,
        f0_ref,
        sr,
        name="untitled",
        noise_level=0.5,
        vib_rate=None,
        amp_env=None,
        amp_hop=512,
        spec_traj=None,
        traj_fps=30.0,
        amp_traj=None,
        amp_tail_ratio=1.0,
        atk_traj=None,
        atk_fps=86.13,
    ):
        self.H_steady = np.ascontiguousarray(H_steady, dtype=np.float64)
        self.ratio_k = np.ascontiguousarray(ratio_k, dtype=np.float64)
        self.oct_k = np.log2(np.maximum(self.ratio_k, 1e-9))
        self.attack_grain = np.ascontiguousarray(attack_grain, dtype=np.float64)
        self.noise_table = np.ascontiguousarray(noise_table, dtype=np.float64)
        self.f0_ref = float(f0_ref)
        self.sr = int(sr)
        self.K = int(len(self.H_steady))
        # attack spectral trajectory [K,Fa]: per-partial factor relative to H_steady, native frame rate, no downsampling.
        # Each partial's own attack / early decay is the physical source of the plucked feel: high partials decay fast, the fundamental lingers.
        at_ = np.asarray(atk_traj, dtype=np.float64) if atk_traj is not None else None
        self.atk_traj = (
            np.ascontiguousarray(at_)
            if at_ is not None and at_.ndim == 2 and at_.shape[0] == self.K and at_.shape[1] >= 2
            else None
        )
        self.atk_fps = float(atk_fps)
        self.name = name
        self.noise_level = float(noise_level)
        # CC1 vibrato rate taken from the source clip measurement (snapshot.vibrato.rate_hz); None uses the engine default
        self.vib_rate = float(vib_rate) if vib_rate else None
        # natural amplitude envelope (post-peak decay ∈[0,1], natural rate): plucked sources decay, sustained ones ≈1. Only loudness decays;
        # spectral shape is still set by gain_k.
        self.amp_env = (
            np.ascontiguousarray(amp_env, dtype=np.float64)
            if amp_env is not None and len(amp_env) > 1
            else None
        )
        self.amp_hop = int(amp_hop)
        # sustain spectral-shape trajectory [K,F]: per-frame factor relative to H_steady, energy-normalised, shape motion only.
        # The renderer ping-pong loops along it so sustained notes carry the source recording's own fluctuation.
        st = np.asarray(spec_traj, dtype=np.float64) if spec_traj is not None else None
        self.spec_traj = (
            np.ascontiguousarray(st)
            if st is not None and st.ndim == 2 and st.shape[1] >= 2
            else None
        )
        self.traj_fps = float(traj_fps)
        # loudness trajectory [F], same frames as spec_traj: level fluctuation of the source sustain, the audible core of breathing
        at = np.asarray(amp_traj, dtype=np.float64) if amp_traj is not None else None
        self.amp_traj = (
            np.ascontiguousarray(at)
            if at is not None and self.spec_traj is not None and len(at) == self.spec_traj.shape[1]
            else None
        )
        self.amp_tail_ratio = float(amp_tail_ratio)  # per-frame decay ratio applied beyond the end of the amp_env table


def build_instrument(
    snap: dict,
    noise_level: float = 0.5,
    noise_table_s: float = 2.0,
    noise_seed: int = 1234,
    pick: float = 0.5,
) -> Instrument:
    """Build an Instrument from a snapshot (one-off precomputation, not on the real-time path).

    pick∈[0,1]: pick thickness, shapes the attack grain (see synth.attack_shape). The natural amplitude envelope is extracted
    from the H_env energy contour (post-peak decay) so plucked sources also decay in the real-time engine.
    """
    sr = int(snap.get("meta", {}).get("sr", 44100))
    harm_hop = int(snap.get("meta", {}).get("harm_hop", 512))
    H = np.asarray(snap["H_env"], dtype=np.float64)  # [K, T]
    K, T = H.shape
    # steady harmonic amplitude: median over the middle 50% of frames, avoiding attack and tail
    lo, hi = int(T * 0.25), max(int(T * 0.75), int(T * 0.25) + 1)
    H_steady = np.median(H[:, lo:hi], axis=1)

    # natural amplitude envelope = monotonic decay trend of post-peak energy (linear fit in log domain). The fit window ends at 75%,
    # avoiding the clip's tail-off: the tail-off is not the instrument's sustain behaviour. Plucked/struck sources give a clean exponential
    # decay; sustained sources have slope ≈0, envelope ≈flat, and in-clip fluctuation goes to amp_traj.
    energy = np.sqrt(np.maximum((H**2).sum(axis=0), 0.0))
    pk = int(np.argmax(energy)) if energy.size else 0
    epk = energy[pk] if energy.size and energy[pk] > 0 else 1.0
    amp_env = np.ones(T, dtype=np.float64)
    amp_tail_ratio = 1.0
    fit_start = max(pk, int(T * 0.25))  # fit from the sustain region: attack-to-sustain settling belongs to ADSR, not decay
    fit_end = max(int(T * 0.75), fit_start + 5)
    if T > 1 and fit_end - fit_start >= 5 and fit_end <= T:
        tt = (np.arange(fit_start, fit_end) - fit_start) * (harm_hop / sr)
        ln_e = np.log(np.maximum(energy[fit_start:fit_end] / epk, 1e-6))
        k = min(float(np.polyfit(tt, ln_e, 1)[0]), 0.0)  # decay only; slope in ln domain, unit 1/s
        t_post = np.maximum((np.arange(T) - pk), 0) * (harm_hop / sr)
        amp_env = np.exp(k * t_post)
        amp_tail_ratio = float(np.exp(k * (harm_hop / sr)))  # beyond the table, keep decaying at the same rate

    # attack spectral trajectory: per-partial factor H(t)/H_steady within [0, peak+ATK_MS], native frame rate since the attack needs
    # time resolution. The harmonic layer produces the pluck itself instead of relying on a pasted-on noise grain.
    ATK_MS = 320.0
    atk_traj = None
    frame_ms = harm_hop / sr * 1000.0
    n_atk = int(min(T, max(4, round((ATK_MS + pk * frame_ms) / frame_ms))))
    if T >= 6 and n_atk >= 4:
        a_seg = H[:, :n_atk]
        # per-frame energy normalisation (same convention as spec_traj): the attack trajectory carries only spectral-shape change, not loudness,
        # which is handled by ADSR/amp_env. Keeping the level would double count: partials momentarily jump several times and the bus clips into a pop.
        ref_e = float(np.sqrt((H_steady**2).sum()))
        seg_e = np.sqrt(np.maximum((a_seg**2).sum(axis=0), 1e-24))
        a_seg = a_seg * (ref_e / seg_e)[None, :]
        at_ = a_seg / np.maximum(H_steady, 1e-9)[:, None]
        at_ = np.clip(np.nan_to_num(at_, nan=1.0, posinf=3.0), 0.0, 3.0)
        # smooth the last 6 frames toward 1 for a seamless join with the steady state
        m = min(6, at_.shape[1])
        ramp = np.linspace(0.0, 1.0, m)[None, :]
        at_[:, -m:] = at_[:, -m:] * (1 - ramp) + ramp
        atk_traj = at_

    # spectral-shape and loudness trajectories: sustain H(t) downsampled to about 30 fps. The shape trajectory is the per-frame energy-normalised
    # shape factor; the loudness trajectory is per-frame energy fluctuation, full for sustained sources, scaled down by sustain degree for plucked ones to avoid double counting with amp_env.
    frame_fps = sr / max(harm_hop, 1)
    step = max(1, int(round(frame_fps / 30.0)))
    seg = H[:, lo:hi:step]
    traj_fps = frame_fps / step
    spec_traj = None
    amp_traj = None
    if seg.shape[1] >= 4:
        # breathing-band lowpass: in unstable recordings the trajectory can jump up to 0.15 between frames (normal breathing < 0.01); frame-by-frame
        # playback sounds like rough noise. Breathing is a < 5 Hz phenomenon: two moving-average passes (cutoff about 2.5 Hz) remove analysis
        # noise faster than breathing, then a depth cap is applied, since natural sustained notes never breathe by ±12 dB.
        w = max(3, int(round(traj_fps / 5.0)) | 1)

        def _ma2(x2d, win):
            pad = win // 2
            kk = np.ones(win) / win
            out = x2d
            for _ in range(2):
                xp = np.pad(out, ((0, 0), (pad, pad)), mode="edge")
                out = np.stack(
                    [
                        np.convolve(xp[i], kk, mode="valid")[: x2d.shape[1]]
                        for i in range(x2d.shape[0])
                    ]
                )
            return out

        ref_norm = float(np.sqrt((H_steady**2).sum()))
        en = np.sqrt(np.maximum((seg**2).sum(axis=0), 1e-24))
        seg_n = seg * (ref_norm / en)[None, :]
        traj = np.clip(seg_n / np.maximum(H_steady, 1e-9)[:, None], 0.5, 2.0)
        traj = _ma2(traj, w)
        wgt = H_steady / (H_steady.sum() + 1e-12)
        dev = float((np.abs(traj - 1.0) * wgt[:, None]).sum(axis=0).mean())
        if dev > 0.20:  # spectral-shape breathing depth cap
            traj = 1.0 + (traj - 1.0) * (0.20 / dev)
        spec_traj = traj
        # sustain degree, the gate for amp_traj: amp_env is a decay-trend fit, ≈flat for sustained sources gives 1, already decayed mid-pluck gives 0
        sus = float(np.clip((float(np.median(amp_env[lo:hi])) - 0.55) / 0.3, 0.0, 1.0))
        raw = np.clip(en / max(ref_norm, 1e-12), 0.7, 1.4)
        at = 1.0 + (raw - 1.0) * sus
        at = _ma2(at[None, :], w)[0]
        adev = float(np.abs(at - 1.0).mean())
        if adev > 0.12:  # loudness breathing depth cap, mean about ±1 dB
            at = 1.0 + (at - 1.0) * (0.12 / adev)
        amp_traj = at

    pf = snap.get("partial_freqs")
    f0_ref = float(snap["f0_ref"])
    if pf is not None and np.asarray(pf).size:
        ratio_k = np.asarray(pf, dtype=np.float64)[:K] / f0_ref
    else:
        ratio_k = np.arange(1, K + 1, dtype=np.float64)

    attack = np.asarray(snap.get("attack_res", np.zeros(0)), dtype=np.float64)
    if attack.size:  # shape the attack grain by pick thickness
        from synth.attack_shape import shape_attack

        attack = np.asarray(shape_attack(attack, pick, sr), dtype=np.float64)
        np.nan_to_num(attack, copy=False)  # a NaN-polluted grain gets flushed to 0 at render time, causing an abrupt click

    # noise texture table: prerendered with a fixed seed, the real-time side only looks it up
    N_env = snap.get("N_env")
    centers = snap.get("noise_centers_hz")
    tlen = int(noise_table_s * sr)
    if N_env is not None and np.asarray(N_env).size and noise_level > 0:
        # render an extra crossfade segment and mix the table tail into the head at equal power so the loop is seamless; otherwise a sustained note
        # clicks at the wrap point every time it passes the table length
        xf = int(0.05 * sr)
        ext = synth_noise(np.asarray(N_env), sr, tlen + xf, centers_hz=centers, seed=noise_seed)
        noise_table = ext[:tlen].copy()
        ramp = np.sin(np.linspace(0, np.pi / 2, xf)) ** 2  # equal power 0->1
        noise_table[:xf] = noise_table[:xf] * ramp + ext[tlen : tlen + xf] * ramp[::-1]
    else:
        noise_table = np.zeros(tlen, dtype=np.float32)

    vib = snap.get("vibrato") or {}
    return Instrument(
        H_steady,
        ratio_k,
        attack,
        noise_table,
        f0_ref,
        sr,
        name=snap.get("meta", {}).get("name", "untitled"),
        noise_level=noise_level,
        vib_rate=vib.get("rate_hz"),
        amp_env=amp_env,
        amp_hop=harm_hop,
        spec_traj=spec_traj,
        traj_fps=traj_fps,
        amp_traj=amp_traj,
        amp_tail_ratio=amp_tail_ratio,
        atk_traj=atk_traj,
        atk_fps=sr / max(harm_hop, 1),
    )
