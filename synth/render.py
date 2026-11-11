"""Offline rendering: timbre snapshot -> waveform.

Resynthesises the snapshot at any target f0 (formant-preserving) and adds the noise side and attack residual. Through synth.additive and
synth.noise it depends only on numpy, sharing the same additive and noise math as the real-time callback.
"""

from __future__ import annotations

import numpy as np

from synth.additive import redirect_harmonics, resynth
from synth.noise import synth_noise


def render(
    snap: dict,
    f0_tgt: float | None = None,
    *,
    dur_s: float | None = None,
    length: int | None = None,
    mode: str = "formant",
    add_attack: bool = True,
    noise_gain: float = 1.0,
    seed: int = 0,
    env_correction=None,
) -> np.ndarray:
    """Render the snapshot at `f0_tgt` (default = f0_ref).

    mode="formant": formant-preserving; mode="index": control variant carried over by harmonic index.
    env_correction=(grid_hz, res_log): target-side envelope correction; after redirection, multiply each component at its target absolute
        frequency by exp(interp(f, grid, res)). None means no correction, bit-identical to the uncorrected path.
    """
    meta = snap.get("meta", {})
    sr = int(meta.get("sr", 44100))
    harm_hop = int(meta.get("harm_hop", 512))
    f0_ref = float(snap["f0_ref"])
    f0_tgt = float(f0_tgt) if f0_tgt is not None else f0_ref

    H = np.asarray(snap["H_env"], dtype=np.float64)
    pf = snap.get("partial_freqs")
    T = H.shape[1]

    if length is None:
        length = int(dur_s * sr) if dur_s else T * harm_hop

    # redirect the harmonic side
    H_tgt, pf_tgt = redirect_harmonics(H, f0_ref, f0_tgt, partial_freqs=pf, mode=mode)
    if env_correction is not None:
        grid_hz, res_log = env_correction
        K = H_tgt.shape[0]
        if pf is not None:
            targets = np.asarray(pf, np.float64)[:K] * (f0_tgt / f0_ref)
        else:
            targets = np.arange(1, K + 1, dtype=np.float64) * f0_tgt
        fac = np.exp(
            np.interp(targets, np.asarray(grid_hz, np.float64), np.asarray(res_log, np.float64))
        )
        H_tgt = H_tgt * fac[:, None]
    y_harm = resynth(
        H_tgt, f0_tgt, sr=sr, hop=harm_hop, partial_freqs=pf_tgt, f0_ref=f0_ref, length=length
    )

    # noise side (not pitch-shifted, broadband texture)
    y = y_harm.astype(np.float64)
    N_env = snap.get("N_env")
    if N_env is not None and np.asarray(N_env).size and noise_gain > 0:
        centers = snap.get("noise_centers_hz")
        n_fft = int(meta.get("noise_nfft", 1024))
        hop = int(meta.get("noise_hop", 256))
        y_noise = synth_noise(
            np.asarray(N_env), sr, length, n_fft=n_fft, hop=hop, centers_hz=centers, seed=seed
        )
        y[: len(y_noise)] += noise_gain * y_noise[:length]

    # attack residual: added at the start offline
    if add_attack:
        atk = np.asarray(snap.get("attack_res", []), dtype=np.float64)
        if atk.size:
            m = min(len(atk), length)
            y[:m] += atk[:m]

    # 5 ms equal-power fades at output start and end to avoid start/stop clicks
    fade = max(1, min(int(0.005 * sr), length // 2))
    ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return y.astype(np.float32)
