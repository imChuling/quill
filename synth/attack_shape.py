"""Pick thickness: a playable parameter for attack hardness.

`pick`∈[0,1] compresses attack differences into one interpretable knob: 0 is fingerstyle / thin pick (light, dark,
round attack grain, longer attack fade), 1 is a thick pick (heavy, bright, sharp grain, no fade). Three mappings:
attack-residual grain gain ∝ pick; grain brightness, lowpass cutoff rises with pick; attack fade length ∝ (1-pick).

Depends only on numpy. The lowpass is an rfft brick wall; grains are short, so ringing is negligible.
"""

from __future__ import annotations

import numpy as np


def _lowpass_fft(x, cutoff_hz, sr):
    """Brick-wall lowpass; grain-sized input (< 3k samples), ringing is harmless."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[f > cutoff_hz] = 0.0
    return np.fft.irfft(X, n=len(x))


def shape_attack(grain, pick: float, sr: int = 44100):
    """Shape the attack-residual grain by pick thickness. pick∈[0,1] -> shaped grain (float32).

    pick=0 returns zeros (pure fingerstyle, no transient); pick=1 unchanged (sharpest, brightest); in between = lowpass + attenuation.
    """
    grain = np.asarray(grain, dtype=np.float64)
    pick = float(np.clip(pick, 0.0, 1.0))
    if grain.size == 0 or pick <= 0.0:
        return np.zeros_like(grain, dtype=np.float32)
    if pick >= 1.0:
        return grain.astype(np.float32)
    cutoff = min(600.0 * 2.0 ** (pick * 4.0), sr * 0.49)  # soft 600 Hz -> hard about 9.6 kHz
    g = _lowpass_fft(grain, cutoff, sr)
    return (pick * g).astype(np.float32)


def onset_fade(n_samples_hint, pick: float, sr: int = 44100, max_ms: float = 40.0):
    """Return the attack fade-in envelope (square root of raised cosine, length scales with (1-pick)); length 0 when pick=1."""
    pick = float(np.clip(pick, 0.0, 1.0))
    fade = int(sr * (max_ms / 1000.0) * (1.0 - pick))
    if fade <= 1:
        return None
    return np.sqrt(0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))).astype(np.float32)


def apply_pick(body, attack_res, pick: float, sr: int = 44100):
    """Add the shaped attack onto a body rendered without a hard attack, and apply the attack fade by pick.

    body: render without attack (render(..., add_attack=False)); attack_res: snapshot attack residual.
    Returns audio with a "pick thickness" attack (float32).
    """
    y = np.asarray(body, dtype=np.float64).copy()
    g = shape_attack(attack_res, pick, sr)
    if g.size:
        m = min(len(g), len(y))
        y[:m] += g[:m]
    fade = onset_fade(len(y), pick, sr)
    if fade is not None:
        m = min(len(fade), len(y))
        y[:m] *= fade[:m]
    return y.astype(np.float32)
