"""Real-time additive oscillator kernel: the hot path of voice rendering.

`@njit` fused inner loop: advances each harmonic's phase per sample and accumulates into the output block, with linear ramps of
f0/loudness/gain within the block; harmonics crossing Nyquist fade out by gain instead of a hard cut. Phase is wrapped at block end to avoid cos precision loss on large values.

Depends only on numpy and numba. The kernel allocates nothing: the caller passes preallocated out and phase. When numba is unavailable
(or QUILL_NO_JIT=1) it falls back to a pure-Python function of the same name: slow, only guaranteed to run and be testable.
"""

from __future__ import annotations

import math
import os as _os

import numpy as np

try:
    if _os.environ.get("QUILL_NO_JIT") == "1":
        raise ImportError("JIT disabled by QUILL_NO_JIT")
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn

        return deco(args[0]) if args and callable(args[0]) else deco


_TWO_PI = 2.0 * math.pi


@njit(cache=True, fastmath=True)
def voice_block_add(
    out, n, phase, ratio, gain_a, gain_b, K, f0_a, f0_b, loud_a, loud_b, sr, nyquist, ramp_hz
):
    """Accumulate one voice's harmonic contribution for n samples into out[:n].

    out      : float64[>=n]  main mix block, accumulated in place
    phase    : float64[K]    persistent phase, updated in place; reset by the caller on note-on
    ratio    : float64[K]    frequency ratio of component k (strict harmonic = k, quasi-harmonic = pf/f0_ref)
    gain_a/b : float64[K]    per-harmonic amplitude at block start/end, linearly interpolated within the block. The attack spectral trajectory and tilt
               make gain change fast; holding it constant per block would turn every block boundary into a step, heard as pops or stutter.
    f0_a/f0_b: f0 at block start/end, linear ramp within the block, prevents zipper noise from f0 jumps
    loud_a/b : total loudness at block start/end (ADSR × velocity × fade-in), linear ramp

    Constant-frequency blocks take a phasor-recursion fast path (no cos in the inner loop, only complex rotation); variable-frequency
    bend/slide blocks take a per-sample cos slow path. Both skip components with gain==0 and harmonics above Nyquist.
    """
    inv = 1.0 / sr
    dl = (loud_b - loud_a) / n

    if f0_a == f0_b:  # ---- constant-frequency fast path ----
        base = _TWO_PI * f0_a * inv
        for k in range(K):
            ga = gain_a[k]
            gb = gain_b[k]
            if ga == 0.0 and gb == 0.0:
                continue
            rk = ratio[k]
            fk = rk * f0_a
            if fk >= nyquist:
                continue
            if fk > nyquist - ramp_hz:
                nq = (nyquist - fk) / ramp_hz  # soft fade-out near Nyquist
                ga = ga * nq
                gb = gb * nq
            dg = (gb - ga) / n
            dth = base * rk
            c = math.cos(dth)
            s = math.sin(dth)
            ph0 = phase[k]
            re = math.cos(ph0)  # phasor at block start, reset from the exact phase
            im = math.sin(ph0)
            for i in range(n):
                out[i] += (loud_a + dl * (i + 1.0)) * (ga + dg * (i + 1.0)) * re
                nre = re * c - im * s  # rotation: cos(θ+dθ)
                im = re * s + im * c
                re = nre
            phase[k] = (ph0 + dth * n) % _TWO_PI  # advance phase exactly, no drift across blocks
        return

    # ---- variable-frequency slow path (bend/slide) ----
    for i in range(n):
        frac = (i + 1.0) / n
        f0i = f0_a + (f0_b - f0_a) * frac
        loudi = loud_a + (loud_b - loud_a) * frac
        base = _TWO_PI * f0i * inv
        acc = 0.0
        for k in range(K):
            ga = gain_a[k]
            gb = gain_b[k]
            if ga == 0.0 and gb == 0.0:
                continue
            g0 = ga + (gb - ga) * frac  # linear interpolation within the block
            rk = ratio[k]
            ph = phase[k] + base * rk
            phase[k] = ph
            fk = rk * f0i
            if fk >= nyquist:
                g = 0.0
            elif fk > nyquist - ramp_hz:
                g = g0 * (nyquist - fk) / ramp_hz
            else:
                g = g0
            acc += g * math.cos(ph)
        out[i] += loudi * acc
    for k in range(K):
        phase[k] = phase[k] % _TWO_PI


def warmup():
    """Run the kernel once on dummy data to trigger JIT compilation and avoid a JIT xrun on the first note."""
    if not HAVE_NUMBA:
        return
    out = np.zeros(8, dtype=np.float64)
    phase = np.zeros(4, dtype=np.float64)
    ratio = np.arange(1, 5, dtype=np.float64)
    gain = np.ones(4, dtype=np.float64)
    voice_block_add(
        out, 8, phase, ratio, gain, gain, 4, 220.0, 220.0, 1.0, 1.0, 44100.0, 22050.0, 600.0
    )
