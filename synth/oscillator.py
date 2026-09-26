"""实时加法振荡器内核 —— FR-4 热路径（阶段3）。

`@njit` 融合内循环：逐样本推进每谐波相位并累加到输出块，块内 f0/loudness 线性斜坡，
谐波越 Nyquist 用增益软淡出（不硬切，坑位）。块尾相位取模防大数 cos 精度损失。

引擎铁律 #12：只依赖 numpy + numba。零分配：调用方传入预分配的 out / phase。
numba 不可用时退化为同名纯 Python 函数（慢，仅保证可运行/可测）。
"""
from __future__ import annotations

import math

import numpy as np

try:
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
def voice_block_add(out, n, phase, ratio, gain_a, gain_b, K,
                    f0_a, f0_b, loud_a, loud_b, sr, nyquist, ramp_hz):
    """把一个声部的 n 个采样的谐波贡献**累加**进 out[:n]。

    out      : float64[>=n]  主混音块（原地累加）
    phase    : float64[K]    持久相位（原地更新，note-on 由调用方清零）
    ratio    : float64[K]    第 k 分量频率比例（严格谐波=k，准谐波=pf/f0_ref）
    gain_a/b : float64[K]    块首/块尾每谐波幅度，**块内线性渐变**（2026-08-01）。
               此前 gain 在块内恒定：起音谱轨迹/暗化项让它快速变化时，每个块边界
               都是一次阶跃 → 实测块边界跳变是块内的 4.5 倍 → 可听爆破/断续
               （用户耳测"像要爆炸"）。逐样本渐变后与 loudness/f0 同规范。
    f0_a/f0_b: 块首/块尾基频（块内线性斜坡，防 f0 跳变拉链噪声）
    loud_a/b : 块首/块尾总响度（ADSR×力度×淡入，线性斜坡）

    恒频块走相量递推快路径（内循环无 cos，只复数旋转）；bend/slide 的变频块走
    逐样本 cos 慢路径。两路都跳过 gain==0（剪枝）与越 Nyquist 的谐波。
    """
    inv = 1.0 / sr
    dl = (loud_b - loud_a) / n

    if f0_a == f0_b:                              # ---- 恒频快路径 ----
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
                nq = (nyquist - fk) / ramp_hz       # Nyquist 软淡出（坑位）
                ga = ga * nq
                gb = gb * nq
            dg = (gb - ga) / n
            dth = base * rk
            c = math.cos(dth)
            s = math.sin(dth)
            ph0 = phase[k]
            re = math.cos(ph0)                      # 块首相量（由精确相位重置）
            im = math.sin(ph0)
            for i in range(n):
                out[i] += (loud_a + dl * (i + 1.0)) * (ga + dg * (i + 1.0)) * re
                nre = re * c - im * s               # 旋转：cos(θ+dθ)
                im = re * s + im * c
                re = nre
            phase[k] = (ph0 + dth * n) % _TWO_PI    # 相位精确推进（无跨块漂移）
        return

    # ---- 变频慢路径（bend/slide）----
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
            g0 = ga + (gb - ga) * frac          # 块内线性渐变
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
    """空跑一遍内核触发 JIT 编译（防首音 JIT xrun，Day3 要求）。"""
    if not HAVE_NUMBA:
        return
    out = np.zeros(8, dtype=np.float64)
    phase = np.zeros(4, dtype=np.float64)
    ratio = np.arange(1, 5, dtype=np.float64)
    gain = np.ones(4, dtype=np.float64)
    voice_block_add(out, 8, phase, ratio, gain, gain, 4,
                    220.0, 220.0, 1.0, 1.0, 44100.0, 22050.0, 600.0)
