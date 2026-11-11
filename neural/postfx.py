"""神经渲染后处理:谱倾斜 + 算法混响。桌面版与网页版共用,保证同一旋钮同一声音。"""

from __future__ import annotations

import numpy as np


def apply_reverb(y: np.ndarray, sr: int, wet: float) -> np.ndarray:
    """Simple algorithmic reverb via exponentially decaying comb filters."""
    from scipy.signal import fftconvolve

    wet = float(wet)
    if wet <= 0.01:
        return y
    rt60 = 1.2 + wet * 1.8
    ir_len = int(sr * rt60)
    ir = np.zeros(ir_len, np.float64)
    ir[0] = 1.0
    for delay_ms in [29, 37, 43, 53, 67, 79]:
        d = int(delay_ms * sr / 1000)
        n_echoes = ir_len // d
        for k in range(1, n_echoes):
            ir[k * d] += 0.6**k * (0.7 + 0.3 * np.sin(k * 0.7))
    ir /= np.max(np.abs(ir)) + 1e-12
    reverb = fftconvolve(y, ir, mode="full")[: len(y)]
    return y * (1.0 - wet) + reverb * wet


def apply_tilt(y: np.ndarray, sr: int, brightness: float) -> np.ndarray:
    """一阶谱倾斜(brightness 0.5=直通,>0.5 提高频 <0.5 提低频)。"""
    from scipy.signal import butter, sosfilt

    tilt = (float(brightness) - 0.5) * 2.0
    if abs(tilt) < 0.02:
        return y
    sos = butter(1, 1200.0 / (sr / 2.0), btype="high", output="sos")
    hi = sosfilt(sos, y)
    return y + tilt * 0.85 * (hi if tilt > 0 else (y - hi))
