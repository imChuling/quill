"""滤波噪声合成：快照的噪声端。

SMS 随机模型（Serra & Smith 1990）的合成侧：把残差的 40 mel 带能量包络 `N_env[40, T]`
整形回宽带噪声。只依赖 numpy 和 synth.dsp（STFT/ISTFT/mel 与分析端共用同一份实现，
频带定义一致）。固定 seed 渲染，结果可复现。
"""

from __future__ import annotations

import numpy as np

from synth.dsp import (
    NOISE_HOP,
    NOISE_NFFT,
    istft_np,
    mel_filterbank,
    stft_np,
)

_NOISE_NFFT = NOISE_NFFT
_NOISE_HOP = NOISE_HOP


def synth_noise(
    N_env: np.ndarray,
    sr: int,
    length: int,
    n_fft: int = _NOISE_NFFT,
    hop: int = _NOISE_HOP,
    centers_hz=None,
    seed: int = 0,
) -> np.ndarray:
    """由 mel 带 RMS 包络 `N_env[n_mels, T0]` 合成 `length` 采样的整形噪声。

    分析-修改-合成法：取白噪声的 STFT 相位加目标幅度包络，再 ISTFT。幅度直接由 N_env 经
    mel 中心插值到线性 bin 得到，电平正确、无需标定常数。固定 seed 保证可复现。
    """
    N_env = np.asarray(N_env, dtype=np.float64)
    n_mels = N_env.shape[0]
    if centers_hz is None:
        _, centers_hz = mel_filterbank(sr, n_fft, n_mels)
    centers_hz = np.asarray(centers_hz, dtype=np.float64)
    bin_hz = np.arange(n_fft // 2 + 1) * (sr / n_fft)

    rng = np.random.default_rng(seed)
    w = rng.standard_normal(length + n_fft)
    Wc = stft_np(w, n_fft, hop)
    n_bins, Tn = Wc.shape

    # N_env 时间轴 (T0) -> 渲染帧数 (Tn)
    T0 = N_env.shape[1]
    if T0 == 1:
        N_t = np.repeat(N_env, Tn, axis=1)
    else:
        src = np.linspace(0.0, 1.0, T0)
        dst = np.linspace(0.0, 1.0, Tn)
        N_t = np.empty((n_mels, Tn))
        for b in range(n_mels):
            N_t[b] = np.interp(dst, src, N_env[b])

    # mel 带 -> 线性 bin 幅度（中心插值，平坦包络保持平坦）
    M = np.empty((n_bins, Tn))
    for t in range(Tn):
        M[:, t] = np.interp(bin_hz, centers_hz, N_t[:, t], left=N_t[0, t], right=N_t[-1, t])

    shaped = M * np.exp(1j * np.angle(Wc))
    y = istft_np(shaped, n_fft, hop, length=length)
    return y.astype(np.float32)
