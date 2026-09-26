"""滤波噪声合成 + numpy STFT/ISTFT + mel 滤波器组 —— FR-2 噪声端（阶段2 Day1）。

SMS 随机模型（Serra & Smith 1990）的合成侧：把残差的 40 mel 带能量包络
`N_env[40, T]` 整形回宽带噪声。

引擎铁律 #12：本模块只依赖 numpy（STFT/ISTFT/mel 全部自实现，不碰 librosa），
是离线渲染与未来实时噪声端的共同基座；铁律 #6：固定 seed 渲染模式可复现。
"""
from __future__ import annotations

import numpy as np

_NOISE_NFFT = 1024     # 噪声端独立于谐波端的 STFT 规格
_NOISE_HOP = 256       # n_fft/4 -> Hann 满足 COLA
_MEL_FMIN = 40.0       # mel 下限：低于此的 mel 带会窄于 FFT bin 间距 ->
                       # 三角权重和≈0 -> band_power 除零爆裂（DC 泄漏）。从 40Hz 起。


# --------------------------------------------------------------------------- #
# numpy STFT / ISTFT（COLA 完美重建对）
# --------------------------------------------------------------------------- #
def _hann(n: int) -> np.ndarray:
    # 周期 Hann（与 librosa 默认一致），hop=n/4 时 WOLA 可完美重建
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float64)


def stft_np(x: np.ndarray, n_fft: int = _NOISE_NFFT,
            hop: int = _NOISE_HOP) -> np.ndarray:
    """实数信号 -> 复数 STFT [n_bins, T]（Hann 窗，无 center padding）。"""
    x = np.asarray(x, dtype=np.float64)
    w = _hann(n_fft)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for t in range(n_frames):
        seg = x[t * hop: t * hop + n_fft] * w
        out[:, t] = np.fft.rfft(seg)
    return out


def istft_np(spec: np.ndarray, n_fft: int = _NOISE_NFFT,
             hop: int = _NOISE_HOP, length: int | None = None) -> np.ndarray:
    """复数 STFT -> 实数信号（WOLA，归一化保证 istft(stft(x))==x）。"""
    w = _hann(n_fft)
    n_frames = spec.shape[1]
    total = (n_frames - 1) * hop + n_fft
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    w2 = w * w
    for t in range(n_frames):
        fr = np.fft.irfft(spec[:, t], n=n_fft)
        pos = t * hop
        out[pos:pos + n_fft] += fr * w
        norm[pos:pos + n_fft] += w2
    norm = np.where(norm > 1e-9, norm, 1.0)
    out = out / norm
    if length is not None:
        out = (out[:length] if length <= len(out)
               else np.pad(out, (0, length - len(out))))
    return out.astype(np.float64)


# --------------------------------------------------------------------------- #
# mel 滤波器组（numpy 实现，分析/合成共用 -> 频带定义一致）
# --------------------------------------------------------------------------- #
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40,
                   fmin: float = _MEL_FMIN, fmax: float | None = None):
    """返回 (fb[n_mels, n_bins], centers_hz[n_mels])，三角滤波器。

    fmin 默认 40Hz：避免低频 mel 带窄于 bin 间距导致权重和≈0、band RMS 除零爆裂。
    """
    fmax = float(fmax if fmax is not None else sr / 2.0)
    n_bins = n_fft // 2 + 1
    bin_hz = np.arange(n_bins) * (sr / n_fft)
    m_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    f_edges = mel_to_hz(m_edges)                     # n_mels+2 边界
    centers = f_edges[1:-1]
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for b in range(n_mels):
        lo, ce, hi = f_edges[b], f_edges[b + 1], f_edges[b + 2]
        left = (bin_hz - lo) / max(ce - lo, 1e-9)
        right = (hi - bin_hz) / max(hi - ce, 1e-9)
        tri = np.maximum(0.0, np.minimum(left, right))
        fb[b] = tri
    return fb, centers.astype(np.float64)


# --------------------------------------------------------------------------- #
# 噪声整形合成
# --------------------------------------------------------------------------- #
def synth_noise(N_env: np.ndarray, sr: int, length: int,
                n_fft: int = _NOISE_NFFT, hop: int = _NOISE_HOP,
                centers_hz=None, seed: int = 0) -> np.ndarray:
    """由 mel 带 RMS 包络 `N_env[n_mels, T0]` 合成 `length` 采样的整形噪声。

    分析-修改-合成法：取白噪声的 STFT 相位 + 目标幅度包络 -> ISTFT。
    幅度直接由 N_env 经 mel 中心插值到线性 bin 得到，电平正确、无需标定常数。
    固定 seed 保证可复现（铁律 #6）。
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
        M[:, t] = np.interp(bin_hz, centers_hz, N_t[:, t],
                            left=N_t[0, t], right=N_t[-1, t])

    shaped = M * np.exp(1j * np.angle(Wc))
    y = istft_np(shaped, n_fft, hop, length=length)
    return y.astype(np.float32)
