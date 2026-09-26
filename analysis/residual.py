"""残差提取 + 噪声包络 + attack 残差 —— FR-2 完成（阶段2 Day1）。

SMS 随机模型（Serra & Smith 1990）的分析侧：
  res = y − 谐波模型（相位对齐减法，坑位要求）
然后把残差整形为 40 mel 带 RMS 包络 `N_env[40, T]`（喂 synth.noise）。

相位对齐减法（坑位）：不在时域做 y−resynth（相位不匹配无法对消），
而在 STFT 复数域逐帧把每个谐波的**主瓣幅度**从 |D| 中减去、保留原相位，
ISTFT 得残差——既对消谐波，又把谐波位置的底噪相位留给残差。
"""
from __future__ import annotations

import numpy as np

from synth.noise import stft_np, mel_filterbank, _NOISE_NFFT, _NOISE_HOP

try:
    from quill_config import CFG
    _HCFG = CFG["harmonic"]
    _SR = int(CFG["audio"]["sr"])
except Exception:  # pragma: no cover
    _SR = 44100
    _HCFG = {"n_fft": 4096, "hop": 512, "nyquist_margin_hz": 50.0}

_ATTACK_MS = 60.0          # Part B: attack_res = 60ms@44.1k = 2646 样本
_LOBE_HALFWIDTH_BINS = 6   # Hann 主瓣减法的半窗（bin）


def _window_lobe_lut(n_fft: int, halfwidth_bins: int, oversample: int = 64):
    """预算 Hann 窗的 DTFT 主瓣 |W(δ)|（δ 以 bin 为单位），DC 归一到相干增益 Σwin。

    返回 (offsets[bin], wmag[bin])，用于把分数频率峰的主瓣形状采样出来。
    """
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
    Npad = n_fft * oversample
    W = np.abs(np.fft.rfft(win, n=Npad))      # W[0] = Σwin = 相干增益
    # 第 j 个 pad-bin 对应原始 bin 偏移 j/oversample
    max_j = halfwidth_bins * oversample
    offsets = np.arange(max_j + 1) / oversample
    return offsets, W[:max_j + 1]


def extract_residual(y, H, f0_frame, sr: int = _SR, n_fft: int | None = None,
                     hop: int | None = None, partial_freqs=None,
                     f0_ref: float | None = None,
                     lobe_halfwidth_bins: int = _LOBE_HALFWIDTH_BINS):
    """相位对齐减法得残差 res（时域，长度同 y）。

    参数与 harmonic.harmonics 对齐：H[K, T] 为谐波幅度（真实正弦幅度），
    f0_frame[T] 为逐 STFT 帧基频；partial_freqs 给定则按准谐波频率定位主瓣。
    """
    import librosa

    y = np.asarray(y, dtype=np.float64)
    n_fft = int(n_fft if n_fft is not None else _HCFG["n_fft"])
    hop = int(hop if hop is not None else _HCFG["hop"])
    H = np.asarray(H, dtype=np.float64)
    K, T = H.shape

    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
    cg = win.sum()
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(D)
    phase = np.angle(D)
    n_bins, Td = mag.shape
    T = min(T, Td)
    bin_hz = sr / n_fft
    nyquist = sr / 2.0
    nyq_margin = float(_HCFG.get("nyquist_margin_hz", 50.0))

    offs, wlut = _window_lobe_lut(n_fft, lobe_halfwidth_bins)

    f0_frame = np.asarray(f0_frame, dtype=np.float64)
    if partial_freqs is not None:
        pf = np.asarray(partial_freqs, dtype=np.float64)[:K]
        ref = float(f0_ref) if f0_ref else float(np.nanmedian(f0_frame))
        ratio_k = pf / ref
    else:
        ratio_k = np.arange(1, K + 1, dtype=np.float64)

    harm_mag = np.zeros_like(mag)
    for t in range(T):
        f0_t = f0_frame[t] if t < len(f0_frame) else f0_frame[-1]
        if not (np.isfinite(f0_t) and f0_t > 0):
            continue
        for k in range(K):
            a = H[k, t]
            if a <= 0:
                continue
            target = ratio_k[k] * f0_t
            if target >= nyquist - nyq_margin:
                break
            p = target / bin_hz                       # 分数 bin
            peak = 0.5 * a * cg                        # |D| 峰值（A=2peak/cg）
            lo = int(np.floor(p - lobe_halfwidth_bins))
            hi = int(np.ceil(p + lobe_halfwidth_bins))
            for m in range(max(0, lo), min(n_bins, hi + 1)):
                d = abs(m - p)
                if d <= offs[-1]:
                    harm_mag[m, t] += peak * (np.interp(d, offs, wlut) / cg)

    res_mag = np.maximum(mag - harm_mag, 0.0)
    R = res_mag * np.exp(1j * phase)
    res = librosa.istft(R, hop_length=hop, n_fft=n_fft, window="hann",
                        length=len(y))
    return res.astype(np.float32)


def noise_envelope(res, sr: int = _SR, n_mels: int = 40,
                   n_fft: int = _NOISE_NFFT, hop: int = _NOISE_HOP,
                   guard_samples: int = 4096):
    """残差 -> mel 带 RMS 包络 N_env[n_mels, T]（与 synth.noise 同框架）。

    坑位（边界伪影）：librosa center=True 的 istft 在首尾各约一个谐波窗长内
    会留下大瞬态（谐波模型在信号起止处不匹配），这些边界帧会以 ~100× 主导 N_env。
    对策：① 仅在去掉首尾 guard 的**稳态内区**估包络；② 每带按 中位数+k·MAD 鲁棒封顶。
    """
    res = np.asarray(res, dtype=np.float64)
    g = int(min(guard_samples, max(0, (len(res) - n_fft) // 2)))
    interior = res[g:len(res) - g] if (len(res) - 2 * g) >= n_fft else res

    R = np.abs(stft_np(interior, n_fft, hop))
    power = R ** 2
    fb, centers = mel_filterbank(sr, n_fft, n_mels)
    denom = fb.sum(axis=1, keepdims=True)
    denom = np.where(denom > 1e-9, denom, 1.0)
    band_power = (fb @ power) / denom              # [n_mels, T] 加权平均功率
    N_env = np.sqrt(np.maximum(band_power, 0.0))

    # 鲁棒封顶：单帧瞬态不得主导（中位数 + 8·MAD）
    med = np.median(N_env, axis=1, keepdims=True)
    mad = np.median(np.abs(N_env - med), axis=1, keepdims=True)
    cap = med + 8.0 * 1.4826 * mad
    N_env = np.minimum(N_env, cap)
    return N_env.astype(np.float32), centers


def attack_residual(res, sr: int = _SR, ms: float = _ATTACK_MS,
                    onset_thresh: float = 0.05):
    """截取起音残差 60ms（Part B: float32[2646]@44.1k）。

    从首个超过 onset_thresh*peak 的样本起取 `ms` 毫秒，长度不足右补零。
    """
    res = np.asarray(res, dtype=np.float32)
    n = int(round(ms / 1000.0 * sr))
    if res.size == 0:
        return np.zeros(n, dtype=np.float32)
    peak = float(np.max(np.abs(res)))
    if peak > 0:
        above = np.where(np.abs(res) >= onset_thresh * peak)[0]
        start = int(above[0]) if above.size else 0
    else:
        start = 0
    seg = res[start:start + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    seg = seg.astype(np.float32).copy()
    # 颗粒首尾 2ms 升余弦淡入淡出，避免回放时块边界咔哒（铁律 #4）
    fade = max(1, int(0.002 * sr))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade)))
    seg[:fade] *= ramp
    seg[-fade:] *= ramp[::-1]
    return seg
