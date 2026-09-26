"""取色前置自动降噪(P3+,7.3 用户需求:"用户给的通常不是特别清晰")。

谱门控(spectral gating,noisereduce 思路,纯 numpy/scipy 无新依赖):
  1. 噪声画像 = 每频点最安静 15% 帧的中位数(录音里总有相对安静的瞬间);
  2. 软掩码 = 幅度超出画像门限的程度(sigmoid 软过渡,避免音乐噪声"水泡声");
  3. 掩码时频平滑后乘回复数谱,iSTFT 还原。
在 analyze_timbre 之前清理 → f0 跟踪/谐波提取/噪声层全部受益。
strength=0 直通;1=标准;2=激进。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft


def spectral_gate(y: np.ndarray, sr: int, strength: float = 1.0,
                  n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """谱门控降噪。返回与输入等长的 float32。"""
    if strength <= 0 or len(y) < n_fft * 2:
        return np.asarray(y, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    f, t, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, padded=True)
    mag = np.abs(Z)                                        # [F, T]
    # 无可学习底噪 → 直通(7.4 UAT 实锤):整段连续发声时"最安静帧"仍是信号本身,
    # 门限会压真实谐波(双簧管谐波能量被腰斩,取色与输入无关)。判据:最安静 15% 帧
    # 的电平未显著低于典型帧(<-10dB)=没有真底噪;平稳持续音的噪声本被信号掩蔽,无需降噪。
    frame_rms = np.sqrt((mag ** 2).mean(axis=0)) + 1e-12
    kq = max(2, int(len(frame_rms) * 0.15))
    floor_lvl = float(np.median(np.sort(frame_rms)[:kq]))
    if floor_lvl > 0.3 * float(np.median(frame_rms)):
        return np.asarray(y, dtype=np.float32)
    # 噪声画像:每频点最安静 15% 帧的中位数(鲁棒;整段有声也能取到相对底部)
    k = max(2, int(mag.shape[1] * 0.15))
    quiet = np.sort(mag, axis=1)[:, :k]
    prof = np.median(quiet, axis=1, keepdims=True) + 1e-12
    thr = prof * (1.2 + 1.3 * strength)                    # 门限随强度上移
    # 软掩码:门限处 0.5,±soft 区间平滑过渡(dB 域 sigmoid)
    ratio_db = 20.0 * np.log10(np.maximum(mag, 1e-12) / thr)
    mask = 1.0 / (1.0 + np.exp(-ratio_db / 3.0))
    # 时频平滑(3x3 盒式)防孤立时频点开关=音乐噪声
    m = mask
    m = (m + np.roll(m, 1, 1) + np.roll(m, -1, 1)) / 3.0
    m = (m + np.roll(m, 1, 0) + np.roll(m, -1, 0)) / 3.0
    floor = 10.0 ** (-(12.0 + 12.0 * strength) / 20.0)     # 残余底(全零掩码听感发"真空")
    m = floor + (1.0 - floor) * m
    _, y_out = istft(Z * m, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    y_out = y_out[:len(y)]
    if len(y_out) < len(y):
        y_out = np.pad(y_out, (0, len(y) - len(y_out)))
    return y_out.astype(np.float32)
