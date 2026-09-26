"""加法重合成 —— FR-2 重建（阶段1 Day3）。

resynth(H, f0) -> y：相位累加振荡器组，帧间 H 与 f0 线性插值到采样率，
逐谐波天然带限（高于 Nyquist 的谐波置零，无需 BLEP/过采样——架构红利之一）。

引擎铁律 #12：本模块只依赖 numpy。这里是离线渲染版本（阶段3 的实时
逐 block 回调将共用同一加法合成数学）。
"""
from __future__ import annotations

import numpy as np

_SR_DEFAULT = 44100
_HOP_DEFAULT = 512


def resynth(H: np.ndarray, f0, sr: int = _SR_DEFAULT, hop: int = _HOP_DEFAULT,
            partial_freqs=None, f0_ref: float | None = None,
            length: int | None = None) -> np.ndarray:
    """由谐波幅度包络 H[K, T] 与基频 f0 重建波形。

    参数
    ----
    H : float32[K, T]   每谐波每帧的真实正弦幅度（harmonic.harmonics 输出）
    f0 : 标量 或 float[T]
        目标基频。标量 -> 恒定；数组（长度 T）-> 逐帧，线性插值到采样率。
    partial_freqs, f0_ref :
        准谐波声源：第 k 个分量频率 = partial_freqs[k] * (f0 / f0_ref)，
        否则严格 k * f0。
    length : int 或 None
        输出采样数；默认 T*hop。

    返回 float32[N]，相位连续、无帧边界咔哒。
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"H 应为 [K, T] 二维，收到 {H.shape}")
    K, T = H.shape
    N = int(length) if length is not None else T * hop

    # --- 帧级 f0 -> 采样级 f0（线性插值）---
    f0_arr = np.atleast_1d(np.asarray(f0, dtype=np.float64))
    if f0_arr.size == 1:
        f0_samp = np.full(N, f0_arr[0], dtype=np.float64)
    else:
        f0_samp = _frames_to_samples(f0_arr, T, N, hop)

    # --- 帧级 H -> 采样级幅度（逐谐波线性插值）---
    # 每谐波的频率比例（严格谐波 k 或 准谐波实测比例）
    if partial_freqs is not None:
        pf = np.asarray(partial_freqs, dtype=np.float64)[:K]
        ref = float(f0_ref) if f0_ref else 1.0
        ratio_k = pf / ref if ref else np.arange(1, K + 1, dtype=np.float64)
    else:
        ratio_k = np.arange(1, K + 1, dtype=np.float64)

    nyquist = sr / 2.0
    y = np.zeros(N, dtype=np.float64)

    for k in range(K):
        amp_frames = H[k]
        if not np.any(amp_frames):
            continue
        amp_samp = _frames_to_samples(amp_frames, T, N, hop)
        # 瞬时频率（逐样本）
        freq_samp = ratio_k[k] * f0_samp
        # 带限：高于 Nyquist 的样本幅度置零（阶段1 硬置零；阶段3 改增益斜坡）
        amp_samp = np.where(freq_samp < nyquist, amp_samp, 0.0)
        if not np.any(amp_samp):
            continue
        # 相位累加（连续相位，跨帧无跳变）
        phase = np.cumsum(2.0 * np.pi * freq_samp / sr)
        y += amp_samp * np.cos(phase)

    return y.astype(np.float32)


def _frames_to_samples(frame_vals: np.ndarray, T: int, N: int,
                       hop: int) -> np.ndarray:
    """把帧级序列线性插值/上采样到 N 个采样点。

    帧 i 的中心对应采样位置 i*hop（与 librosa center=True 的帧栅格一致到 hop 量级，
    阶段1 重建用此近似已足够；绝对相位由各谐波独立累加保证连续）。
    """
    frame_vals = np.asarray(frame_vals, dtype=np.float64)
    if T == 1:
        return np.full(N, frame_vals[0], dtype=np.float64)
    frame_pos = np.arange(T) * hop
    samp_idx = np.arange(N)
    return np.interp(samp_idx, frame_pos, frame_vals,
                     left=frame_vals[0], right=frame_vals[-1])


# --------------------------------------------------------------------------- #
# FR-3 共振峰保持的音高重定向（阶段2 Day2）
# --------------------------------------------------------------------------- #
def redirect_harmonics(H: np.ndarray, f0_ref: float, f0_tgt: float,
                       partial_freqs=None, mode: str = "formant",
                       rolloff_db_oct: float = 12.0):
    """把参考基频 f0_ref 的谐波包络 H[K,T] 重定向到目标基频 f0_tgt。

    mode="formant"（FR-3，正确版）：把 H[k] 视作**绝对频率** k·f0_ref 上的
        谱包络采样点；目标第 k' 个分量的幅度 = 在该绝对频率处对包络插值
        （True Envelope 简化, Röbel & Rodet 2005）。共振峰位置随频率轴不动，
        ±12 半音无花栗鼠效应。
    mode="index"（错误对照版，报告素材）：H_tgt[k']=H[k']，按谐波**序号**
        搬运 -> 共振峰随基频整体平移 -> 花栗鼠效应。

    返回 (H_tgt[K,T], partial_freqs_tgt 或 None)：
      * 严格谐波(partial_freqs=None)：H_tgt 落在 k'·f0_tgt，返回 (H_tgt, None)
      * 准谐波：返回 (H_tgt, partial_freqs)（频率仍由 resynth 按 pf*(f0_tgt/f0_ref) 发振）

    高音区坑位：目标频率超出包络支撑（> 最高参考分量）时按 rolloff_db_oct
    分贝/八度衰减外推，而非 np.interp 默认的平台钳制。
    """
    H = np.asarray(H, dtype=np.float64)
    K, T = H.shape

    if partial_freqs is not None:
        xp = np.asarray(partial_freqs, dtype=np.float64)[:K]          # 绝对参考频率
        targets = xp * (f0_tgt / f0_ref)                              # 目标分量频率
        pf_tgt = xp
    else:
        xp = np.arange(1, K + 1, dtype=np.float64) * f0_ref
        targets = np.arange(1, K + 1, dtype=np.float64) * f0_tgt
        pf_tgt = None

    if mode == "index":
        # 按序号搬运：幅度不变，仅频率轴整体缩放（resynth 用 f0_tgt 即可）
        return H.astype(np.float32), pf_tgt

    if mode != "formant":
        raise ValueError(f"未知 mode={mode!r}（应为 'formant' 或 'index'）")

    H_tgt = np.empty_like(H)
    for t in range(T):
        H_tgt[:, t] = _sample_envelope(targets, xp, H[:, t], rolloff_db_oct)
    return H_tgt.astype(np.float32), pf_tgt


def _sample_envelope(freqs_q, xp, fp, rolloff_db_oct):
    """在谱包络 (xp, fp) 上按绝对频率采样，支撑外按分贝/八度衰减外推。"""
    freqs_q = np.asarray(freqs_q, dtype=np.float64)
    out = np.interp(freqs_q, xp, fp, left=fp[0], right=fp[-1])
    # 高频外推：超出最高参考分量的查询点，按 rolloff 衰减最后一个值
    hi = xp[-1]
    above = freqs_q > hi
    if np.any(above) and fp[-1] > 0 and hi > 0:
        octaves = np.log2(np.maximum(freqs_q[above], hi) / hi)
        out[above] = fp[-1] * 10.0 ** (-rolloff_db_oct * octaves / 20.0)
    return out
