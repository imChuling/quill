"""True Envelope 谱包络估计 —— FR-3 探索（Röbel & Rodet 2005）。

倒谱迭代真包络：反复"取谱与当前包络的逐点最大 → 倒谱低通平滑"，直到包络处处压在谱
峰之上，给出连续、平滑、穿过谐波峰的谱包络（共振峰结构）。

**实测结论（诚实负结果，见 docs/design_notes.md）**：在跨音高保真上，True Envelope **不
如**直接对实测谐波幅度做线性插值（人声 -4 半音谱距 2.83 vs 1.13）——平滑包络丢掉了对匹配
真实谱有用的谐波级细节。故**不接入合成路径**，本模块保留为已测工具（可用于谱包络可视化 /
未来 L2 / 噪声端）。只依赖 numpy。
"""
from __future__ import annotations

import numpy as np


def true_envelope(logmag, order, n_iter=200, thresh_db=2.0):
    """对 log 幅度谱 `logmag`（rfft bins）估计 True Envelope（同长度 log 包络）。

    order: 倒谱阶（保留的低 quefrency 系数数；越小越平滑）。经验 order≈sr/(2*f0) 量级，
           取略小于基频周期对应的 quefrency，使包络平滑过谐波但保住共振峰。
    """
    logmag = np.asarray(logmag, dtype=np.float64)
    B = len(logmag)
    A = logmag.copy()
    V = np.full(B, -60.0)                       # 初始包络（dB 量级下限）
    thresh = thresh_db / (20.0 / np.log(10.0))  # dB 阈 -> 自然 log 阈
    for _ in range(n_iter):
        A = np.maximum(A, V)                    # 包络须压在谱之上
        full = np.concatenate([A, A[-2:0:-1]])  # 对称延拓到全 DFT，长度 M=2(B-1)
        cep = np.fft.ifft(full).real            # 实倒谱（偶对称）
        M = len(cep)
        lifter = np.zeros(M)                    # 低通 lifter：保留低 quefrency（两端因对称）
        lifter[:order + 1] = 1.0
        if order > 0:
            lifter[M - order:] = 1.0
        Vfull = np.fft.fft(cep * lifter).real
        V = Vfull[:B]
        if np.max(A - V) < thresh:              # 收敛：包络处处压住谱
            break
    return V


def envelope_from_audio(y, sr, f0_ref, n_fft=4096, order=None, fmax=None):
    """从稳态音频估计 True Envelope，返回 (freqs, log_env)（用于按绝对频率采样）。"""
    y = np.asarray(y, dtype=np.float64)
    seg = y[len(y) // 5: -len(y) // 5] if len(y) > 10 else y
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    # 稳态平均幅度谱
    S = []
    for i in range(0, len(seg) - n_fft, n_fft // 2):
        S.append(np.abs(np.fft.rfft(seg[i:i + n_fft] * np.hanning(n_fft))))
    mag = np.mean(S, axis=0) if S else np.abs(np.fft.rfft(seg[:n_fft] * np.hanning(n_fft)))
    logmag = np.log(mag + 1e-9)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    if order is None:
        # 略小于基频周期对应 quefrency，包络平滑过谐波又保共振峰
        order = max(8, int(0.7 * sr / max(f0_ref, 1.0)))
    env = true_envelope(logmag, order=order)
    return freqs, env


def sample_envelope_db(freqs, log_env, query_hz, rolloff_db_oct=12.0):
    """在 (freqs, log_env) 上按绝对频率采样 -> 线性幅度；支撑外按 rolloff 衰减外推。"""
    query_hz = np.asarray(query_hz, dtype=np.float64)
    v = np.interp(query_hz, freqs, log_env, left=log_env[0], right=log_env[-1])
    hi = freqs[-1]
    above = query_hz > hi
    if np.any(above) and hi > 0:
        octv = np.log2(np.maximum(query_hz[above], hi) / hi)
        v[above] = log_env[-1] - (rolloff_db_oct * octv) * (np.log(10.0) / 20.0)
    return np.exp(v)
