"""离线渲染：TimbreSnapshot -> 波形 —— FR-2/FR-3（阶段2 Day3）。

把音色快照在任意目标基频上重合成（共振峰保持），叠加噪声端与起音残差。
引擎铁律 #12：只依赖 numpy（经 synth.additive / synth.noise）。这条渲染路径
与阶段3 实时回调将共用同一加法+噪声数学。
"""
from __future__ import annotations

import numpy as np

from synth.additive import resynth, redirect_harmonics
from synth.noise import synth_noise


def render(snap: dict, f0_tgt: float | None = None, *, dur_s: float | None = None,
           length: int | None = None, mode: str = "formant",
           add_attack: bool = True, noise_gain: float = 1.0,
           seed: int = 0, env_correction=None) -> np.ndarray:
    """渲染快照到 `f0_tgt`（默认 = f0_ref）。

    mode="formant"：FR-3 共振峰保持（正确版）；mode="index"：按序号搬运（花栗鼠对照版）。
    env_correction=(grid_hz, res_log)：**target 侧**包络校正(Atlas 残差;Sprint 2)——
        重定向后在各分量**目标绝对频率**处乘 exp(interp(f, grid, res))。None=不校正,
        输出与既有路径逐位一致(α=0 回退)。此入口即引擎未来消费 Atlas 的位置
        (note_on 的 gain_k 同一数学,离线先验证)。
    """
    meta = snap.get("meta", {})
    sr = int(meta.get("sr", 44100))
    harm_hop = int(meta.get("harm_hop", 512))
    f0_ref = float(snap["f0_ref"])
    f0_tgt = float(f0_tgt) if f0_tgt is not None else f0_ref

    H = np.asarray(snap["H_env"], dtype=np.float64)
    pf = snap.get("partial_freqs")
    T = H.shape[1]

    if length is None:
        length = int(dur_s * sr) if dur_s else T * harm_hop

    # FR-3：谐波端重定向
    H_tgt, pf_tgt = redirect_harmonics(H, f0_ref, f0_tgt, partial_freqs=pf,
                                       mode=mode)
    if env_correction is not None:
        grid_hz, res_log = env_correction
        K = H_tgt.shape[0]
        if pf is not None:
            targets = np.asarray(pf, np.float64)[:K] * (f0_tgt / f0_ref)
        else:
            targets = np.arange(1, K + 1, dtype=np.float64) * f0_tgt
        fac = np.exp(np.interp(targets, np.asarray(grid_hz, np.float64),
                               np.asarray(res_log, np.float64)))
        H_tgt = H_tgt * fac[:, None]
    y_harm = resynth(H_tgt, f0_tgt, sr=sr, hop=harm_hop,
                     partial_freqs=pf_tgt, f0_ref=f0_ref, length=length)

    # 噪声端（不随音高搬移，宽带纹理）
    y = y_harm.astype(np.float64)
    N_env = snap.get("N_env")
    if N_env is not None and np.asarray(N_env).size and noise_gain > 0:
        centers = snap.get("noise_centers_hz")
        n_fft = int(meta.get("noise_nfft", 1024))
        hop = int(meta.get("noise_hop", 256))
        y_noise = synth_noise(np.asarray(N_env), sr, length, n_fft=n_fft,
                              hop=hop, centers_hz=centers, seed=seed)
        y[:len(y_noise)] += noise_gain * y_noise[:length]

    # 起音残差（铁律：note-on 颗粒；此处离线叠加在起点）
    if add_attack:
        atk = np.asarray(snap.get("attack_res", []), dtype=np.float64)
        if atk.size:
            m = min(len(atk), length)
            y[:m] += atk[:m]

    # 输出首尾 5ms 等功率淡变，避免离线渲染的起止咔哒
    fade = max(1, min(int(0.005 * sr), length // 2))
    ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return y.astype(np.float32)
