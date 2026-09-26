"""SNT 残差深化 —— 把 SMS 残差再分解为「瞬态 + 随机」(FR-2 深化)。

SMS（Serra-Smith 1990）只把信号分成谐波 + 随机两层；起音的宽带瞬态（拨片/唇齿/弓
毛触发的"咔"）其实既不谐波、也不平稳，硬塞进随机层会污染噪声底（听感发"嗡"）。

这里对**谐波减除后的残差**再做一次中值滤波分离（Fitzgerald 2010 / Driedger HPSS）：
  - 沿时间中值 -> 时间平稳成分 = **随机噪声底**（stochastic）
  - 沿频率中值 -> 宽带时间局部成分 = **瞬态**（transient）
得到三层 S(正弦) + N(随机) + T(瞬态)。本模块不改默认合成路径（人声已调好，避免回归），
而是：① 提供干净分离工具；② 给 eval/residual_quality.py 量化「残差确实像噪声」以证 SMS 假设。
"""
from __future__ import annotations

import numpy as np

try:
    from quill_config import CFG
    _SR = int(CFG["audio"]["sr"])
except Exception:  # pragma: no cover
    _SR = 44100

_NFFT = 1024
_HOP = 256


def transient_stochastic_split(res, sr: int = _SR, n_fft: int = _NFFT,
                               hop: int = _HOP, margin: float = 2.0):
    """残差 -> (transient, stochastic)，时域，长度同 res。

    margin>1 给更"干净"（更稀疏）的分离：两掩码间留软间隔，能量不重复计入。
    """
    import librosa
    res = np.asarray(res, dtype=np.float32)
    if res.size < n_fft:
        res = np.pad(res, (0, n_fft - res.size))
    D = librosa.stft(res, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    # HPSS：harmonic=时间平稳=随机底；percussive=宽带瞬态
    D_stoch, D_trans = librosa.decompose.hpss(D, margin=(margin, margin))
    stoch = librosa.istft(D_stoch, hop_length=hop, n_fft=n_fft, length=len(res))
    trans = librosa.istft(D_trans, hop_length=hop, n_fft=n_fft, length=len(res))
    return trans.astype(np.float32), stoch.astype(np.float32)


def spectral_flatness(x, sr: int = _SR, n_fft: int = 2048,
                      fmin: float = 200.0, fmax: float = 8000.0):
    """Wiener entropy / 谱平坦度 ∈ (0,1]：1=白噪(平坦)，→0=纯音(尖峰)。

    噪声底应高平坦；谐波成分应低平坦。**仅在 [fmin,fmax] 信号带内**算（带外空 bin
    会把几何均值拉塌），floor 用每帧峰值的相对下限（-80dB），避免空 bin 主导。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (f >= fmin) & (f <= fmax)
    win = np.hanning(n_fft)
    flats = []
    for i in range(0, len(x) - n_fft, n_fft // 2):
        p = np.abs(np.fft.rfft(x[i:i + n_fft] * win)) ** 2
        p = p[band]
        pk = p.max()
        if pk <= 0:
            continue
        p = np.maximum(p, pk * 1e-8)        # 相对 floor（-80dB），稳住几何均值
        gm = np.exp(np.mean(np.log(p)))
        am = np.mean(p)
        flats.append(gm / am)
    return float(np.mean(flats)) if flats else 0.0


def temporal_concentration(x, sr: int = _SR, frac: float = 0.9):
    """时间集中度：包络能量前 `frac` 集中在多少比例的时长内（越小越"瞬态/局部"）。"""
    x = np.asarray(x, dtype=np.float64)
    e = x ** 2
    tot = e.sum()
    if tot <= 0:
        return 1.0
    order = np.sort(e)[::-1]               # 能量从大到小
    cum = np.cumsum(order) / tot
    n_needed = int(np.searchsorted(cum, frac)) + 1
    return n_needed / len(e)               # 占总样本比例


def energy_split(y, transient, stochastic, harmonic_res=None):
    """返回各层能量占比 dict（相对输入 y 的能量）。"""
    ey = float(np.sum(np.asarray(y, np.float64) ** 2)) + 1e-12
    et = float(np.sum(np.asarray(transient, np.float64) ** 2))
    es = float(np.sum(np.asarray(stochastic, np.float64) ** 2))
    out = {"transient": et / ey, "stochastic": es / ey}
    if harmonic_res is not None:
        out["residual"] = (et + es) / ey
        out["harmonic"] = 1.0 - (et + es) / ey
    return out
