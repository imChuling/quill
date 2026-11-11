"""SNT 残差再分解（analysis/snt.py）测试。"""

import numpy as np
import pytest

from analysis.snt import (
    energy_split,
    spectral_flatness,
    temporal_concentration,
    transient_stochastic_split,
)

SR = 44100


def _click_plus_noise(sr=SR, dur=1.0):
    """合成：t=0.3s 一个宽带瞬态 click + 全程低电平白噪底。"""
    n = int(sr * dur)
    rng = np.random.default_rng(0)
    x = 0.02 * rng.standard_normal(n).astype(np.float32)  # 稳态噪底
    c = int(0.3 * sr)
    x[c : c + 64] += np.hanning(64).astype(np.float32) * 0.8  # 瞬态爆发
    return x


def test_flatness_orders_noise_above_tone():
    sr = SR
    t = np.arange(sr) / sr
    rng = np.random.default_rng(1)
    white = rng.standard_normal(sr).astype(np.float32)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    assert spectral_flatness(white, sr) > 0.3  # 白噪平坦
    assert spectral_flatness(sine, sr) < 0.01  # 纯音尖峰
    assert spectral_flatness(white, sr) > spectral_flatness(sine, sr)


def test_split_separates_transient_from_floor():
    x = _click_plus_noise()
    trans, stoch = transient_stochastic_split(x, SR)
    # 瞬态在时间上更集中、随机底更弥散
    assert temporal_concentration(trans, SR) < temporal_concentration(stoch, SR)
    # click 时刻附近瞬态能量应高于其它处
    c = int(0.3 * SR)
    near = np.sum(trans[c - 256 : c + 256] ** 2)
    far = np.sum(trans[:512] ** 2)
    assert near > far


def test_energy_split_sums_sane():
    x = _click_plus_noise()
    trans, stoch = transient_stochastic_split(x, SR)
    sp = energy_split(x, trans, stoch)
    assert 0.0 <= sp["transient"] <= 1.5
    assert 0.0 <= sp["stochastic"] <= 1.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
