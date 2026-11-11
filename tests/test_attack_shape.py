"""拨片厚度 / 起音整形（synth/attack_shape.py）测试。"""

import numpy as np
import pytest

from synth.attack_shape import apply_pick, onset_fade, shape_attack

SR = 44100


def _grain(n=2646):
    rng = np.random.default_rng(0)
    g = rng.standard_normal(n).astype(np.float32)
    g *= np.exp(-np.linspace(0, 6, n))  # 衰减瞬态
    return g


def test_pick0_silent():
    assert np.allclose(shape_attack(_grain(), 0.0), 0.0)


def test_pick1_identity():
    g = _grain()
    assert np.allclose(shape_attack(g, 1.0), g, atol=1e-5)


def _hf_energy(x, sr=SR, fmin=4000.0):
    X = np.abs(np.fft.rfft(np.asarray(x, np.float64)))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float((X[f >= fmin] ** 2).sum())


def test_harder_pick_brighter():
    """pick 越大，高频能量越多（更亮/更尖）。"""
    g = _grain()
    e = [_hf_energy(shape_attack(g, p)) for p in (0.2, 0.5, 1.0)]
    assert e[0] < e[1] < e[2]


def test_onset_fade_length():
    assert onset_fade(1000, 1.0, SR) is None  # 硬拨片无 fade
    f_soft = onset_fade(1000, 0.1, SR)
    f_med = onset_fade(1000, 0.6, SR)
    assert f_soft is not None and len(f_soft) > len(f_med)  # 越软 fade 越长


def test_apply_pick_finite_and_shaped():
    body = 0.3 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR).astype(np.float32)
    y_soft = apply_pick(body, _grain(), 0.15, SR)
    y_hard = apply_pick(body, _grain(), 1.0, SR)
    assert np.isfinite(y_soft).all() and len(y_soft) == len(body)
    # 硬拨片起音瞬时能量应高于软拨片
    assert np.max(np.abs(y_hard[:1000])) > np.max(np.abs(y_soft[:1000]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
