"""True Envelope（analysis/envelope.py）回归测试。

注意：True Envelope 实测**不优于**直接谐波幅度插值（见 docs/design_notes.md 负结果），
故不接入合成路径；这里只验证实现本身正确——包络压住谱、过共振峰、谷低于峰。
"""

import numpy as np
import pytest

from analysis.envelope import envelope_from_audio, sample_envelope_db, true_envelope

SR = 44100


def _formant_tone(f0=174.6, dur=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t)
    for k in range(1, 60):
        fk = k * f0
        if fk >= sr / 2:
            break
        # 双共振峰 700 / 1800 Hz + 本底
        a = np.exp(-(((fk - 700) / 350) ** 2)) + 0.6 * np.exp(-(((fk - 1800) / 600) ** 2)) + 0.03
        y += a * np.cos(2 * np.pi * fk * t)
    return (y / np.max(np.abs(y))).astype(np.float32)


def test_envelope_covers_spectrum():
    """True Envelope 应（近似）处处压在 log 幅度谱之上。"""
    rng = np.random.default_rng(0)
    logmag = -3 + 2 * np.sin(np.linspace(0, 8 * np.pi, 1025)) + 0.5 * rng.standard_normal(1025)
    env = true_envelope(logmag, order=40)
    # 允许收敛阈内的少量越界
    over = np.mean(logmag - env > 0.3)
    assert over < 0.05, f"包络未压住谱：{over:.2%} 处越界"


def test_envelope_tracks_formants():
    """共振峰处包络高于相邻谷。"""
    y = _formant_tone()
    freqs, env = envelope_from_audio(y, SR, 174.6)
    f1 = sample_envelope_db(freqs, env, [700])[0]  # 主共振峰
    valley = sample_envelope_db(freqs, env, [1200])[0]  # 谷
    high = sample_envelope_db(freqs, env, [3500])[0]  # 高频滚降
    assert f1 > valley > high, f"包络形状错: F1={f1:.2f} valley={valley:.2f} high={high:.2f}"


def test_sample_extrapolation_decays():
    """支撑外（超过 Nyquist 采样上限）应单调衰减，不爆。"""
    freqs = np.linspace(0, 8000, 200)
    env = np.linspace(0, -5, 200)
    a = sample_envelope_db(freqs, env, [7000, 12000, 20000])
    assert a[0] > a[1] > a[2] >= 0
    assert np.all(np.isfinite(a))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
