"""加法重合成单测 —— 阶段1 Day3 验收：分析->重合成 往返误差 + 无混叠。"""

import numpy as np

from analysis.harmonic import harmonics
from synth.additive import resynth

SR = 44100


def _make_harmonic_tone(f0, amps, dur=1.0, sr=SR, seed=1):
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=len(amps))
    y = sum(a * np.cos(2 * np.pi * (k + 1) * f0 * t + phases[k]) for k, a in enumerate(amps))
    return y.astype(np.float32)


def test_additive_rms_matches_theory():
    """直接喂已知 H：输出 RMS 应等于 sqrt(Σ A_k²/2)（验证加法数学）。"""
    f0 = 200.0
    amps = np.array([1.0, 0.5, 0.25, 0.125])
    T = 100
    H = np.tile(amps[:, None], (1, T)).astype(np.float32)
    y = resynth(H, f0, sr=SR, hop=512)
    rms = np.sqrt(np.mean(y[2000:-2000] ** 2))
    rms_theory = np.sqrt(np.sum(amps**2) / 2.0)
    err_db = 20 * np.log10(rms / rms_theory)
    assert abs(err_db) < 0.5, f"RMS 误差 {err_db:.3f} dB"


def test_analysis_resynth_roundtrip_within_1db():
    """完整往返：原音 -> harmonics -> resynth -> 再分析，逐谐波 < 1dB。"""
    f0 = 196.0  # G3
    amps = np.array([1.0, 0.7, 0.5, 0.35, 0.2, 0.12, 0.08])
    y = _make_harmonic_tone(f0, amps, dur=1.2)

    H, times, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))
    y_re = resynth(H, info["f0_per_frame"], sr=SR, hop=512, length=len(y))

    # 对重合成结果再做谐波分析，与原始幅度比对
    H2, _, _ = harmonics(y_re, f0, sr=SR, n_harmonics=len(amps))
    T2 = H2.shape[1]
    est = np.median(H2[:, int(T2 * 0.3) : int(T2 * 0.7)], axis=1)
    err_db = 20 * np.log10(est / amps)
    assert np.max(np.abs(err_db)) < 1.0, f"往返逐谐波误差(dB)={np.round(err_db, 3)}"


def test_resynth_no_aliasing_above_nyquist():
    """重合成不得在 Nyquist 以下产生折返峰（坑位：混叠验证）。

    取高 f0，使部分谐波超过 Nyquist；这些谐波应被丢弃且不折返。
    判据：FFT 中所有显著峰都落在某个真实谐波 k*f0 附近。
    """
    f0 = 2500.0  # 谐波 9 起(>22500Hz)越过 Nyquist
    amps = np.ones(12)
    T = 200
    H = np.tile(amps[:, None], (1, T)).astype(np.float32)
    y = resynth(H, f0, sr=SR, hop=512)
    y = y[1000:-1000]

    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    spec_db = 20 * np.log10(spec / (spec.max() + 1e-12) + 1e-12)

    # 真实谐波频率（仅 < Nyquist 的）
    valid_harm = [(k + 1) * f0 for k in range(12) if (k + 1) * f0 < SR / 2]
    peak_bins = np.where(spec_db > -40)[0]  # 显著峰（-40dB 以上）
    for b in peak_bins:
        nearest = min(abs(freqs[b] - h) for h in valid_harm)
        assert nearest < 60.0, (
            f"{freqs[b]:.1f}Hz 处出现非谐波峰（最近谐波 {nearest:.1f}Hz），疑似混叠折返"
        )


def test_resynth_output_finite_and_bounded():
    """重合成输出必须有限、有界（无 NaN/inf）。"""
    f0 = 261.63
    amps = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
    y = _make_harmonic_tone(f0, amps)
    H, _, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))
    y_re = resynth(H, info["f0_per_frame"], sr=SR)
    assert np.all(np.isfinite(y_re))
    assert np.max(np.abs(y_re)) < 100.0
