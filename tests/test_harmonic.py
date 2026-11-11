"""谐波提取单测 —— 阶段1 Day2 验收：合成已知谐波信号 -> 提取误差 < 1dB。"""

import numpy as np
import pytest

from analysis.harmonic import f0_scalar, harmonics, track_f0

SR = 44100


def _make_harmonic_tone(f0, amps, dur=1.0, sr=SR, phases=None):
    """合成 y = Σ amps[k] * cos(2π (k+1) f0 t + φ_k)。"""
    t = np.arange(int(sr * dur)) / sr
    if phases is None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, size=len(amps))
    y = np.zeros_like(t)
    for k, a in enumerate(amps):
        y += a * np.cos(2 * np.pi * (k + 1) * f0 * t + phases[k])
    return y.astype(np.float32)


def _extract_amps(H, lo=0.25, hi=0.75):
    """取中间帧（避开 center padding 边缘）的逐谐波幅度中位数。"""
    T = H.shape[1]
    sl = slice(int(T * lo), int(T * hi))
    return np.median(H[:, sl], axis=1)


@pytest.mark.parametrize(
    "f0",
    [
        215.33203125,  # = 20 * 44100/4096，谐波严格落 bin（扇贝损失为 0）
        233.7,  # 离 bin，检验抛物线插值校正
        146.83,  # D3，偏低
    ],
)
def test_harmonic_extraction_within_1db(f0):
    amps = np.array([1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.07, 0.05])
    y = _make_harmonic_tone(f0, amps)
    H, times, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))

    est = _extract_amps(H)
    err_db = 20 * np.log10(est / amps)
    assert np.max(np.abs(err_db)) < 1.0, (
        f"f0={f0}: 最大谐波误差 {np.max(np.abs(err_db)):.3f} dB >= 1dB\n"
        f"逐谐波误差(dB)={np.round(err_db, 3)}"
    )


def test_nyquist_truncation():
    """k*f0 越过 Nyquist 的谐波应被截断为 0（坑位）。"""
    f0 = 3000.0  # Nyquist=22050 -> 第 8 谐波 24000Hz 应被截断
    amps = np.ones(10)
    y = _make_harmonic_tone(f0, amps)
    H, _, _ = harmonics(y, f0, sr=SR, n_harmonics=10)
    est = _extract_amps(H)
    # 前 7 个谐波(<=21000Hz)应有能量；第 8 起(>=24000Hz)应为 0
    assert np.all(est[:7] > 0.5)
    assert np.all(est[7:] < 1e-3)


def test_low_f0_auto_bumps_nfft():
    """低音 f0 触发 n_fft 自动升到 8192（坑位：低音谐波间距不足）。"""
    f0 = 65.41  # C2，低于 100Hz 阈值
    amps = np.array([1.0, 0.7, 0.5, 0.3])
    y = _make_harmonic_tone(f0, amps, dur=1.5)
    H, _, info = harmonics(y, f0, sr=SR, n_harmonics=len(amps))
    assert info["n_fft"] == 8192
    est = _extract_amps(H)
    err_db = 20 * np.log10(est / amps)
    assert np.max(np.abs(err_db)) < 1.0, f"低音误差 {np.round(err_db, 3)} dB"


def test_quasi_harmonic_partial_measurement():
    """准谐波声源：放宽窗 + 记录实测部分音频率（酒杯/碗路径）。"""
    f0 = 300.0
    # 故意让高位分音偏离 k*f0（模拟酒杯部分音 stretch）
    partials = [300.0, 605.0, 915.0, 1230.0]  # 略高于 k*300
    t = np.arange(int(SR * 1.0)) / SR
    y = sum(a * np.cos(2 * np.pi * fp * t) for fp, a in zip(partials, [1.0, 0.6, 0.4, 0.3])).astype(
        np.float32
    )
    H, _, info = harmonics(y, f0, sr=SR, n_harmonics=4, measure_partials=True)
    pf = info["partial_freqs"]
    assert pf is not None
    # 实测频率应接近真实部分音（±15Hz）
    assert np.allclose(pf, partials, atol=15.0), f"实测部分音 {np.round(pf, 1)}"


def test_track_f0_on_steady_tone():
    """track_f0 在稳态正弦上应给出接近真值的中位 f0。"""
    f0_true = 220.0
    t = np.arange(int(SR * 1.0)) / SR
    y = (0.7 * np.sin(2 * np.pi * f0_true * t)).astype(np.float32)
    f0, conf, times = track_f0(y, sr=SR)
    med = f0_scalar(f0)
    cents = 1200 * np.log2(med / f0_true)
    assert abs(cents) < 50.0, f"f0 中位={med:.2f}Hz 偏离 {cents:+.1f} cents"
    assert len(f0) == len(conf) == len(times)
