"""synth/oscillator.py unit tests: correctness of the voice_block_add additive synthesis kernel."""

import math

import numpy as np
import pytest

from synth.oscillator import voice_block_add
from tests.helpers import SR as _SR

SR = float(_SR)
NYQ = SR / 2.0
RAMP_HZ = 600.0


def _run_kernel(f0, K=4, n=2048, gain=None, f0_b=None, loud_a=1.0, loud_b=1.0):
    out = np.zeros(n, dtype=np.float64)
    phase = np.zeros(K, dtype=np.float64)
    ratio = np.arange(1, K + 1, dtype=np.float64)
    if gain is None:
        gain = np.ones(K, dtype=np.float64)
    if f0_b is None:
        f0_b = f0
    voice_block_add(out, n, phase, ratio, gain, gain, K, f0, f0_b, loud_a, loud_b, SR, NYQ, RAMP_HZ)
    return out, phase


class TestConstantFrequency:
    def test_fundamental_peak(self):
        n = int(SR)
        out, _ = _run_kernel(440.0, K=1, n=n)
        spec = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(n, 1.0 / SR)
        peak = freqs[np.argmax(spec[1:]) + 1]
        assert peak == pytest.approx(440.0, abs=2.0)

    def test_harmonic_series(self):
        K = 4
        out, _ = _run_kernel(440.0, K=K, n=int(SR))
        spec = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(int(SR), 1.0 / SR)
        for k in range(1, K + 1):
            lo = int((k * 440 - 20) * len(freqs) / (SR / 2))
            hi = int((k * 440 + 20) * len(freqs) / (SR / 2))
            assert np.max(spec[lo:hi]) > 0.1

    def test_zero_gain_skipped(self):
        K = 4
        gain = np.zeros(K, dtype=np.float64)
        out, _ = _run_kernel(440.0, K=K, gain=gain)
        assert np.all(out == 0.0)

    def test_output_accumulates(self):
        n = 1024
        out = np.ones(n, dtype=np.float64) * 0.5
        phase = np.zeros(1, dtype=np.float64)
        ratio = np.ones(1, dtype=np.float64)
        gain = np.ones(1, dtype=np.float64)
        voice_block_add(
            out, n, phase, ratio, gain, gain, 1, 440.0, 440.0, 1.0, 1.0, SR, NYQ, RAMP_HZ
        )
        assert not np.allclose(out, 0.5)


class TestPhaseAccumulation:
    def test_phase_advances(self):
        _, phase = _run_kernel(440.0, K=1, n=1024)
        assert phase[0] != 0.0

    def test_phase_wraps_to_two_pi(self):
        _, phase = _run_kernel(440.0, K=1, n=int(SR))
        assert 0.0 <= phase[0] < 2.0 * math.pi

    def test_phase_continuity_across_blocks(self):
        K = 1
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.ones(K, dtype=np.float64)
        gain = np.ones(K, dtype=np.float64)
        chunks = []
        for _ in range(4):
            out = np.zeros(512, dtype=np.float64)
            voice_block_add(
                out, 512, phase, ratio, gain, gain, K, 440.0, 440.0, 1.0, 1.0, SR, NYQ, RAMP_HZ
            )
            chunks.append(out)
        joined = np.concatenate(chunks)
        diffs = np.abs(np.diff(joined))
        assert np.max(diffs) < 0.1


class TestNyquistRamp:
    def test_harmonic_above_nyquist_silent(self):
        f0 = 5000.0
        K = 8
        gain = np.ones(K, dtype=np.float64)
        n = int(SR)
        out, _ = _run_kernel(f0, K=K, gain=gain, n=n)
        spec = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(n, 1.0 / SR)
        fund_bin = np.argmin(np.abs(freqs - f0))
        assert spec[fund_bin] > 1.0
        for k in [5, 6, 7, 8]:
            fk = k * f0
            if fk >= NYQ:
                out2 = np.zeros(n, dtype=np.float64)
                phase2 = np.zeros(1, dtype=np.float64)
                ratio2 = np.array([float(k)])
                voice_block_add(
                    out2,
                    n,
                    phase2,
                    ratio2,
                    gain[:1],
                    gain[:1],
                    1,
                    f0,
                    f0,
                    1.0,
                    1.0,
                    SR,
                    NYQ,
                    RAMP_HZ,
                )
                assert np.all(out2 == 0.0), f"harmonic {k} at {fk}Hz should be silent"

    def test_ramp_attenuates_near_nyquist(self):
        f0 = 7000.0
        K = 4
        gain = np.ones(K, dtype=np.float64)
        out_ramped, _ = _run_kernel(f0, K=K, gain=gain, n=4096)
        spec = np.abs(np.fft.rfft(out_ramped))
        freqs = np.fft.rfftfreq(4096, 1.0 / SR)
        k1_bin = np.argmin(np.abs(freqs - f0))
        k3_bin = np.argmin(np.abs(freqs - 3 * f0))
        k3_freq = 3 * f0
        if k3_freq > NYQ - RAMP_HZ and k3_freq < NYQ:
            assert spec[k3_bin] < spec[k1_bin]


class TestVariableFrequency:
    def test_sweep_produces_output(self):
        K = 2
        n = 4096
        out = np.zeros(n, dtype=np.float64)
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.arange(1, K + 1, dtype=np.float64)
        gain = np.ones(K, dtype=np.float64)
        voice_block_add(
            out, n, phase, ratio, gain, gain, K, 220.0, 440.0, 1.0, 1.0, SR, NYQ, RAMP_HZ
        )
        assert np.max(np.abs(out)) > 0

    def test_variable_freq_phase_wraps(self):
        K = 2
        n = int(SR)
        out = np.zeros(n, dtype=np.float64)
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.arange(1, K + 1, dtype=np.float64)
        gain = np.ones(K, dtype=np.float64)
        voice_block_add(
            out, n, phase, ratio, gain, gain, K, 220.0, 440.0, 1.0, 1.0, SR, NYQ, RAMP_HZ
        )
        for k in range(K):
            assert 0.0 <= phase[k] < 2.0 * math.pi


class TestGainInterpolation:
    def test_gain_ramp(self):
        K = 1
        n = 2048
        out = np.zeros(n, dtype=np.float64)
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.ones(K, dtype=np.float64)
        gain_a = np.ones(K, dtype=np.float64)
        gain_b = np.zeros(K, dtype=np.float64)
        voice_block_add(
            out, n, phase, ratio, gain_a, gain_b, K, 440.0, 440.0, 1.0, 1.0, SR, NYQ, RAMP_HZ
        )
        head_rms = np.sqrt(np.mean(out[: n // 4] ** 2))
        tail_rms = np.sqrt(np.mean(out[-n // 4 :] ** 2))
        assert head_rms > 2 * tail_rms

    def test_loudness_ramp(self):
        K = 1
        n = 2048
        out = np.zeros(n, dtype=np.float64)
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.ones(K, dtype=np.float64)
        gain = np.ones(K, dtype=np.float64)
        voice_block_add(
            out, n, phase, ratio, gain, gain, K, 440.0, 440.0, 0.0, 1.0, SR, NYQ, RAMP_HZ
        )
        head_rms = np.sqrt(np.mean(out[: n // 4] ** 2))
        tail_rms = np.sqrt(np.mean(out[-n // 4 :] ** 2))
        assert tail_rms > 2 * head_rms


class TestPhaseAccumulationPrecision:
    """D1: accumulated phase error < 1e-6 rad after 1s at constant frequency."""

    def test_1s_phase_error_fundamental(self):
        K = 1
        f0 = 440.0
        block = 512
        total = int(SR)
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.ones(K, dtype=np.float64)
        gain = np.ones(K, dtype=np.float64)
        out = np.zeros(block, dtype=np.float64)
        n_blocks = total // block
        for _ in range(n_blocks):
            out[:] = 0.0
            voice_block_add(
                out, block, phase, ratio, gain, gain, K, f0, f0, 1.0, 1.0, SR, NYQ, RAMP_HZ
            )
        expected = (2.0 * math.pi * f0 / SR * block * n_blocks) % (2.0 * math.pi)
        assert abs(phase[0] - expected) < 1e-6

    def test_1s_phase_error_high_harmonic(self):
        K = 4
        f0 = 440.0
        block = 256
        total = int(SR)
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.arange(1, K + 1, dtype=np.float64)
        gain = np.ones(K, dtype=np.float64)
        out = np.zeros(block, dtype=np.float64)
        n_blocks = total // block
        for _ in range(n_blocks):
            out[:] = 0.0
            voice_block_add(
                out, block, phase, ratio, gain, gain, K, f0, f0, 1.0, 1.0, SR, NYQ, RAMP_HZ
            )
        for k in range(K):
            fk = (k + 1) * f0
            if fk >= NYQ:
                continue
            expected = (2.0 * math.pi * fk / SR * block * n_blocks) % (2.0 * math.pi)
            assert abs(phase[k] - expected) < 1e-6, f"harmonic {k+1} phase drift"


class TestVariableFrequencyPrecision:
    """D1: instantaneous frequency of the linear ramp within a variable-frequency block matches expectation."""

    def test_linear_ramp_instantaneous_freq(self):
        K = 1
        n = 8192
        f0_a, f0_b = 220.0, 440.0
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.ones(K, dtype=np.float64)
        gain = np.ones(K, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        voice_block_add(
            out, n, phase, ratio, gain, gain, K, f0_a, f0_b, 1.0, 1.0, SR, NYQ, RAMP_HZ
        )
        analytic = np.unwrap(np.angle(
            np.exp(1j * np.cumsum(
                2.0 * np.pi / SR * np.array([
                    f0_a + (f0_b - f0_a) * (i + 1.0) / n for i in range(n)
                ])
            ))
        ))
        inst_freq = np.diff(analytic) / (2.0 * np.pi) * SR
        mid = n // 2
        expected_mid = f0_a + (f0_b - f0_a) * (mid + 1.0) / n
        measured = inst_freq[mid]
        assert measured == pytest.approx(expected_mid, rel=0.01)

    def test_sweep_total_phase(self):
        K = 1
        n = 4096
        f0_a, f0_b = 300.0, 600.0
        phase = np.zeros(K, dtype=np.float64)
        ratio = np.ones(K, dtype=np.float64)
        gain = np.ones(K, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        voice_block_add(
            out, n, phase, ratio, gain, gain, K, f0_a, f0_b, 1.0, 1.0, SR, NYQ, RAMP_HZ
        )
        expected_phase = sum(
            2.0 * math.pi * (f0_a + (f0_b - f0_a) * (i + 1.0) / n) / SR
            for i in range(n)
        ) % (2.0 * math.pi)
        assert phase[0] == pytest.approx(expected_phase, abs=0.01)


class TestNyquistRampPrecision:
    """D1: exact attenuation of the Nyquist fade-out."""

    def test_ramp_factor_matches_formula(self):
        f0 = NYQ - RAMP_HZ * 0.5
        K = 1
        n = 4096
        ratio_val = 1.0
        fk = ratio_val * f0
        assert NYQ - RAMP_HZ < fk < NYQ
        expected_factor = (NYQ - fk) / RAMP_HZ
        gain_full = np.ones(K, dtype=np.float64)
        out_ramped = np.zeros(n, dtype=np.float64)
        phase_r = np.zeros(K, dtype=np.float64)
        ratio_arr = np.array([ratio_val])
        voice_block_add(
            out_ramped, n, phase_r, ratio_arr, gain_full, gain_full,
            K, f0, f0, 1.0, 1.0, SR, NYQ, RAMP_HZ,
        )
        out_full = np.zeros(n, dtype=np.float64)
        phase_f = np.zeros(K, dtype=np.float64)
        low_f0 = 440.0
        voice_block_add(
            out_full, n, phase_f, ratio_arr, gain_full, gain_full,
            K, low_f0, low_f0, 1.0, 1.0, SR, NYQ, RAMP_HZ,
        )
        rms_ramped = np.sqrt(np.mean(out_ramped ** 2))
        rms_full = np.sqrt(np.mean(out_full ** 2))
        measured_factor = rms_ramped / rms_full
        assert measured_factor == pytest.approx(expected_factor, rel=0.05)


class TestWarmup:
    def test_warmup_runs(self):
        from synth.oscillator import warmup

        warmup()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
