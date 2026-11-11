"""HarmonicSynth 分块合成 = 整段合成(数值一致),且峰值内存与乐句长度无关。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural.ddsp import HarmonicSynth, _upsample, _upsample_slice  # noqa: E402


def _reference(h, f0, amp, dist):
    n = f0.shape[1] * h.hop
    d = dist / (dist.sum(-1, keepdim=True) + 1e-7)
    d = d * ((f0 * h.k) < h.sr / 2).float()
    amps = _upsample(amp * d, n)
    phase = torch.cumsum(2 * np.pi * _upsample(f0, n) / h.sr, dim=1)
    return (amps * torch.sin(phase * h.k)).sum(-1)


def test_upsample_slice_matches_interpolate():
    torch.manual_seed(0)
    x = torch.randn(2, 37, 5)
    n = 37 * 64
    full = _upsample(x, n)
    for s0, s1 in ((0, n), (100, 2000), (n - 300, n), (1234, 1235)):
        torch.testing.assert_close(_upsample_slice(x, n, s0, s1), full[:, s0:s1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("T", (3, 100, 700))  # 700 帧 × hop 64 = 44800 样本 > CHUNK,跨两块
def test_chunked_forward_equals_reference(T):
    torch.manual_seed(1)
    h = HarmonicSynth(n_harm=12, sr=16000, hop=64)
    f0 = 200 + 50 * torch.rand(1, T, 1)
    amp = torch.rand(1, T, 1)
    dist = torch.rand(1, T, 12)
    y = h(f0, amp, dist)
    y_ref = _reference(h, f0, amp, dist)
    assert y.shape == y_ref.shape == (1, T * 64)
    torch.testing.assert_close(y, y_ref, atol=2e-4, rtol=1e-4)


def test_chunk_boundary_is_seamless():
    """块边界处不得出现台阶:相邻样本差分与块内同量级。"""
    h = HarmonicSynth(n_harm=8, sr=16000, hop=64)
    T = 1024  # 65536 样本 = 两整块
    f0 = torch.full((1, T, 1), 220.0)
    amp = torch.full((1, T, 1), 0.5)
    dist = torch.ones(1, T, 8)
    y = h(f0, amp, dist)[0]
    d = (y[1:] - y[:-1]).abs()
    b = HarmonicSynth.CHUNK
    assert float(d[b - 2 : b + 2].max()) <= float(d.max()) * 1.01
