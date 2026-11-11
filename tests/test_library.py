"""冲刺新代码回归：f0 兜底修正 + 语料 build/retrieve + flatten/reconstruct 表示。"""

import numpy as np
import pytest

from analysis.timbre import _snap_to_hint
from library import SnapshotLibrary, _flatten_snapshot, build_corpus, reconstruct_snapshot
from snapshot import make_snapshot
from synth.noise import _NOISE_NFFT, mel_filterbank
from synth.render import render

SR = 44100


# ---- f0 hint 修正（审核 #2）----
def test_snap_harmonic_error():
    hint = 46.0
    out = _snap_to_hint(np.full(50, 3 * hint), hint)  # 锁 3 次谐波
    assert abs(np.median(out) - hint) < hint * 0.03


def test_snap_subharmonic():
    hint = 200.0
    out = _snap_to_hint(np.full(50, hint / 2), hint)  # 锁 1/2 次
    assert abs(np.median(out) - hint) < hint * 0.03


def test_snap_keeps_correct():
    hint = 220.0
    out = _snap_to_hint(np.full(50, hint), hint)
    assert abs(np.median(out) - hint) < hint * 0.01


def test_snap_deepbass_fallback():
    """极低音：抖动大、非简单比值 -> 退化常量 hint。"""
    hint = 20.0
    rng = np.random.default_rng(0)
    f0 = 40 + rng.standard_normal(50) * 18
    out = _snap_to_hint(f0, hint, max_dev_cents=50)
    assert np.allclose(out, hint)


def test_snap_all_invalid():
    out = _snap_to_hint(np.full(10, np.nan), 100.0)
    assert np.allclose(out, 100.0)


# ---- flatten / reconstruct 表示（审核 #1）----
def _synthetic_snapshot(f0=200.0, T=24, formant_hz=700.0):
    K, M = 60, 40
    k = np.arange(1, K + 1)
    steady = np.exp(-(((k * f0 - formant_hz) / 500) ** 2)) + 0.05
    H = np.outer(steady, np.linspace(0.2, 1.0, T)).astype(np.float32)  # 时间涌动
    N = (0.01 * np.ones((M, T))).astype(np.float32)
    _, centers = mel_filterbank(SR, _NOISE_NFFT, M)
    return make_snapshot(f0, H, N, np.zeros(2646, np.float32), noise_centers_hz=centers, sr=SR)


def test_flatten_dim():
    assert _flatten_snapshot(_synthetic_snapshot()).shape == (101,)


def test_reconstruct_self_identity():
    s = _synthetic_snapshot()
    r = reconstruct_snapshot(_flatten_snapshot(s), s)
    assert np.allclose(r["H_env"], s["H_env"], rtol=1e-3, atol=1e-5)


def test_reconstruct_takes_target_spectrum():
    A = _synthetic_snapshot(formant_hz=700.0)
    B = _synthetic_snapshot(formant_hz=1800.0)  # 不同谱形
    r = reconstruct_snapshot(_flatten_snapshot(A), B)  # A 谱 × B 动态
    rec = np.log(np.maximum(np.asarray(r["H_env"]).mean(1), 1e-6))[:60]
    tgt = np.log(np.maximum(np.asarray(A["H_env"]).mean(1), 1e-6))[:60]
    assert np.allclose(rec, tgt, atol=0.1)  # 稳态截面取自 A
    assert np.asarray(r["H_env"]).shape == np.asarray(B["H_env"]).shape  # 动态来自 B


def test_reconstruct_renders():
    s = _synthetic_snapshot()
    r = reconstruct_snapshot(_flatten_snapshot(s), s)
    y = render(r, 220.0, length=8000, mode="formant", add_attack=False)
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 1e-4


# ---- 语料 build -> retrieve（审核 #4）----
def _tone(midi, sr=SR, dur=0.6):
    f0 = 440 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(sr * dur)) / sr
    y = sum(np.cos(2 * np.pi * k * f0 * t) / k for k in range(1, 20))
    return (y / np.max(np.abs(y)) * 0.9).astype(np.float32)


def test_corpus_build_retrieve(tmp_path):
    groups = [("TinySOL/Strings/TestInstr", [(60, _tone(60)), (64, _tone(64))])]
    build_corpus(groups, str(tmp_path), verbose=False)
    lib = SnapshotLibrary(str(tmp_path))
    assert "TestInstr" in lib.tags()
    snap, meta = lib.retrieve(tag="TestInstr", pitch_midi=62)
    assert snap["H_env"].shape[0] == 60
    X, m = lib.matrix()
    assert X.shape[1] == 101 and len(m) == len(lib.index)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
