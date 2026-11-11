"""L0 控制驱动模式(10.14c/g)回归:Voice.drive / Engine.render_control / ctl_legato /
释放无咔哒 / note_on 清残留 / 近正弦源的共振峰混合 / 旋钮 setter。

全部离线、轻量快照,不依赖 checkpoint 与声卡。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import synth.voice as V  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from tests.helpers import make_test_snapshot  # noqa: E402

FPS = 250.0


@pytest.fixture
def eng():
    return Engine(build_instrument(make_test_snapshot(), noise_level=0.0), eval_mode=True)


def _comb_f0(y, sr, a, b, candidates=(330.0, 440.0, 294.0)):
    """测试快照的谱形峰在 700Hz 附近(第 2 泛音比基频响),不能取最大谱峰当基频;
    改成谐波梳匹配:候选基频中各次谐波能量总和最大者。"""
    seg = y[int(a * sr) : int(b * sr)] * np.hanning(int(b * sr) - int(a * sr))
    S = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    best, best_e = None, -1.0
    for c in candidates:
        e = sum(S[(f > k * c * 0.97) & (f < k * c * 1.03)].max() for k in range(1, 6) if k * c < sr / 2)
        if e > best_e:
            best, best_e = c, e
    return best


def _hf_burst_max(y, sr, a, b, w=0.005):
    """5ms 窗 >4kHz 能量占比 / 中位:咔哒探测(10.14 教训:样本差指标只量响度起伏)。"""
    n = int(w * sr)
    hr = []
    for s in range(int(a * sr), int(b * sr) - n, n):
        seg = y[s : s + n] * np.hanning(n)
        S = np.abs(np.fft.rfft(seg)) ** 2
        f = np.fft.rfftfreq(n, 1 / sr)
        if S.sum() > 1e-10:
            hr.append(S[f > 4000].sum() / S.sum())
    hr = np.array(hr)
    return float(hr.max() / (np.median(hr) + 1e-12))


def test_render_control_follows_f0(eng):
    sr = eng.sr
    T = int(1.0 * FPS)
    f0 = np.concatenate([np.full(T // 2, 330.0), np.full(T - T // 2, 440.0)])
    y = eng.render_control(f0, np.zeros(T), tail_s=0.0)
    assert _comb_f0(y, sr, 0.1, 0.45) == 330.0
    assert _comb_f0(y, sr, 0.55, 0.95) == 440.0


def test_render_control_loud_gain_sets_dynamic_range(eng):
    T = int(0.6 * FPS)
    loud = np.concatenate([np.zeros(T // 2), np.full(T - T // 2, -3.0)])  # 后半 z=-3
    # 低力度:避开母线 tanh 软限幅(满幅时响度比会被压缩)
    y = eng.render_control(np.full(T, 220.0), loud, vel=40, loud_gain=0.4, tail_s=0.0)
    r1 = np.sqrt(np.mean(y[int(0.1 * eng.sr) : int(0.25 * eng.sr)] ** 2))
    r2 = np.sqrt(np.mean(y[int(0.4 * eng.sr) : int(0.55 * eng.sr)] ** 2))
    assert 20 * np.log10(r2 / r1) == pytest.approx(20 * np.log10(np.exp(-0.4 * 3.0)), abs=1.5)


def test_release_after_note_off_is_smooth(eng):
    """松键后:幅度单调衰减到 0,没有满幅末块(10.14 实测 15× 高频突发)。"""
    sr = eng.sr
    T = int(0.8 * FPS)
    y = eng.render_control(np.full(T, 294.0), np.zeros(T), tail_s=0.4)
    t_off = T / FPS
    rms = [
        np.sqrt(np.mean(y[int((t_off + o) * sr) : int((t_off + o + 0.01) * sr)] ** 2))
        for o in (0.0, 0.03, 0.06, 0.09, 0.12, 0.2)
    ]
    assert all(b <= a * 1.05 for a, b in zip(rms, rms[1:])), rms
    assert rms[-1] < 1e-4
    assert _hf_burst_max(y, sr, 0.2, 1.1) < 5.0


def test_ctl_legato_continuity(eng):
    """连音换轨迹:沿用声部,20ms 过渡,边界无幅度坑、无高频突发。"""
    sr = eng.sr
    T = int(0.6 * FPS)
    a = (np.full(T, 294.0), np.ones(T), FPS)
    b = (np.full(T, 440.0), np.ones(T), FPS)
    evs = [(0.0, "ctl", 62, 100, a, -1), (0.5, "ctl_legato", 69, 100, b, 62), (1.0, "off", 69, 0, 0.0, -1)]
    y = np.asarray(eng.render_offline(evs, 1.3), np.float64)
    pre = np.sqrt(np.mean(y[int(0.47 * sr) : int(0.49 * sr)] ** 2))
    mid = np.sqrt(np.mean(y[int(0.505 * sr) : int(0.515 * sr)] ** 2))
    post = np.sqrt(np.mean(y[int(0.53 * sr) : int(0.55 * sr)] ** 2))
    assert mid > 0.5 * min(pre, post), (pre, mid, post)
    assert _comb_f0(y, sr, 0.6, 0.95) == 440.0
    assert _hf_burst_max(y, sr, 0.2, 0.9) < 5.0
    assert eng.pool[1].age == 0  # 连音沿用第一声部,从未分配第二声部


def test_note_on_clears_stale_control_stream(eng):
    """复用的声部不能带着上一个控制流:普通起音必须是事件驱动。"""
    T = int(0.3 * FPS)
    eng.render_control(np.full(T, 300.0), np.zeros(T), tail_s=0.1)
    v = eng.pool[0]
    assert v._ctl_f0 is not None  # 上一次控制驱动的残留
    eng.render_offline([(0.0, "on", 60, 100, 0.0, -1), (0.2, "off", 60, 0, 0.0, -1)], 0.4)
    assert eng.pool[0]._ctl_f0 is None


def test_formant_blend_for_near_sine_source():
    """近正弦源(质心 ≈1)弹到基频以下时,第 2 泛音不能被采到源基频峰上(10.14 "泛音不自然")。"""
    snap = make_test_snapshot(f0=990.0)
    H = np.asarray(snap["H_env"], np.float32).copy()
    H[1:, :] *= 1e-3  # 近似纯正弦:只有基频
    snap["H_env"] = H
    inst = build_instrument(snap, noise_level=0.0)
    eng = Engine(inst, eval_mode=True)
    y = np.asarray(eng.render_offline([(0.0, "on", 67, 100, 0.0, -1), (0.6, "off", 67, 0, 0.0, -1)], 0.8), np.float64)
    seg = y[int(0.2 * eng.sr) : int(0.5 * eng.sr)]
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / eng.sr)
    h1 = S[(f > 392 * 0.95) & (f < 392 * 1.05)].max()
    h2 = S[(f > 784 * 0.95) & (f < 784 * 1.05)].max()
    assert getattr(inst, "_formant_w", 1.0) < 0.1
    assert h2 < 0.1 * h1


def test_tone_tilt_and_articulation_setters(eng):
    sr = eng.sr
    try:
        V.set_tone_tilt(6.0)
        y_b = np.asarray(eng.render_offline([(0.0, "on", 60, 100, 0.0, -1), (0.5, "off", 60, 0, 0.0, -1)], 0.6), np.float64)
        V.set_tone_tilt(-6.0)
        y_d = np.asarray(eng.render_offline([(0.0, "on", 60, 100, 0.0, -1), (0.5, "off", 60, 0, 0.0, -1)], 0.6), np.float64)
    finally:
        V.set_tone_tilt(0.0)

    def centroid(y):
        S = np.abs(np.fft.rfft(y[int(0.2 * sr) : int(0.45 * sr)]))
        f = np.fft.rfftfreq(len(y[int(0.2 * sr) : int(0.45 * sr)]), 1 / sr)
        return float((S * f).sum() / (S.sum() + 1e-9))

    assert centroid(y_b) > centroid(y_d) * 1.2
    base = V._SLIDE_FEEL_BASE
    try:
        V.set_articulation_intensity(2.0)
        assert pytest.approx(base[0] * 2.0) == V._P.slide_noise
        assert V._P.art_dip_legato <= 0.6
        V.set_articulation_intensity(0.0)
        assert V._P.slide_noise == 0.0
    finally:
        V.set_articulation_intensity(1.0)
