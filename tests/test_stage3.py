"""阶段3 单测：ADSR 时间常数 / 力度映射 / 偷音 / 采样级触发 / 离线确定性。"""

import numpy as np
import pytest

from runtime.engine import Engine, Event
from synth.instrument import Instrument
from synth.voice import _REL, _SUS, Voice

SR = 44100
VEL_CFG = {"loud_exp": 1.5, "tilt_db_oct": 6.0}
RT = {"attack_fade_ms": 20.0, "steal_fade_ms": 4.0, "nyquist_ramp_hz": 600.0}


def _inst(K=8, noise=0.0):
    H = 1.0 / np.arange(1, K + 1)
    return Instrument(
        H,
        np.arange(1, K + 1, dtype=float),
        np.zeros(0),
        np.zeros(0),
        220.0,
        SR,
        name="t",
        noise_level=noise,
    )


def _engine(K=8, noise=0.0, eval_mode=True):
    cfg = {
        "runtime": {
            "blocksize": 256,
            "max_voices": 16,
            "channels": 1,
            "noise_level": noise,
            "attack_fade_ms": 20.0,
            "steal_fade_ms": 4.0,
            "nyquist_ramp_hz": 600.0,
        },
        "adsr": {"attack_ms": 8.0, "decay_ms": 120.0, "sustain": 0.7, "release_ms": 180.0},
        "velocity": VEL_CFG,
    }
    return Engine(_inst(K, noise), cfg=cfg, eval_mode=eval_mode)


# --------------------------------------------------------------------------- #
# ADSR 时间常数实测 vs 设定 ±10%
# --------------------------------------------------------------------------- #
def test_adsr_attack_release_within_10pct():
    adsr = {"attack_ms": 12.0, "decay_ms": 60.0, "sustain": 0.6, "release_ms": 150.0}
    v = Voice(8, SR)
    v.note_on(69, 100, _inst(), adsr, VEL_CFG, RT, age=1)

    n = 8
    t = 0
    # 测起音时间：env 首次达 ~1.0
    atk_t = None
    while t < int(SR * 0.05):
        _, e = v._advance_adsr(n)
        t += n
        if atk_t is None and e >= 0.999:
            atk_t = t / SR * 1000.0
    assert atk_t is not None
    assert abs(atk_t - adsr["attack_ms"]) / adsr["attack_ms"] < 0.10, (
        f"起音 {atk_t:.2f}ms vs 设定 {adsr['attack_ms']}ms"
    )

    # 推到 sustain，note_off，测从 sustain 到静音的全程换算
    while v.stage != _SUS and t < int(SR * 0.5):
        v._advance_adsr(n)
        t += n
    sus_level = v.env
    v.note_off()
    t_off = 0
    while v.active and t_off < int(SR * 0.5):
        _, e = v._advance_adsr(n)
        t_off += n
    rel_ms = (t_off / SR * 1000.0) / sus_level  # 归一到满量程
    assert abs(rel_ms - adsr["release_ms"]) / adsr["release_ms"] < 0.10, (
        f"释放(满量程) {rel_ms:.1f}ms vs 设定 {adsr['release_ms']}ms"
    )


# --------------------------------------------------------------------------- #
# 力度映射：loud=(vel/127)^1.5；高力度更亮
# --------------------------------------------------------------------------- #
def test_velocity_loudness_and_brightness():
    v = Voice(8, SR)
    inst = _inst(8)
    adsr = {"attack_ms": 8.0, "decay_ms": 60.0, "sustain": 0.7, "release_ms": 150.0}

    v.note_on(69, 127, inst, adsr, VEL_CFG, RT, age=1)
    assert v.vel_loud == pytest.approx(1.0, abs=1e-6)
    bright_hi = v.gain_k[6] / v.gain_k[0]  # 高/低谐波比

    v.note_on(69, 40, inst, adsr, VEL_CFG, RT, age=2)
    assert v.vel_loud == pytest.approx((40 / 127) ** 1.5, rel=1e-5)
    bright_lo = v.gain_k[6] / v.gain_k[0]

    assert bright_hi > bright_lo * 1.5, "高力度应明显更亮（高谐波相对更强）"


# --------------------------------------------------------------------------- #
# 偷音：优先最旧的 released，否则最旧
# --------------------------------------------------------------------------- #
def test_voice_steal_prefers_oldest_released():
    eng = _engine()
    # 占满 16 个声部，age 递增
    for i in range(16):
        eng._apply_event(Event("on", 60 + i, 100, 0.0))
    # 让其中两个进入 release，age 较小的应被偷
    eng.pool[5].stage = _REL
    eng.pool[5].age = 1
    eng.pool[9].stage = _REL
    eng.pool[9].age = 3
    victim = eng._alloc_voice()
    assert victim is eng.pool[5], "应偷 age 最小的 released 声部"


def test_alloc_prefers_free_voice():
    eng = _engine()
    for i in range(3):
        eng._apply_event(Event("on", 60 + i, 100, 0.0))
    v = eng._alloc_voice()
    assert not v.active, "有空闲声部时应优先空闲"


# --------------------------------------------------------------------------- #
# 块内采样级触发：note-on 之前静音，之后起声
# --------------------------------------------------------------------------- #
def test_sample_accurate_onset():
    eng = _engine()
    y = eng.render_offline([(0.05, "on", 69, 110, 0.0)], 0.2)
    on = int(np.argmax(np.abs(y) > 0.01))
    expect = int(0.05 * SR)
    assert abs(on - expect) < 512, f"起声 {on} 期望 ~{expect}"
    assert np.max(np.abs(y[: expect - 300])) < 1e-4, "note-on 之前应静音"


def test_pitch_matches_note():
    eng = _engine(K=20)
    y = eng.render_offline([(0.02, "on", 57, 100, 0.0), (0.45, "off", 57, 0, 0.0)], 0.5)
    seg = y[int(0.1 * SR) : int(0.35 * SR)]
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    f0 = 440.0 * 2 ** ((57 - 69) / 12.0)  # A3 = 220Hz
    assert abs(f[np.argmax(S)] - f0) < 5.0, f"基频 {f[np.argmax(S)]:.1f} vs {f0:.1f}"


# --------------------------------------------------------------------------- #
# 离线渲染：确定性 + 母线有界有限
# --------------------------------------------------------------------------- #
def test_offline_determinism():
    evs = [
        (0.0, "on", 60, 90, 0.0),
        (0.1, "on", 64, 90, 0.0),
        (0.3, "off", 60, 0, 0.0),
        (0.3, "off", 64, 0, 0.0),
    ]
    y1 = _engine(noise=0.5).render_offline(evs, 0.5)
    y2 = _engine(noise=0.5).render_offline(evs, 0.5)
    assert np.array_equal(y1, y2), "eval_mode 固定 seed 应逐样本一致"


def test_master_bus_bounded_finite():
    eng = _engine(K=40, noise=0.5)
    # 16 音齐发满载
    evs = [(0.0, "on", 48 + i, 120, 0.0) for i in range(16)]
    y = eng.render_offline(evs, 0.3)
    assert np.all(np.isfinite(y))
    # tanh 渐近 ±1（float32 可能舍入到 1.0）；要求有界不溢出
    assert np.max(np.abs(y)) <= 1.0, "tanh 软限幅应把母线限在 ±1"
    assert not eng.error_flag, "渲染不应触发错误标志"


def test_callback_never_crashes_on_bad_state():
    """回调最外层 try/except：即使乐器损坏也输出静音、不抛。"""
    eng = _engine()
    eng.instrument = None  # 故意破坏
    out = np.zeros(256, dtype=np.float32)
    eng._apply_event_safe = None
    eng.process_block(out, 256, 0)  # 不应抛异常
    # 注：触发渲染需要 active voice；这里仅验证不抛
    assert out.dtype == np.float32
