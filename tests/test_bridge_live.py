"""桌面桥 10.14 新路径回归:旋钮全部接引擎、strum 错峰、自动噪声底、烘焙守门、
实时行为模式(需要 checkpoint 的用例在缺失时跳过)。QuillBridge() 不开流,安全。"""

import queue
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

import bridge as bridge_mod  # noqa: E402
from bridge import QuillBridge  # noqa: E402

from neural.neural_take import INSTRUMENTS  # noqa: E402

HAVE_CKPT = all(s["ckpt"].exists() and s["traj"].exists() for s in INSTRUMENTS.values())
needs_ckpt = pytest.mark.skipif(not HAVE_CKPT, reason="DDSP checkpoints / trajectory banks missing")


@pytest.fixture(scope="module")
def b():
    return QuillBridge()


def _drain(b):
    out = []
    while True:
        try:
            out.append(b._q.get_nowait())
        except queue.Empty:
            return out


# ---------- 旋钮:每个都必须真的作用于引擎 ---------- #
def test_knobs_reach_engine(b):
    eng = b.engine
    b.set_param("master", 0.5)
    assert eng.master_gain == pytest.approx(0.7)
    b.set_param("noise", 0.3)
    assert eng.instrument.noise_level == 0.3
    b.set_param("vibrate", 0.5)
    assert eng.vib_rate_hz == pytest.approx(6.0)
    b.set_param("strum", 1.0)
    assert b._gi.strum_spread == pytest.approx(0.078)
    b.set_param("gsens", 1.0)
    assert b._gi.legato_dur == pytest.approx(0.1) and b._gi.slide_dur == pytest.approx(0.22)
    import synth.voice as V

    b.set_param("brightness", 1.0)
    assert pytest.approx(6.0) == V._P.tone_tilt
    b.set_param("artint", 0.5)
    assert pytest.approx(V._SLIDE_FEEL_BASE[0]) == V._P.slide_noise
    b.set_param("brightness", 0.5)
    _drain(b)


def test_pick_rebuilds_instrument_grain(b):
    b.params["pick"] = 0.0
    b._rebuild_instrument()
    g0 = np.asarray(b.engine.instrument.attack_grain, np.float64).copy()
    b.params["pick"] = 1.0
    b._rebuild_instrument()
    g1 = np.asarray(b.engine.instrument.attack_grain, np.float64)
    assert len(g0) == len(g1) and not np.allclose(g0, g1)
    b.params["pick"] = 0.5
    b._rebuild_instrument()


def test_strum_offsets_cluster(b):
    b.set_param("strum", 0.0)  # 8ms
    chord = [{"midi": m, "start": 1.0, "len": 0.5, "vel": 100, "art": "strum"} for m in (52, 57, 61)]
    solo = [{"midi": 60, "start": 2.0, "len": 0.5, "vel": 100, "art": "pluck"}]
    offs = b._gi.strum_offsets(sorted(chord + solo, key=lambda n: n["start"]))
    assert offs == pytest.approx([0.0, 0.008, 0.016, 0.0])
    ons = [round(e[0], 3) for e in b._take_events(chord) if e[1] == "on"]
    assert ons == [1.0, 1.008, 1.016]


def test_auto_noise_level_by_centroid():
    sine = np.zeros(60)
    sine[0] = 1.0
    assert QuillBridge._auto_noise_level(sine) == pytest.approx(0.30)
    rich = np.exp(-np.arange(60) * 0.15)
    assert QuillBridge._auto_noise_level(rich) == pytest.approx(0.02)


def test_bake_guards(b, tmp_path, monkeypatch):
    monkeypatch.setattr(bridge_mod, "USER_TONES", tmp_path)
    import ui.bridge as _ub
    monkeypatch.setattr(_ub, "USER_TONES", tmp_path)
    r = b.bake_ddsp("x")
    assert not r["ok"] and "Capture" in r["msg"]
    assert b.bake_status()["state"] == "idle"
    assert b.behavior_status()["state"] == "idle"
    assert b._bake_auto_base() in ("violin", "trumpet", "guitar", "glass")


# ---------- 实时行为模式 ---------- #
@needs_ckpt
def test_live_behavior_control_stream(b):
    r = b.set_live_behavior("nope", wait=True)
    assert not r["ok"] and b.live_behavior_status()["name"] is None
    r = b.set_live_behavior("violin", wait=True)
    assert r["ok"] and r["ready"] and b._live_behavior == "violin"
    # 实时检索对象是离线那份的零拷贝副本:共享数组,各自 last_idx
    live_nn = b._live_src[0]
    offline_nn = b._neural().voice("violin").nn
    assert live_nn is not offline_nn and live_nn.trajectories is offline_nn.trajectories
    # vibrato 标签:旋钮或 CC1 的最近深度 >0.15
    b.set_param("vibdepth", 0.3)
    assert b._live_art_label("pluck", -1) == 1
    b.vibrato(0.0)
    assert b._live_art_label("pluck", -1) == 0
    assert b._live_art_label("slide", 60) == 2
    b.set_param("vibdepth", 0.3)
    _drain(b)
    f0, amp, fps = b._live_ctl(62, 100, "pluck", -1)
    assert fps == 250.0 and len(f0) == len(amp) == int(max(0.8, b._live_dur) * 250)
    assert amp.max() == pytest.approx(1.0) and amp.min() > 0.0
    nom = 440.0 * 2 ** ((62 - 69) / 12)
    assert abs(1200 * np.log2(f0[-1] / nom)) < 1.0  # 末帧收回标称值
    assert np.abs(1200 * np.log2(f0 / nom)).max() < 60.0
    f0s, amps, _ = b._live_ctl(64, 100, "legato", 62)
    assert len(f0s) == len(f0)
    b.set_live_behavior(None)


@needs_ckpt
def test_live_behavior_event_kinds(b):
    b.set_live_behavior("violin", wait=True)
    b._held, b._onsets = [], []
    _drain(b)
    assert b.note_on(62, 100, "auto") == "pluck"
    kinds = [e.kind for e in _drain(b)]
    assert kinds == ["ctl"]
    time.sleep(0.1)  # 避开 strum 窗口
    assert b.note_on(64, 100, "auto") == "legato"
    ev = _drain(b)
    assert [e.kind for e in ev] == ["ctl_legato"] and ev[0].note2 == 62
    b.note_off(62)
    b.note_off(64)
    _drain(b)
    b.set_live_behavior(None)
    b._held, b._onsets = [], []
    assert b.note_on(60, 100, "auto") == "pluck"
    assert [e.kind for e in _drain(b)] == ["on"]
    b.note_off(60)
    _drain(b)


@needs_ckpt
def test_live_phrase_renders_without_bursts(b):
    """同一事件流离线复现:换音与松键处无高频突发(10.14 两轮耳测修的两个 bug)。"""
    b.set_live_behavior("violin", wait=True)
    b._held, b._onsets = [], []
    _drain(b)
    evs = [(0.0, "vibrato", -1, 0, 0.3, -1)]

    def on(m, t):
        time.sleep(0.1)
        b.note_on(m, 100, "auto")
        for ev in _drain(b):
            evs.append((t, ev.kind, ev.note, ev.vel, ev.value, ev.note2))

    def off(m, t):
        b.note_off(m)
        for ev in _drain(b):
            evs.append((t, ev.kind, ev.note, ev.vel, ev.value, ev.note2))

    on(62, 0.0)
    on(64, 1.0)
    off(62, 1.01)
    off(64, 1.6)
    y = np.asarray(b.engine.render_offline(evs, 2.0), np.float64)
    sr = b.engine.sr
    n = int(0.005 * sr)
    hr = []
    for s in range(int(0.2 * sr), int(1.9 * sr) - n, n):
        seg = y[s : s + n] * np.hanning(n)
        S = np.abs(np.fft.rfft(seg)) ** 2
        f = np.fft.rfftfreq(n, 1 / sr)
        if S.sum() > 1e-6 * (y**2).max() * n:
            hr.append(S[f > 4000].sum() / S.sum())
    hr = np.array(hr)
    assert hr.max() / np.median(hr) < 6.0
    assert np.sqrt(np.mean(y[int(1.9 * sr) :] ** 2)) < 1e-3  # 释放走完真静音
    b.set_live_behavior(None)
    b._held, b._onsets = [], []


@needs_ckpt
def test_rapid_behavior_switch_no_crash(b):
    """压力测试:预热中连续切换行为库 20 次,不崩不挂。"""
    for i in range(20):
        name = "violin" if i % 2 == 0 else "trumpet"
        b.set_live_behavior(name, wait=False)
    b.set_live_behavior("violin", wait=True)
    assert b.live_behavior_status()["ready"]
    b.set_live_behavior(None)
