"""ui/bridge 纯函数单测(7.4 工程审计:补 ui 层零测试的账)。

只测不依赖声卡/PESTO 的确定性逻辑:演奏法判定语义、录制配对(art 保留)、
导出(MIDI 往返/事件编译)、音量平衡。QuillBridge() 构造不开流,安全。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

from bridge import QuillBridge, _RecQueue  # noqa: E402

from runtime.engine import Event  # noqa: E402


@pytest.fixture(scope="module")
def bridge():
    return QuillBridge()


# ---------- 演奏法实时判定:语义必须与 runtime/gesture.py 一致 ---------- #
def test_classify_pluck_legato_slide(bridge):
    b = bridge
    b._held, b._onsets = [], []
    art, frm = b._classify(60)
    assert art == "pluck" and frm == -1
    b._held = [60]  # 按住 60
    b._onsets = []
    art, frm = b._classify(62)  # +2 半音 ≤ legato_max_st
    assert art == "legato" and frm == 60
    b._onsets = []
    art, frm = b._classify(67)  # +7 半音 > legato_max_st
    assert art == "slide" and frm == 60
    b._held, b._onsets = [], []


def test_classify_strum_window(bridge):
    b = bridge
    b._held, b._onsets = [], []
    arts = [b._classify(60 + i)[0] for i in range(3)]  # 紧凑三连(远快于 80ms)
    assert arts[2] == "strum"
    b._held, b._onsets = [], []


def test_velocity_to_slide_speed(bridge):
    """力度→滑速:力度大滑得快(时长短)。参数唯一来源=GestureInterpreter。"""
    gi = bridge._gi
    assert gi._slide_dur(120) < gi._slide_dur(30)


# ---------- 录制配对:演奏法必须穿过录制 ---------- #
def test_pair_keeps_articulation():
    log = [
        (0.00, Event("on", 60, 95, 0.0, -1)),
        (0.50, Event("legato", 62, 90, 0.04, 60)),
        (0.55, Event("off", 60, 0, 0.0, -1)),
        (1.00, Event("slide", 69, 105, 0.08, 62)),
        (1.05, Event("off", 62, 0, 0.0, -1)),
        (1.60, Event("off", 69, 0, 0.0, -1)),
    ]
    notes = QuillBridge._pair(log, 1.60)
    arts = {n["midi"]: n["art"] for n in notes}
    assert arts == {60: "pluck", 62: "legato", 69: "slide"}
    assert all(n["len"] > 0 for n in notes)


def test_recqueue_disarm_clears_log():
    q = _RecQueue()
    q.arm()
    q.put(Event("on", 60, 100, 0.0, -1))
    out = q.disarm()
    assert len(out) == 1 and len(q.log) == 0  # 数据带走,内部清空(审计修复)


# ---------- 导出:MIDI 往返 / 事件编译语义 ---------- #
NOTES = [
    {"midi": 60, "start": 0.0, "len": 0.5, "vel": 100, "art": "pluck"},
    {"midi": 62, "start": 0.45, "len": 0.5, "vel": 90, "art": "legato"},
    {"midi": 69, "start": 0.9, "len": 0.8, "vel": 105, "art": "slide"},
]


def test_take_events_articulation(bridge):
    evs = bridge._take_events(NOTES)
    kinds = [e[1] for e in evs if e[1] != "off"]
    assert kinds == ["on", "legato", "slide"]
    slide = [e for e in evs if e[1] == "slide"][0]
    assert slide[5] == 62  # prev = 时间序前一音


def test_export_midi_roundtrip(bridge, tmp_path):
    import mido

    p = tmp_path / "t.mid"
    r = bridge.export_take(NOTES, bpm=120, fmt="midi", path=str(p))
    assert r["ok"]
    mid = mido.MidiFile(str(p))
    ons = [m for tr in mid.tracks for m in tr if m.type == "note_on" and m.velocity > 0]
    assert [m.note for m in ons] == [60, 62, 69]
    tempos = [m for tr in mid.tracks for m in tr if m.type == "set_tempo"]
    assert round(mido.tempo2bpm(tempos[0].tempo)) == 120


def test_export_empty_rejected(bridge, tmp_path):
    r = bridge.export_take([], fmt="midi", path=str(tmp_path / "x.mid"))
    assert not r["ok"]


# ---------- 音量平衡:取色响度拉齐到基准 ---------- #
def test_balance_normalizes_energy(bridge):
    import copy

    inst = copy.copy(bridge.engine.instrument)
    inst.H_steady = (np.asarray(inst.H_steady, np.float64) * 3.7).astype(np.float32)  # 假装录得很响
    bridge._balance(inst)
    e = float(np.sqrt((np.asarray(inst.H_steady, np.float64) ** 2).sum()))
    assert abs(e - bridge._ref_energy) / bridge._ref_energy < 1e-3
