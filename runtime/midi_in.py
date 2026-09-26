"""MIDI 输入 —— FR-4（阶段3 Day1）。

mido callback 线程把消息转成 `engine.Event` 投进 `queue.SimpleQueue`；引擎在音频
回调里 `get_nowait()` 排空（铁律 #8：跨线程只走队列，回调内不加锁）。

坑位：MIDI 设备拔插 -> 捕获异常继续运行 + 置标志，不退出（铁律 #16）。

引擎铁律 #12：runtime 只依赖 numpy + numba；mido 是 MIDI I/O，不属被禁的
librosa/torch/UI，允许在本模块。
"""
from __future__ import annotations

import queue

from runtime.engine import Event

_BEND_RANGE_ST = 2.0   # pitch-bend 默认 ±2 半音


def list_ports():
    """返回可用 MIDI 输入端口名列表（无 mido/后端则空表）。"""
    try:
        import mido
        return list(mido.get_input_names())
    except Exception:
        return []


def msg_to_event(msg):
    """mido 消息 -> Event 或 None。"""
    t = msg.type
    if t == "note_on" and msg.velocity > 0:
        return Event("on", msg.note, msg.velocity, 0.0)
    if t == "note_off" or (t == "note_on" and msg.velocity == 0):
        return Event("off", msg.note, 0, 0.0)
    if t == "pitchwheel":
        return Event("bend", -1, 0, (msg.pitch / 8192.0) * _BEND_RANGE_ST)
    if t == "control_change" and msg.control == 1:   # CC1 调制轮 -> 颤音深度
        return Event("cc", -1, 0, msg.value / 127.0)
    if t == "control_change" and msg.control == 64:  # CC64 延音踏板
        return Event("pedal", -1, 0, 1.0 if msg.value >= 64 else 0.0)
    return None


class MidiInput:
    """打开 MIDI 端口，消息 -> 队列。拔插安全。"""

    def __init__(self, port_name: str | None = None):
        self.port_name = port_name
        self.queue: "queue.SimpleQueue[Event]" = queue.SimpleQueue()
        self._port = None
        self.error_flag = False
        self.connected = False

    def open(self):
        try:
            import mido
        except Exception as exc:
            self.error_flag = True
            raise RuntimeError(f"mido 不可用：{exc}") from exc
        try:
            name = self.port_name
            if name is None:
                names = mido.get_input_names()
                if not names:
                    raise RuntimeError("未发现 MIDI 输入设备（确认键盘已连接）")
                name = names[0]
                self.port_name = name
            self._port = mido.open_input(name, callback=self._on_message)
            self.connected = True
        except Exception as exc:
            self.error_flag = True
            self.connected = False
            raise RuntimeError(f"打开 MIDI 端口失败：{exc}") from exc
        return self

    def _on_message(self, msg):
        # 回调在 mido 线程；任何异常都吞掉，绝不让 MIDI 线程崩掉影响音频
        try:
            ev = msg_to_event(msg)
            if ev is not None:
                self.queue.put(ev)
        except Exception:
            self.error_flag = True

    def close(self):
        try:
            if self._port is not None:
                self._port.close()
        except Exception:
            pass
        finally:
            self._port = None
            self.connected = False
