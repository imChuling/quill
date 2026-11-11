"""MIDI 输入：mido 消息 -> 引擎事件队列。

mido 回调线程把消息转成 `engine.Event` 投进 `queue.SimpleQueue`，引擎在音频回调里
`get_nowait()` 排空：跨线程只走队列，回调内不加锁。MIDI 设备拔插时捕获异常、置标志并
继续运行。mido 只在本模块 import。
"""

from __future__ import annotations

import queue

from runtime.engine import Event

_BEND_RANGE_ST = 2.0  # pitch-bend 默认 ±2 半音
def msg_to_event(msg):
    """mido 消息 -> Event 或 None。"""
    t = msg.type
    if t == "note_on" and msg.velocity > 0:
        return Event("on", msg.note, msg.velocity, 0.0)
    if t == "note_off" or (t == "note_on" and msg.velocity == 0):
        return Event("off", msg.note, 0, 0.0)
    if t == "pitchwheel":
        return Event("bend", -1, 0, (msg.pitch / 8192.0) * _BEND_RANGE_ST)
    if t == "control_change" and msg.control == 1:  # CC1 调制轮 -> 颤音深度
        return Event("cc", -1, 0, msg.value / 127.0)
    if t == "control_change" and msg.control == 64:  # CC64 延音踏板
        return Event("pedal", -1, 0, 1.0 if msg.value >= 64 else 0.0)
    return None


class MidiInput:
    """打开 MIDI 端口，消息进队列。拔插安全。"""

    def __init__(self, port_name: str | None = None):
        self.port_name = port_name
        self.queue: queue.SimpleQueue[Event] = queue.SimpleQueue()
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
        # 回调在 mido 线程；任何异常都吞掉，MIDI 线程崩掉不能影响音频
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
