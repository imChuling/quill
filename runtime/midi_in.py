"""MIDI input: mido messages -> engine event queue.

The mido callback thread converts messages into `engine.Event` and puts them on a `queue.SimpleQueue`; the engine drains it with
`get_nowait()` in the audio callback: threads communicate only through the queue, no locks in the callback. Exceptions on MIDI device
hot-plug are caught, a flag is set and execution continues. mido is only imported in this module.
"""

from __future__ import annotations

import queue

from runtime.engine import Event

_BEND_RANGE_ST = 2.0  # pitch-bend default ±2 semitones
def msg_to_event(msg):
    """mido message -> Event or None."""
    t = msg.type
    if t == "note_on" and msg.velocity > 0:
        return Event("on", msg.note, msg.velocity, 0.0)
    if t == "note_off" or (t == "note_on" and msg.velocity == 0):
        return Event("off", msg.note, 0, 0.0)
    if t == "pitchwheel":
        return Event("bend", -1, 0, (msg.pitch / 8192.0) * _BEND_RANGE_ST)
    if t == "control_change" and msg.control == 1:  # CC1 mod wheel -> vibrato depth
        return Event("cc", -1, 0, msg.value / 127.0)
    if t == "control_change" and msg.control == 64:  # CC64 sustain pedal
        return Event("pedal", -1, 0, 1.0 if msg.value >= 64 else 0.0)
    return None


class MidiInput:
    """Open the MIDI port, messages go into the queue. Safe on hot-plug."""

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
            raise RuntimeError(f"mido unavailable: {exc}") from exc
        try:
            name = self.port_name
            if name is None:
                names = mido.get_input_names()
                if not names:
                    raise RuntimeError("No MIDI input device found (check that the keyboard is connected)")
                name = names[0]
                self.port_name = name
            self._port = mido.open_input(name, callback=self._on_message)
            self.connected = True
        except Exception as exc:
            self.error_flag = True
            self.connected = False
            raise RuntimeError(f"Failed to open MIDI port: {exc}") from exc
        return self

    def _on_message(self, msg):
        # the callback runs on the mido thread; swallow every exception, a crashed MIDI thread must not affect audio
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
