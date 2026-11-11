"""Audio/MIDI device enumeration and selection (Bridge mixin).

Called by the device panel on the settings page: list_devices returns {input, output, midi, engine}, set_* take effect immediately,
probe_input briefly opens an input stream to measure the peak. The port-name encoding issue of rtmidi on non-English macOS is undone by
_fix_midi_name, and the original name is swapped back when opening the port.
"""

from __future__ import annotations


class BridgeDeviceMixin:

    def _open_midi(self, name=None):
        """Open external MIDI input (name=None takes the first one); messages go straight into the engine queue (skipped if there is no device)."""
        try:
            from runtime.midi_in import MidiInput

            from .bridge import _MidiRouter

            raw = self._raw_midi_name(name) if name else None
            m = MidiInput(port_name=raw)
            m.queue = _MidiRouter(self)
            m.open()
            self._midi = m
            print(f"[bridge] MIDI connected: {m.port_name}", flush=True)
            return m.port_name
        except Exception as e:
            self._midi = None
            print(f"[bridge] no external MIDI: {str(e)[:70]}", flush=True)
            return None

    @staticmethod
    def _fix_midi_name(name):
        """On non-English macOS rtmidi mis-decodes CoreMIDI's UTF-8 port names as MacRoman
        (e.g. a Chinese 'Bluetooth' becomes 'ËìùÁâô'). Re-encoding to MacRoman and decoding as UTF-8 restores it; otherwise returned unchanged."""
        if not isinstance(name, str):
            return name
        try:
            return name.encode("mac-roman").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return name

    def list_midi(self):
        try:
            import mido

            return [self._fix_midi_name(n) for n in mido.get_input_names()]
        except Exception:
            return []

    def _raw_midi_name(self, disp):
        """Display name (fixed) -> rtmidi's original port name (opening a port must match on the original name)."""
        try:
            import mido

            for r in mido.get_input_names():
                if self._fix_midi_name(r) == disp:
                    return r
        except Exception:
            pass
        return disp

    def midi_name(self):
        return self._fix_midi_name(self._midi.port_name) if self._midi else None

    # ---- Device panel ----
    def list_devices(self):
        """Enumerate audio inputs/outputs + MIDI + engine info for the settings page to render."""
        import sounddevice as sd

        try:
            devs = sd.query_devices()
        except Exception as e:
            return {"input": [], "output": [], "midi": [], "engine": {}, "err": str(e)[:60]}
        try:
            din, dout = sd.default.device
        except Exception:
            din, dout = -1, -1
        cur_in = self._in_dev if self._in_dev is not None else din
        cur_out = self._out_dev if self._out_dev is not None else dout
        ins, outs, seen_i, seen_o = [], [], set(), set()
        for i, d in enumerate(devs):
            nm = d["name"]
            if d.get("max_input_channels", 0) > 0 and nm not in seen_i:
                seen_i.add(nm)
                ins.append({"idx": i, "name": nm, "sel": i == cur_in})
            if d.get("max_output_channels", 0) > 0 and nm not in seen_o:
                seen_o.add(nm)
                outs.append({"idx": i, "name": nm, "sel": i == cur_out})
        cur_midi = self.midi_name()
        midi = [{"name": n, "sel": n == cur_midi} for n in self.list_midi()]
        eng = {}
        try:
            lat = self.engine.stream_latency_ms()
            eng = {
                "sr": int(self.engine.sr),
                "blocksize": int(self.engine.blocksize),
                "latency_ms": round(float(lat), 1) if lat == lat else None,
                "voices": int(self.engine.max_voices),
            }
        except Exception:
            pass
        return {"input": ins, "output": outs, "midi": midi, "engine": eng}

    def set_input_device(self, idx):
        """Select the recording input device (takes effect on the next capture)."""
        self._in_dev = int(idx) if idx is not None else None
        return {"ok": True}

    def set_output_device(self, idx):
        """Select the output device -> restart the engine output stream on the new device (falls back to the system default on failure)."""
        target = int(idx) if idx is not None else None
        if not self._running:
            self._out_dev = target
            return {"ok": True}
        try:
            self.engine.stop()
            self.engine.start(queue=self._q, device=target)
            self._out_dev = target
            return {"ok": True}
        except Exception as e:
            try:
                self.engine.start(queue=self._q, device=None)
                self._out_dev = None
            except Exception:
                pass
            return {"ok": False, "msg": str(e)[:80]}

    def probe_input(self, ms=350):
        """Briefly probe the peak level of the current input device (the "Test" button in the device panel)."""
        import time

        import numpy as np
        import sounddevice as sd

        buf = []
        try:
            st = sd.InputStream(
                samplerate=44100,
                channels=1,
                device=self._in_dev,
                callback=lambda ind, f, t, s: buf.append(ind[:, 0].copy()),
            )
            st.start()
            time.sleep(max(0.1, min(1.0, ms / 1000.0)))
            st.stop()
            st.close()
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        if not buf:
            return {"ok": True, "peak": 0.0}
        y = np.concatenate(buf)
        return {"ok": True, "peak": float(np.abs(y).max())}

    def set_midi_input(self, name):
        """Connect the given MIDI port (name=None closes it)."""
        if self._midi:
            try:
                self._midi.close()
            except Exception:
                pass
            self._midi = None
        if not name:
            return {"ok": True, "name": None}
        pn = self._open_midi(name)
        return {"ok": bool(pn), "name": pn}
