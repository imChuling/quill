"""音频/MIDI 设备枚举与选择(Bridge 的 mixin)。

供设置页的设备面板调用:list_devices 返回 {input, output, midi, engine},set_* 立即生效,
probe_input 短暂开一条输入流测峰值。rtmidi 在非英文 macOS 上的端口名编码问题由
_fix_midi_name 还原,打开端口时再换回原始名。
"""

from __future__ import annotations


class BridgeDeviceMixin:

    def _open_midi(self, name=None):
        """打开外接 MIDI 输入(name=None 取第一个),消息直接进引擎队列(没设备就跳过)。"""
        try:
            from runtime.midi_in import MidiInput

            from .bridge import _MidiRouter

            raw = self._raw_midi_name(name) if name else None
            m = MidiInput(port_name=raw)
            m.queue = _MidiRouter(self)
            m.open()
            self._midi = m
            print(f"[bridge] MIDI 已连接: {m.port_name}", flush=True)
            return m.port_name
        except Exception as e:
            self._midi = None
            print(f"[bridge] 无外接 MIDI: {str(e)[:70]}", flush=True)
            return None

    @staticmethod
    def _fix_midi_name(name):
        """rtmidi 在非英文 macOS 上把 CoreMIDI 的 UTF-8 端口名按 MacRoman 误解码
        (如'蓝牙'→'ËìùÁâô')。重编回 MacRoman 再按 UTF-8 解 = 还原;不是这种情况则原样返回。"""
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
        """显示名(已修)→ rtmidi 的原始端口名(打开端口要用原始名匹配)。"""
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

    # ---- 设备面板 ----
    def list_devices(self):
        """枚举音频输入/输出 + MIDI + 引擎信息,给设置页渲染。"""
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
        """选录音输入设备(下次取色生效)。"""
        self._in_dev = int(idx) if idx is not None else None
        return {"ok": True}

    def set_output_device(self, idx):
        """选输出设备 → 重启引擎输出流到新设备(失败回落系统默认)。"""
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
        """短暂探测当前输入设备峰值电平(设备面板"测试"按钮)。"""
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
        """接指定 MIDI 端口(name=None 关闭)。"""
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
