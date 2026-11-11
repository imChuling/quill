"""卷帘导出(MIDI / WAV / 神经 WAV)、节拍器与录制(Bridge 的 mixin)。

录制走引擎事件队列的 arm/disarm:录制中队列记录 (t, event),record_poll 把已记录的事件
配对成音符供卷帘实时显示,record_stop 配对后把首音对齐到 0。
"""

from __future__ import annotations

import time


class BridgeExportMixin:

    def _take_events(self, notes):
        """卷帘音符 → 引擎 cooked 事件(秒)。"""
        return self._gi.notes_to_events(notes)

    def export_take(self, notes, bpm=120.0, fmt="wav", path=None, neural_opts=None):
        """卷帘整段导出:fmt='midi' / 'wav' / 'wav_neural:<instrument>'。"""
        if str(fmt).startswith("wav_neural"):
            if not notes:
                return {"ok": False, "msg": "Nothing to export"}
            if not path:
                return {"ok": False, "msg": "No path"}
            try:
                import soundfile as sf

                inst = fmt.split(":", 1)[1] if ":" in fmt else "violin"
                y, sr = self._render_neural(notes, inst, neural_opts)
                sf.write(str(path), y, sr)
                return {"ok": True, "path": str(path)}
            except Exception as e:
                return {"ok": False, "msg": str(e)[:80]}
        if not notes:
            return {"ok": False, "msg": "Nothing to export — record or draw notes first"}
        if not path:
            return {"ok": False, "msg": "No path"}
        try:
            if fmt == "midi":
                import mido

                mid = mido.MidiFile(ticks_per_beat=480)
                tr = mido.MidiTrack()
                mid.tracks.append(tr)
                tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=0))
                evs = []
                for n in notes:
                    t0 = float(n["start"])
                    t1 = t0 + float(n["len"])
                    evs.append((t0, "note_on", int(n["midi"]), int(n.get("vel", 100))))
                    evs.append((t1, "note_off", int(n["midi"]), 0))
                evs.sort(key=lambda e: (e[0], e[1] == "note_on"))
                last = 0
                for t, kind, m, v in evs:
                    tick = int(round(t * float(bpm) / 60.0 * 480))
                    tr.append(mido.Message(kind, note=m, velocity=v, time=max(0, tick - last)))
                    last = tick
                mid.save(str(path))
            else:
                import copy

                import numpy as np
                import soundfile as sf

                from runtime.engine import Engine

                inst = copy.copy(self.engine.instrument)
                eng = Engine(inst, eval_mode=True)
                evs = self._take_events(notes)
                dur = max(float(n["start"]) + float(n["len"]) for n in notes) + 1.5
                y = eng.render_offline(evs, dur)
                sf.write(str(path), np.asarray(y, np.float32), eng.sr)
            return {"ok": True, "path": str(path)}
        except Exception as e:
            return {"ok": False, "msg": str(e)[:80]}

    def click_on(self, bpm, bpb=4):
        self.engine.set_click(float(bpm), accent=int(bpb))
        return True

    def click_off(self):
        self.engine.set_click(None)
        return True

    # ---- 录制 ----
    def record_start(self):
        self._q.arm()
        return True

    @staticmethod
    def _pair(log, end_t):
        """事件流 → 音符:on/slide/legato=起,off=止。"""
        ART = {"on": "pluck", "slide": "slide", "legato": "legato"}
        notes, open_ = [], {}

        def close(m, t):
            t0, v0, a0 = open_.pop(m)
            notes.append(
                {
                    "midi": m,
                    "start": round(t0, 4),
                    "len": round(max(0.05, t - t0), 4),
                    "vel": v0,
                    "art": a0,
                }
            )

        for t, ev in log:
            m = int(ev.note)
            if ev.kind in ART and m >= 0:
                if m in open_:
                    close(m, t)
                open_[m] = (t, int(ev.vel) or 100, ART[ev.kind])
            elif ev.kind == "off" and m in open_:
                close(m, t)
        for m in list(open_):
            close(m, end_t)
        notes.sort(key=lambda n: (n["start"], n["midi"]))
        return notes

    def record_poll(self):
        if not self._q.rec:
            return {"notes": [], "t": 0.0}
        now = time.perf_counter() - self._q._t0
        return {"notes": self._pair(list(self._q.log), now), "t": round(now, 4)}

    def record_stop(self):
        log = self._q.disarm()
        end_t = log[-1][0] if log else 0.0
        notes = self._pair(log, end_t)
        if notes and notes[0]["start"] > 0:
            t0 = notes[0]["start"]
            for n in notes:
                n["start"] = round(n["start"] - t0, 4)
        return notes
