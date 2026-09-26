"""Quill app 桥 —— UI 事件 → 真实实时引擎(runtime/engine)。M2。

UI(WebView)里按键/旋钮通过 pywebview js_api 调这里;本类把事件 push 进引擎的
SimpleQueue(midi_in 同款路径),引擎在 sounddevice 音频回调里实时发声。线程安全:
SimpleQueue 跨线程,音频在独立回调线程跑,UI 调用只入队。

见 docs/app_plan.md §3 契约 / §4 参数映射。
"""
from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.engine import Event, engine_from_snapshot_file  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402  演奏法单一事实来源(FR-8)

DEFAULT_SNAPSHOT = ROOT / "assets" / "snapshots" / "synth_pad.npz"  # 温暖持续 pad:更讨好的开场默认
USER_TONES = ROOT / "assets" / "user_tones"                          # 用户保存的取色音色(.npz)


class _RecQueue:
    """事件队列代理:所有音符源(屏幕键盘/电脑键盘/外接 MIDI)都经此入队;
    录制中则给每个事件打时间戳记进 log。引擎只调 get_nowait(),透明转发。"""

    def __init__(self):
        self._q: "queue.SimpleQueue[Event]" = queue.SimpleQueue()
        self.rec = False
        self._t0 = 0.0
        self.log = []                         # [(t_sec, Event)]

    def arm(self):
        self.log = []; self._t0 = time.perf_counter(); self.rec = True

    def disarm(self):
        self.rec = False
        out = self.log
        self.log = []                         # 审计修复:停录即清,防残留(录制数据由返回值带走)
        return out

    def put(self, ev):
        if self.rec:
            self.log.append((time.perf_counter() - self._t0, ev))
        self._q.put(ev)

    def get_nowait(self):
        return self._q.get_nowait()


class _MidiRouter:
    """外接 MIDI → 与屏幕键盘同一条演奏法判定路径(on/off 走 note_on/off,其余直通队列)。"""

    def __init__(self, bridge):
        self._b = bridge

    def put(self, ev):
        if ev.kind == "on":
            art = self._b.note_on(ev.note, ev.vel, "auto")
            hook = getattr(self._b, "on_art", None)      # MIDI 路径:判定结果推回 UI 标签
            if hook and art:
                try:
                    hook(art)
                except Exception:
                    pass
        elif ev.kind == "off":
            self._b.note_off(ev.note)
        else:
            self._b._q.put(ev)                           # bend/vibrato/pedal 直通


class QuillBridge:
    """真实引擎桥:note_on/off + set_param。预设/取色热切换见 M4。"""

    def __init__(self, snapshot=DEFAULT_SNAPSHOT, noise_level=0.02):  # 噪声层默认 2%(7.3 用户耳测拍板;Air/Noise 旋钮可调)
        import numpy as np
        from snapshot import load_snapshot
        from synth.instrument import build_instrument
        from runtime.engine import Engine
        self._q = _RecQueue()                            # 录制代理队列(所有音符源汇合处)
        snap = load_snapshot(str(snapshot))
        inst = build_instrument(snap, noise_level=noise_level)
        self._ref_energy = float(np.sqrt((np.asarray(inst.H_steady, np.float64) ** 2).sum())) or 0.3
        self.engine = Engine(inst, eval_mode=False)      # 默认音色稳态能量 = 音量平衡基准
        self.params = {}
        self._running = False
        self._midi = None
        self._default_path = Path(snapshot)              # 音色库:默认音色快照
        self._last_snap = None                           # 最近取色/加载的快照(用于保存)
        self._last_disp = "Synth Pad"
        self._last_centroid = 6.0
        self._gi = GestureInterpreter()                  # 阈值/时长的唯一来源(legato_max_st、力度→滑速…)
        self._held = []                                  # 现场按住的键(按下顺序)
        self._onsets = []                                # 最近 onset 时刻(strum 窗口)
        self._sep_model = None                           # Demucs 惰性加载缓存(Mode 3)
        self._in_dev = None                              # 选中的录音输入设备(None=系统默认)
        self._out_dev = None                             # 选中的输出设备(None=系统默认)

    def _classify(self, midi):
        """现场演奏实时判定 —— 规则与 GestureInterpreter.process 一致,参数直接取自它:
        80ms 窗口内 ≥3 音 → strum;有按住音: |音程|≤legato_max_st → legato,更大 → slide;否则 pluck。"""
        now = time.perf_counter()
        self._onsets = [t for t in self._onsets if now - t < self._gi.strum_window]
        self._onsets.append(now)
        if len(self._onsets) >= self._gi.strum_min:
            return "strum", -1
        if getattr(self, "_voice_mode", "solo") == "poly":
            return "pluck", -1                       # POLY:重叠=和弦,不做连/滑(行业惯例 Poly/Legato 开关)
        frm = self._held[-1] if self._held else None
        if frm is not None and frm != midi:
            if abs(midi - frm) <= self._gi.legato_max_st:
                return "legato", frm
            return "slide", frm
        return "pluck", -1

    def _balance(self, inst):
        """把取色乐器的稳态谐波响度归一到基准(不同捕捉音量拉齐;起音颗粒同比缩放)。"""
        import numpy as np
        Hs = np.asarray(inst.H_steady, np.float64)
        norm = float(np.sqrt((Hs ** 2).sum()))
        if norm > 1e-6:
            g = self._ref_energy / norm
            inst.H_steady = (Hs * g).astype(np.float32)
            ag = np.asarray(getattr(inst, "attack_grain", []), np.float64)
            if ag.size:
                inst.attack_grain = (ag * g).astype(np.float32)
        return inst

    # ---- 生命周期 ---- #
    def start(self):
        """开实时音频流(sounddevice OutputStream,独立回调线程)+ 尝试接外接 MIDI。"""
        if not self._running:
            self.engine.start(queue=self._q)
            self._running = True
            self._open_midi()
            self._start_companion_api()
        return self._running

    def _start_companion_api(self):
        """Local Companion API(Sprint 1):TakeStore + 127.0.0.1 HTTP,供 Nexus Companion 拉取。
        失败(端口占用等)只降级不报错——Nexus 集成失败不得影响本地演奏(铁律)。"""
        if getattr(self, "_companion_srv", None):
            return
        try:
            from companion_api import TakeStore, serve
            self._takes = TakeStore(ROOT / "assets" / "takes")
            self._companion_srv = serve(
                self._takes, self._companion_status,
                render_fn=lambda notes, bpm, title:      # INK:Companion 送谱来渲染(同 stage 链路)
                    self.stage_take(notes, bpm=bpm, title=title))
        except Exception:
            self._companion_srv = None

    def _companion_status(self):
        import subprocess
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            commit = "unknown"
        return {"app": "quill", "version": f"git:{commit}",
                "timbre": self._last_disp, "engine": "L0-dsp",
                "running": self._running}

    def stage_take(self, notes, bpm=120.0, title=None):
        """SEND TO AUDIOTOOL 第一步:登记 take 并后台离线渲染(渲染完 audio.state=ready,
        Companion 轮询到即可上传)。立即返回 take_id —— 主线程/音频回调零阻塞。"""
        if not notes:
            return {"ok": False, "msg": "Nothing to send — record or draw notes first"}
        if getattr(self, "_companion_srv", None) is None:
            self._start_companion_api()
            if getattr(self, "_companion_srv", None) is None:
                return {"ok": False, "msg": f"Companion API failed (port {8723}?)"}
        title = title or f"{self._last_disp} take"
        dur = max(float(n["start"]) + float(n["len"]) for n in notes) + 1.5
        timbre = {"name": self._last_disp,
                  "f0_ref_hz": float(getattr(self.engine.instrument, "f0_ref", 0.0)),
                  "noise_level": float(getattr(self.engine.instrument, "noise_level", 0.0))}
        take = self._takes.stage(
            title=title, notes=notes, bpm=float(bpm), timbre=timbre,
            duration_s=dur, sr=self.engine.sr, provenance={
                "quill_version": self._companion_status()["version"],
                "engine": "L0-dsp", "atlas": None, "source_region": None})
        if take["audio"]["state"] == "ready":       # 幂等命中:已渲染过,直接可发
            return {"ok": True, "take_id": take["take_id"], "state": "ready"}

        def _render():
            try:
                import copy
                import io
                import numpy as np
                import soundfile as sf
                from runtime.engine import Engine
                inst = copy.copy(self.engine.instrument)
                eng = Engine(inst, eval_mode=True)
                y = eng.render_offline(self._take_events(notes), dur)
                y = np.asarray(y, np.float32)
                peak = float(np.max(np.abs(y))) if y.size else 0.0
                peak_db = 20.0 * np.log10(peak) if peak > 0 else -120.0
                buf = io.BytesIO()
                sf.write(buf, y, eng.sr, format="WAV")
                self._takes.attach_audio(take["take_id"], buf.getvalue(), peak_db)
            except Exception as e:
                self._takes.fail(take["take_id"], str(e))

        import threading
        threading.Thread(target=_render, daemon=True, name="take-render").start()
        return {"ok": True, "take_id": take["take_id"], "state": "pending"}

    def _open_midi(self, name=None):
        """打开外接 MIDI 输入(name=None 取第一个),消息直接进引擎队列(没设备就跳过)。"""
        try:
            from runtime.midi_in import MidiInput
            raw = self._raw_midi_name(name) if name else None   # UI 传显示名 → 还原原始名匹配端口
            m = MidiInput(port_name=raw)
            m.queue = _MidiRouter(self)                  # 外接 MIDI → 同一条演奏法判定路径
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
        (如'蓝牙'→'ËìùÁâô')。重编回 MacRoman 再按 UTF-8 解 = 还原;不是这种情况则原样返回
        (纯 ASCII 与真 UTF-8 名字都会因编/解码失败而保持不变,天然自保护)。"""
        if not isinstance(name, str):
            return name
        try:
            return name.encode("mac-roman").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return name

    def list_midi(self):
        try:
            import mido
            return [self._fix_midi_name(n) for n in mido.get_input_names()]   # 显示用:修乱码
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

    # ---- 设备面板(内置 I/O + MIDI 选择,真数据)---- #
    def list_devices(self):
        """枚举音频输入/输出 + MIDI + 引擎信息,给设置页渲染(真数据,非装饰)。"""
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
                seen_i.add(nm); ins.append({"idx": i, "name": nm, "sel": i == cur_in})
            if d.get("max_output_channels", 0) > 0 and nm not in seen_o:
                seen_o.add(nm); outs.append({"idx": i, "name": nm, "sel": i == cur_out})
        cur_midi = self.midi_name()
        midi = [{"name": n, "sel": n == cur_midi} for n in self.list_midi()]
        eng = {}
        try:
            lat = self.engine.stream_latency_ms()
            eng = {"sr": int(self.engine.sr), "blocksize": int(self.engine.blocksize),
                   "latency_ms": round(float(lat), 1) if lat == lat else None,
                   "voices": int(self.engine.max_voices)}
        except Exception:
            pass
        return {"input": ins, "output": outs, "midi": midi, "engine": eng}

    def set_input_device(self, idx):
        """选录音输入设备(下次取色生效;不影响正在进行的流)。"""
        self._in_dev = int(idx) if idx is not None else None
        return {"ok": True}

    def set_output_device(self, idx):
        """选输出设备 → 重启引擎输出流到新设备(失败回落系统默认,保音不断)。"""
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
            try:                                         # 新设备打不开 → 回落默认,别把声音搞哑
                self.engine.start(queue=self._q, device=None)
                self._out_dev = None
            except Exception:
                pass
            return {"ok": False, "msg": str(e)[:80]}

    def probe_input(self, ms=350):
        """短暂探测当前输入设备峰值电平(设备面板"测试"按钮:一眼看出选的设备有没有信号)。"""
        import sounddevice as sd
        import numpy as np
        import time
        buf = []
        try:
            st = sd.InputStream(samplerate=44100, channels=1, device=self._in_dev,
                                callback=lambda ind, f, t, s: buf.append(ind[:, 0].copy()))
            st.start(); time.sleep(max(0.1, min(1.0, ms / 1000.0))); st.stop(); st.close()
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        if not buf:
            return {"ok": True, "peak": 0.0}
        y = np.concatenate(buf)
        return {"ok": True, "peak": float(np.abs(y).max())}

    def set_midi_input(self, name):
        """接指定 MIDI 端口(name=None 关闭)。返回实际连上的端口名。"""
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

    def stop(self):
        self._close_cap_stream()
        srv = getattr(self, "_companion_srv", None)
        if srv:
            try:
                srv.shutdown()
            except Exception:
                pass
            self._companion_srv = None
        prev = getattr(self, "_stems_dir", None)
        if prev:
            import shutil
            shutil.rmtree(prev, ignore_errors=True)
            self._stems_dir = None
        if self._midi:
            try:
                self._midi.close()
            except Exception:
                pass
            self._midi = None
        if self._running:
            try:
                self.engine.stop()
            finally:
                self._running = False

    # ---- 演奏(JS / MIDI 调用)---- #
    def note_on(self, midi, vel=100, art="auto", prev=-1):
        """art="auto"(现场演奏)→ 引擎侧统一判定(gesture.py 语义);显式 art(卷帘回放)→ 照办。
        返回判定结果字符串,UI 用它点亮演奏法标签。"""
        midi, vel, prev = int(midi), int(vel), int(prev)
        if art == "auto":
            art, frm = self._classify(midi)
            if prev < 0:
                prev = frm                                # 滑音/连音起点 = 当前按住的音
            if midi in self._held:
                self._held.remove(midi)
            self._held.append(midi)
        if art == "slide" and prev >= 0:
            self._q.put(Event("slide", midi, vel, self._gi._slide_dur(vel), prev))   # 力度→滑速
        elif art == "legato" and prev >= 0:
            self._q.put(Event("legato", midi, vel, self._gi.legato_dur, prev))
        else:
            self._q.put(Event("on", midi, vel, 0.0, -1))            # pluck/strum = 普通起音
        return art

    def note_off(self, midi):
        m = int(midi)
        if m in self._held:
            self._held.remove(m)
        self._q.put(Event("off", m, 0, 0.0, -1))
        return True

    # ---- 卷帘回放:引擎采样级调度(消 JS setTimeout 抖动;与 click 同一时钟)---- #
    def play_notes(self, notes, bpm=120.0, metro=False, bpb=4):
        """整段乐句写进 engine._sched(绝对采样时间戳,实时回调按块消费)。
        notes: [{midi,start(秒),len,vel,art}];metro=True 时 click 与首音同一基准采样对齐。"""
        eng = self.engine
        sr = eng.sr
        base = int(getattr(eng, "_stream_samples", 0)) + int(0.05 * sr)   # 50ms 起播余量
        plan, midis, prev = [], set(), -1
        for nt in sorted(notes, key=lambda x: float(x["start"])):
            m, v = int(nt["midi"]), int(nt.get("vel", 100))
            art = nt.get("art", "pluck")
            t0 = base + int(float(nt["start"]) * sr)
            t1 = base + int((float(nt["start"]) + float(nt["len"])) * sr)
            if art == "slide" and prev >= 0:
                plan.append((t0, Event("slide", m, v, self._gi._slide_dur(v), prev)))
            elif art == "legato" and prev >= 0:
                plan.append((t0, Event("legato", m, v, self._gi.legato_dur, prev)))
            else:
                plan.append((t0, Event("on", m, v, 0.0, -1)))
            plan.append((t1, Event("off", m, 0, 0.0, -1)))
            midis.add(m); prev = m
        plan.sort(key=lambda x: x[0])
        self._play_midis = midis
        eng._sched_i = 0                                  # 先归零再换 plan:旧 plan 过期事件会被跳过
        eng._sched = plan
        if metro:
            eng.set_click(float(bpm), accent=int(bpb), start=base)
        return True

    def stop_playback(self):
        """取消未发事件 + 已排音符全部收音 + 关 click。"""
        eng = self.engine
        eng._sched = []
        eng.set_click(None)
        for m in getattr(self, "_play_midis", set()):
            self._q.put(Event("off", int(m), 0, 0.0, -1))
        import sounddevice as sd
        try:
            sd.stop()                                     # 神经渲染回放也一并停
        except Exception:
            pass
        return True

    # ---- 神经音色(小提琴 DDSP + 演奏法条件化检索,8/3 接入)---- #
    def _neural(self):
        """懒加载单例:首次调用才 import torch/载 checkpoint(不拖慢 app 启动)。"""
        if not hasattr(self, "_neural_take"):
            from neural.neural_take import get_neural_take
            self._neural_take = get_neural_take()
        return self._neural_take

    def play_notes_neural(self, notes, bpm=120.0, instrument="violin",
                          behavior=None, traj_gain=1.0,
                          vibrato_source="retrieval", randomness=0.3):
        """卷帘回放(神经版):离线渲染整段 → sd.play 独立小流(与引擎并行)。
        behavior: 行为源名称(内置乐器名或自定义库名);None=同 instrument。
        traj_gain: 轨迹强度 0.3–2.0。"""
        import sounddevice as sd
        try:
            y, sr = self._neural().render(
                notes, instrument=instrument, behavior=behavior,
                traj_gain=float(traj_gain),
                vibrato_source=vibrato_source, randomness=float(randomness))
        except Exception as e:
            return {"ok": False, "msg": str(e)[:80]}
        try:
            sd.play(y, sr, device=self._out_dev)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        return {"ok": True, "dur": round(len(y) / sr, 3)}

    def list_behavior_sources(self):
        """列出可用行为源(内置+用户导入)。"""
        return self._neural().list_behavior_sources()

    def import_style_from_wav(self, wav_path):
        """从 wav 提取行为库 → 导入到 user_styles。"""
        import subprocess, tempfile
        from pathlib import Path
        p = Path(wav_path)
        if not p.exists():
            return {"ok": False, "msg": "File not found"}
        name = p.stem
        with tempfile.TemporaryDirectory(prefix="quill_style_") as tmp:
            npz = Path(tmp) / f"{name}.npz"
            ROOT = Path(__file__).resolve().parents[1]
            cmd = ["/opt/anaconda3/envs/mt/bin/python",
                   str(ROOT / "tools" / "extract_style.py"),
                   str(p), "-o", str(npz)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return {"ok": False, "msg": (r.stderr or r.stdout)[:120]}
            if not npz.exists():
                return {"ok": False, "msg": "extract_style produced no output"}
            imported = self._neural().import_style(str(npz))
        return {"ok": True, "name": imported}

    def delete_behavior_source(self, name):
        """删除用户导入的行为库。"""
        self._neural().delete_style(name)
        return {"ok": True}

    # ---- 卷帘导出(7.4 用户点单):Save MIDI = 互操作硬证据;Export WAV = 离线渲染 ---- #
    def _take_events(self, notes):
        """卷帘音符 → 引擎 cooked 事件(秒;带演奏法,与 play_notes 同语义)。"""
        seq = sorted(notes, key=lambda x: float(x["start"]))
        evs, prev = [], -1
        for nt in seq:
            m, v = int(nt["midi"]), int(nt.get("vel", 100))
            art = nt.get("art", "pluck")
            t0 = float(nt["start"]); t1 = t0 + float(nt["len"])
            if art == "slide" and prev >= 0:
                evs.append((t0, "slide", m, v, self._gi._slide_dur(v), prev))
            elif art == "legato" and prev >= 0:
                evs.append((t0, "legato", m, v, self._gi.legato_dur, prev))
            else:
                evs.append((t0, "on", m, v, 0.0, -1))
            evs.append((t1, "off", m, 0, 0.0, -1))
            prev = m
        return evs

    def export_take(self, notes, bpm=120.0, fmt="wav", path=None):
        """卷帘整段导出:fmt='midi' 存标准 .mid(任何 DAW 可开);'wav' 用当前音色离线渲染;
        'wav_neural' 用神经音色(小提琴 DDSP)渲染。"""
        if str(fmt).startswith("wav_neural"):
            if not notes:
                return {"ok": False, "msg": "Nothing to export"}
            if not path:
                return {"ok": False, "msg": "No path"}
            try:
                import soundfile as sf
                inst = fmt.split(":", 1)[1] if ":" in fmt else "violin"   # "wav_neural:guitar"
                y, sr = self._neural().render(notes, instrument=inst)
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
                tr = mido.MidiTrack(); mid.tracks.append(tr)
                tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=0))
                evs = []
                for n in notes:
                    t0 = float(n["start"]); t1 = t0 + float(n["len"])
                    evs.append((t0, "note_on", int(n["midi"]), int(n.get("vel", 100))))
                    evs.append((t1, "note_off", int(n["midi"]), 0))
                evs.sort(key=lambda e: (e[0], e[1] == "note_on"))     # off 排 on 前(同刻换音安全)
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
                inst = copy.copy(self.engine.instrument)              # 当前音色,离线复刻
                eng = Engine(inst, eval_mode=True)
                evs = self._take_events(notes)
                dur = max(float(n["start"]) + float(n["len"]) for n in notes) + 1.5
                y = eng.render_offline(evs, dur)
                sf.write(str(path), np.asarray(y, np.float32), eng.sr)
            return {"ok": True, "path": str(path)}
        except Exception as e:
            return {"ok": False, "msg": str(e)[:80]}

    def click_on(self, bpm, bpb=4):
        """录制/预备拍的节拍器(引擎采样级,立即从当下起拍,第 1 拍重音)。"""
        self.engine.set_click(float(bpm), accent=int(bpb))
        return True

    def click_off(self):
        self.engine.set_click(None)
        return True

    # ---- 录制(Logic 式:录下所弹 MIDI → 卷帘显示)---- #
    def record_start(self):
        """开录:此后所有音符源(键盘/电脑键盘/外接 MIDI)带时间戳记下。"""
        self._q.arm()
        return True

    @staticmethod
    def _pair(log, end_t):
        """事件流 → 音符:on/slide/legato=起(演奏法保留!),off=止;仍按住的音延到 end_t。"""
        ART = {"on": "pluck", "slide": "slide", "legato": "legato"}
        notes, open_ = [], {}                            # open_: midi -> (t_on, vel, art)
        def close(m, t):
            t0, v0, a0 = open_.pop(m)
            notes.append({"midi": m, "start": round(t0, 4), "len": round(max(0.05, t - t0), 4),
                          "vel": v0, "art": a0})
        for t, ev in log:
            m = int(ev.note)
            if ev.kind in ART and m >= 0:
                if m in open_:                            # 同音重叠:先收尾前一个
                    close(m, t)
                open_[m] = (t, int(ev.vel) or 100, ART[ev.kind])
            elif ev.kind == "off" and m in open_:
                close(m, t)
        for m in list(open_):                            # 仍按住的音:延到 end_t
            close(m, end_t)
        notes.sort(key=lambda n: (n["start"], n["midi"]))
        return notes

    def record_poll(self):
        """录制中实时取当前已录音符 + 已录时长(秒),给卷帘实时显示 + 跟随滚动。"""
        if not self._q.rec:
            return {"notes": [], "t": 0.0}
        now = time.perf_counter() - self._q._t0
        return {"notes": self._pair(list(self._q.log), now), "t": round(now, 4)}

    def record_stop(self):
        """停录 → 配对成音符,take 对齐到第一个音。返回 [{midi,start,len,vel}](秒)。"""
        log = self._q.disarm()
        end_t = log[-1][0] if log else 0.0
        notes = self._pair(log, end_t)
        if notes and notes[0]["start"] > 0:              # 去掉第一个音前的空拍
            t0 = notes[0]["start"]
            for n in notes:
                n["start"] = round(n["start"] - t0, 4)
        return notes

    def pitch_bend(self, semitones):
        self._q.put(Event("bend", -1, 0, float(semitones), -1))
        return True

    def vibrato(self, depth01):
        self._q.put(Event("vibrato", -1, 0, float(depth01), -1))
        return True

    # ---- 旋钮(JS 调用)---- #
    def set_param(self, name, value):
        """旋钮实时改声(无需重建乐器的廉价参数)。"""
        v = float(value)
        self.params[name] = v
        eng = self.engine
        if name == "master":
            eng.master_gain = max(0.0, v * 1.4)               # 输出乘子(见 process_block)
        elif name == "noise":
            if eng.instrument is not None:                    # 噪声端比例:直接改,Voice 逐块读
                eng.instrument.noise_level = v
        elif name == "vibrate":
            eng.vib_rate_hz = 2.0 + v * 8.0                   # 2–10 Hz
        elif name == "vibdepth":
            eng.vib_max_cents = v * 80.0
            self._q.put(Event("vibrato", -1, 0, v, -1))       # 让按住的音立刻揉起来
        elif name == "strum":
            eng.set_gesture_templates()                       # 占位:strum 速度在 M3 演奏法里用
        # brightness / pick → 需 Voice 改造,M3 续接
        return True

    # ---- 取色(M4)---- #
    def _close_cap_stream(self):
        st = getattr(self, "_cap_stream", None)
        if st is not None:
            try:
                st.stop(); st.close()
            except Exception:
                pass
            self._cap_stream = None

    def capture_start(self):
        """开麦克风录音(独立 InputStream,与引擎输出并行)。"""
        import sounddevice as sd
        self._close_cap_stream()                 # 防双开:旧流未关就再开=原生流泄漏(7.4 终审)
        self._cap_frames = []
        self._cap_stream = sd.InputStream(samplerate=44100, channels=1,
                                          device=self._in_dev,   # 用户选中的麦克风(None=系统默认)
                                          callback=self._cap_cb)
        self._cap_stream.start()
        return True

    def _cap_cb(self, indata, frames, t, status):
        if len(self._cap_frames) * indata.shape[0] < 44100 * 65:   # 硬上限 65s(UI 60s 停的兜底)
            self._cap_frames.append(indata[:, 0].copy())

    def capture_level(self):
        """录制中实时电平(取色小窗电平表,~50ms RMS → 0..1)。"""
        fr = getattr(self, "_cap_frames", None)
        if not fr:
            return 0.0
        import numpy as np
        x = np.concatenate(fr[-3:]) if len(fr) > 1 else fr[-1]
        return float(min(1.0, np.sqrt(float((x ** 2).mean())) * 4.0))

    def capture_active(self):
        """录音流是否在跑(主窗 SOURCE 实时波形用;与取色小窗共享同一 bridge)。"""
        st = getattr(self, "_cap_stream", None)
        try:
            return bool(st is not None and st.active)
        except Exception:
            return False

    def capture_meter(self):
        """一次拿实时电平 + 是否在录(省一次 IPC;主窗 SOURCE 轮询)。"""
        return {"lvl": self.capture_level(), "active": self.capture_active()}

    def capture_stop(self):
        """停录 → analyze_timbre 取色 → 热切换当前乐器。返回色卡信息给 UI。"""
        import numpy as np
        try:
            self._cap_stream.stop(); self._cap_stream.close()
        except Exception:
            pass
        frames = getattr(self, "_cap_frames", [])
        y = np.concatenate(frames).astype(np.float32) if frames else np.zeros(4410, np.float32)
        if len(y) > 22050 and float(np.abs(y).max()) < 1e-5:
            # 全零输入:macOS 无麦权限 / 选了 BlackHole 等虚拟设备时,CoreAudio 静默给零不报错
            # (7.7 实战:系统输入被屏幕录制工具切到 BlackHole 2ch,23s 全零)→ 指名道姓可行动
            dev = "input"
            try:
                import sounddevice as sd
                dev = sd.query_devices(kind="input")["name"]
            except Exception:
                pass
            return {"ok": False, "msg": f"Silent input — '{dev}' gave no signal. "
                                        f"Pick your real mic: System Settings → Sound → Input."}
        return self._capture_audio(y, 44100, name="capture")

    def separate_stems(self, path):
        """Mode 3(从编曲提取):Demucs 6-stem 分离 → 写临时 wav,返回可选轨列表。
        用户从中挑一轨(吉他/pad/人声…)再走 capture_from_file 取色。CPU ~s 级。"""
        import tempfile
        import numpy as np
        import soundfile as sf
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            from analysis.io import load
        except Exception as e:
            return {"ok": False, "msg": f"Demucs unavailable: {str(e)[:50]}"}
        try:
            y, sr = load(str(path), sr=44100)
            y = y[: 44100 * 30]                          # 上限 30s(足够取色,控耗时)
            if self._sep_model is None:
                m = get_model("htdemucs_6s"); m.eval()
                self._sep_model = m
            model = self._sep_model
            wav = torch.tensor(np.asarray(y, np.float32))
            if wav.ndim == 1:
                wav = wav[None].repeat(2, 1)
            with torch.no_grad():
                out = apply_model(model, wav[None], device="cpu", progress=False)[0]
            prev = getattr(self, "_stems_dir", None)     # 磁盘卫生:上一次分离的临时 stems 删掉
            if prev:
                import shutil
                shutil.rmtree(prev, ignore_errors=True)
            outdir = Path(tempfile.mkdtemp(prefix="quill_stems_"))
            self._stems_dir = str(outdir)
            stems = []
            for i, name in enumerate(model.sources):
                x = out[i].mean(0).numpy()
                rms = float(np.sqrt((x ** 2).mean()))
                if rms < 1e-4:                           # 近静音轨不展示
                    continue
                p = outdir / f"{name}.wav"
                sf.write(str(p), x, 44100)
                stems.append({"name": name, "path": str(p), "rms": round(rms, 4)})
            stems.sort(key=lambda s: -s["rms"])
            return {"ok": True, "stems": stems}
        except Exception as e:
            return {"ok": False, "msg": str(e)[:80]}

    def capture_from_file(self, path):
        """拖入文件取色(同一条分析路径,可离线测)。"""
        from analysis.io import load
        y, sr = load(str(path), sr=44100)
        return self._capture_audio(y, sr, name=Path(path).stem)

    def set_voice_mode(self, mode="solo"):
        """声部模式(7.4 C2 验收问题,行业惯例):solo=重叠键触发连音/滑音(单声部表现);
        poly=重叠键各自发声(和弦)。默认 solo(与已验收行为一致)。"""
        self._voice_mode = "poly" if str(mode) == "poly" else "solo"
        return self._voice_mode

    def set_capture_mode(self, texture=False):
        """Texture 模式(用户 7.4 点单):专收乐音属性不强的声音(雨/风/机器/环境)。
        开启时跳过乐音守门与稳定窗,不降噪(纹理=噪声,不能洗),噪声层开大。默认关。"""
        self._texture_mode = bool(texture)
        return True

    def _capture_audio(self, y, sr, name="capture", f0_hint=None, keep_raw=True, gate=1.0):
        import warnings
        import numpy as np
        import soundfile as sf
        from analysis.io import trim_silence
        from analysis.timbre import analyze_timbre
        from synth.instrument import build_instrument
        tex = bool(getattr(self, "_texture_mode", False))
        USER_TONES.mkdir(parents=True, exist_ok=True)    # 取色黑匣子:存各阶段产物供杂音溯源
        y = np.asarray(y, np.float32)
        if keep_raw:                                     # 精修(refine)重分析时不覆盖原始件
            try:
                sf.write(str(USER_TONES / "_last_take_raw.wav"), y, sr)
            except Exception:
                pass
        y = trim_silence(y, sr)                          # P3 取色地板:剪首尾静音(呼吸/杂音多在此)
        if len(y) < int(0.4 * sr):
            return {"ok": False, "msg": "Too short — hold a steady note for 1s+"}
        if not tex and gate > 0:                         # Texture:纹理就是"噪声",不洗
            from analysis.denoise import spectral_gate
            y = spectral_gate(y, sr, strength=float(gate))   # 降噪档位:off=0 / auto=1 / strong=1.6(取色窗可调)
        try:
            sf.write(str(USER_TONES / "_last_take_gated.wav"), y, sr)
        except Exception:
            pass
        if tex and f0_hint is None:                      # 伪音高锚:给键盘一个映射基准
            seg = y[: int(4 * sr)].astype(np.float64)
            S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
            band = (fr > 60) & (fr < 1000)
            f0_hint = float(fr[band][np.argmax(S[band])]) if band.any() and S[band].max() > 0 else 220.0
            f0_hint = min(max(f0_hint, 60.0), 1000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # 静音 f0 交叉校验等分析警告(无害)
            # window="auto":自然录音(唱一句/哼一段)里自动挑出稳定音段取色——不强制单音
            snap = analyze_timbre(y[: int(6 * sr)] if tex else y, sr=sr, name=name,
                                  denoise=not tex, window=None if tex else "auto",
                                  f0_hint=f0_hint, cross_check=False)  # pyin 校验仅 warning,省 ~1s(refine 8va 为人工兜底)       # 用户修正标准音(八度错/对齐 A440)
        if not (np.isfinite(snap.get("f0_ref", 0)) and snap["f0_ref"] > 0):
            return {"ok": False, "msg": "No stable pitch — hum or sing a held note (a moving melody is fine, just hold one note somewhere)"}
        meta = snap.get("meta", {}) or {}
        warn = None
        if not tex:
            # 乐音守门(7.4 UAT,自动选窗后判):选窗成功=录音里有可取色的持续音段 → 放行;
            # 选窗失败(window_found=False)=纯噪声/快速说话/无持续音 → 此时 cv 仍高 → 拒。
            vr = float(meta.get("voiced_ratio", 1.0))
            cv = float(meta.get("f0_cv", 0.0))
            win_ok = meta.get("window_found", True)
            if not win_ok and (vr < 0.5 or cv > 0.15):   # 找不到任何稳定持续音
                if vr >= 0.85:                           # 音高满格但全程不稳 = 多半复音/整首歌
                    return {"ok": False, "msg": "Sounds like several notes at once — use ♫ From a song to pull out one instrument"}
                return {"ok": False, "msg": "No steady note to capture — hum or hold a single pitch for ~1 second (or turn on Texture mode)"}
            if cv > 0.09:                                # 软警:能用,但建议录得更稳
                warn = "Tip: hold a steadier note for a cleaner capture"
        try:                                             # 黑匣子:快照本体(杂音溯源用)
            from snapshot import save_snapshot
            save_snapshot(str(USER_TONES / "_last_capture.npz"), snap)
        except Exception:
            pass
        meta2 = dict(snap.get("meta", {}) or {})
        meta2["texture"] = tex                           # 持久化:保存/加载后噪声层不走样
        snap["meta"] = meta2
        inst = build_instrument(snap, noise_level=0.6 if tex else 0.02)   # 纹理住在噪声层;乐音默认 2%
        self._balance(inst)                              # 音量平衡:稳态响度拉齐到默认音色
        if not tex and "noise" in self.params:           # Air/Noise 旋钮位置在热切换后依然作数
            inst.noise_level = float(self.params["noise"])
        self.engine.set_instrument(inst)
        Hs = np.asarray(inst.H_steady, np.float64)
        idx = np.arange(1, len(Hs) + 1)                  # 谐波质心 → 亮度形容词
        centroid = float((idx * Hs).sum() / (Hs.sum() + 1e-9))
        adj = ("Texture" if tex else
               "Bright" if centroid > 8 else ("Warm" if centroid > 4 else "Mellow"))
        self._last_snap = snap                           # 留存以便用户命名保存
        self._last_disp = adj + " Capture"
        self._last_centroid = centroid
        take_s = round(len(y) / sr, 1)                    # 修剪后 take 时长(结果页文案用)
        win_used = bool((snap.get("meta", {}) or {}).get("window_found"))
        n_bins = 96                                       # SOURCE 波形:实际分析段的包络(真形状)
        L = max(1, len(y) // n_bins)
        wave = [float(np.sqrt((y[i * L:(i + 1) * L] ** 2).mean())) for i in range(n_bins)]
        wmax = max(wave) or 1.0
        wave = [round(v / wmax, 3) for v in wave]
        return {"ok": True, "name": name, "disp": adj + " Capture", "warn": warn, "wave": wave,
                "take_s": take_s, "win_used": win_used,
                "f0": round(float(snap["f0_ref"]), 1),
                "n_harm": int((Hs > Hs.max() * 0.01).sum()),
                "harm": [float(x) for x in (Hs / (Hs.max() + 1e-9))[:24]]}

    def play_take(self, t0=None, t1=None):
        """试听最近的 take(波形显示的那份音频=黑匣子 gated 件;t0/t1 秒=选区坐标)。
        sd.play 独立小流,与引擎输出并行(CoreAudio 混音),不进合成链。"""
        import sounddevice as sd
        import soundfile as sf
        p = USER_TONES / "_last_take_gated.wav"
        if not p.exists():
            return {"ok": False, "msg": "Nothing captured yet"}
        y, sr = sf.read(str(p), dtype="float32")
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        a = int(max(0.0, float(t0 or 0.0)) * sr)
        b = int(float(t1) * sr) if t1 else len(y)
        seg = y[a:b]
        if len(seg) < 100:
            return {"ok": False, "msg": "Empty span"}
        try:
            sd.play(seg, sr, device=self._out_dev)       # 跟随设备面板选的输出
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        return {"ok": True, "dur": round(len(seg) / sr, 3)}

    def stop_take(self):
        import sounddevice as sd
        try:
            sd.stop()
        except Exception:
            pass
        return True

    def refine_capture(self, f0_hint=None, lo_hz=None, hi_hz=None, t0=None, t1=None, denoise="auto"):
        """精修最近一次捕捉(用户旋钮,7.4 用户需求):
        f0_hint = 修正标准音(八度错纠正 / 吸附 A440 律网格);
        lo_hz/hi_hz = 低切/高切(先带通再分析——隆隆声/嘶声/串音时圈出主体);
        t0/t1 = 时间段(秒,对应修剪后 take 的波形显示坐标)——波形上拖选"只用这一段";
        denoise = 降噪档:auto(默认谱门控)/ off(留气声,不洗)/ strong(嘈杂环境)。
        数据源 = 黑匣子原始件,可反复精修不劣化。"""
        import numpy as np
        import soundfile as sf
        raw = USER_TONES / "_last_take_raw.wav"
        if not raw.exists():
            return {"ok": False, "msg": "Nothing captured yet"}
        y, sr = sf.read(str(raw), dtype="float32")
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if lo_hz or hi_hz:
            from scipy.signal import butter, sosfiltfilt
            lo = max(20.0, float(lo_hz or 20.0))
            hi = min(sr / 2 - 200.0, float(hi_hz or sr / 2 - 200.0))
            if hi > lo + 50:
                sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
                y = sosfiltfilt(sos, np.asarray(y, np.float64)).astype(np.float32)
        if t0 is not None or t1 is not None:             # 段选:按显示坐标(修剪后)切片
            from analysis.io import trim_silence
            y = trim_silence(y, sr)                      # 与取色链同一修剪 → 坐标对齐波形显示
            a = int(max(0.0, float(t0 or 0.0)) * sr)
            b = int(float(t1) * sr) if t1 else len(y)
            if b - a < int(0.35 * sr):
                return {"ok": False, "msg": "Selection too short — drag a longer span"}
            y = y[a:b]
        gate = {"off": 0.0, "auto": 1.0, "strong": 1.6}.get(str(denoise), 1.0)
        return self._capture_audio(y, sr, name="capture",
                                   f0_hint=(float(f0_hint) if f0_hint else None),
                                   keep_raw=False, gate=gate)

    # ---- 音色库:保存 / 列表 / 加载 / 删除 ---- #
    @staticmethod
    def _tone_css(c, name=""):
        """色卡 = iPhone 金属漆(银 / 金 / 玫瑰金),明亮通透带玻璃高光,不土不暗沉:
        暗音色→冷银钢,中→香槟金,亮→玫瑰金;RGB 在金属中性区插值(不经绿/紫/棕),叠 iOS 柔光高光。
        颜色只由音色亮度(centroid)决定——同一个声音无论叫什么名字,色卡都一样
        (7.7 用户:同名不同、其实声音相同 → 去掉名字哈希,让色卡诚实反映声音)。name 仅保留兼容签名。"""
        t = max(0.0, min(1.0, (float(c) - 2) / 12.0))
        lerp = lambda a, b, u: [a[i] + (b[i] - a[i]) * u for i in range(3)]
        s_t, g_t, r_t = (190, 206, 220), (226, 192, 110), (234, 168, 146)      # 顶部亮面:银·金·玫瑰金(更艳)
        s_b, g_b, r_b = (116, 150, 184), (196, 148, 40), (212, 106, 82)          # 底部实色(更艳)
        top, bot = ((lerp(s_t, g_t, t * 2), lerp(s_b, g_b, t * 2)) if t < 0.5
                    else (lerp(g_t, r_t, (t - .5) * 2), lerp(g_b, r_b, (t - .5) * 2)))
        hx = lambda rgb: "".join(f"{int(v):02x}" for v in rgb)
        return ("linear-gradient(180deg,rgba(255,255,255,.44),rgba(255,255,255,.06) 15%,transparent 34%),"  # 顶部镜面高光带
                "linear-gradient(0deg,rgba(40,30,14,.23),transparent 42%),"                                     # 底部压暗
                f"linear-gradient(160deg,#{hx(top)},#{hx(bot)})")                                              # 本体

    def list_tones(self):
        """默认音色 + 用户保存的音色,给前端音色库渲染。"""
        out = [{"name": "__default__", "disp": "Synth Pad", "builtin": True, "css": self._tone_css(6.0, "Synth Pad")}]
        try:
            from snapshot import load_snapshot
            for f in sorted(USER_TONES.glob("*.npz")):
                if f.stem.startswith("_"):               # 黑匣子内部件(_last_capture 等)不进音色库
                    continue
                disp, c = f.stem, 6.0
                try:
                    m = load_snapshot(str(f)).get("meta", {}) or {}
                    disp = m.get("disp", f.stem); c = float(m.get("centroid", 6.0))
                except Exception:
                    pass
                out.append({"name": f.stem, "disp": disp, "builtin": False, "css": self._tone_css(c, disp)})
        except Exception:
            pass
        return out

    def save_timbre(self, name):
        """把最近取色的音色命名保存到 user_tones/<name>.npz。"""
        import re
        from snapshot import save_snapshot
        if self._last_snap is None:
            return {"ok": False, "msg": "Capture a tone first"}
        safe = re.sub(r"[^\w\- ]", "", str(name or "")).strip()[:40] or "tone"
        snap = dict(self._last_snap)
        meta = dict(snap.get("meta", {}) or {})
        meta["disp"] = (str(name).strip() or safe); meta["centroid"] = self._last_centroid
        snap["meta"] = meta
        USER_TONES.mkdir(parents=True, exist_ok=True)
        save_snapshot(str(USER_TONES / safe), snap)
        return {"ok": True, "name": safe, "disp": meta["disp"]}

    def load_tone(self, name):
        """从库加载音色 → 热切换(含音量平衡)。返回色卡。"""
        import numpy as np
        from snapshot import load_snapshot
        from synth.instrument import build_instrument
        path = self._default_path if name == "__default__" else (USER_TONES / (str(name) + ".npz"))
        if not Path(path).exists():
            return {"ok": False, "msg": "Tone not found"}
        snap = load_snapshot(str(path))
        tex = bool((snap.get("meta", {}) or {}).get("texture"))
        inst = build_instrument(snap, noise_level=0.6 if tex else 0.02)   # 纹理音色噪声层随快照恢复
        self._balance(inst)
        if not tex and "noise" in self.params:            # Air/Noise 旋钮位置在换音色后依然作数
            inst.noise_level = float(self.params["noise"])
        self.engine.set_instrument(inst)
        self._last_snap = snap
        disp = "Synth Pad" if name == "__default__" else (snap.get("meta", {}) or {}).get("disp", name)
        self._last_disp = disp
        Hs = np.asarray(inst.H_steady, np.float64); idx = np.arange(1, len(Hs) + 1)
        self._last_centroid = float((idx * Hs).sum() / (Hs.sum() + 1e-9))
        return {"ok": True, "name": name, "disp": disp,
                "f0": round(float(snap.get("f0_ref", 0)) or 0, 1),
                "harm": [float(x) for x in (Hs / (Hs.max() + 1e-9))[:24]]}

    def rename_tone(self, name, new_name):
        """改名保存的音色:更新 meta.disp 并迁移 .npz 文件名(默认音色不可改)。"""
        import re
        from snapshot import load_snapshot, save_snapshot
        if name == "__default__":
            return {"ok": False, "msg": "Cannot rename default"}
        src = USER_TONES / (str(name) + ".npz")
        if not src.exists():
            return {"ok": False, "msg": "Tone not found"}
        safe = re.sub(r"[^\w\- ]", "", str(new_name or "")).strip()[:40]
        if not safe:
            return {"ok": False, "msg": "Empty name"}
        dst = USER_TONES / (safe + ".npz")
        if dst.exists() and dst != src:
            return {"ok": False, "msg": "Name already taken"}
        try:
            snap = load_snapshot(str(src))
            meta = dict(snap.get("meta", {}) or {})
            meta["disp"] = str(new_name).strip() or safe
            snap["meta"] = meta
            save_snapshot(str(USER_TONES / safe), snap)
            if dst != src:
                src.unlink()
            return {"ok": True, "name": safe, "disp": meta["disp"]}
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}

    def delete_tone(self, name):
        if name == "__default__":
            return {"ok": False, "msg": "Cannot delete default"}
        try:
            (USER_TONES / (str(name) + ".npz")).unlink()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}  # 色卡用

    def set_adsr(self, a, d, s, r):
        """ADSR 滑块(0–1)→ engine.adsr(影响下一个音);镜像 UI 手感。"""
        self.engine.adsr = {
            "attack_ms": 5.0 + float(a) * 1000.0,
            "decay_ms": 20.0 + float(d) * 1000.0,
            "sustain": float(s),
            "release_ms": 50.0 + float(r) * 1400.0,
        }
        return True
