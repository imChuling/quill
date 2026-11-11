"""Quill app 桥:UI 事件 → 实时引擎(runtime/engine)。

WebView 里的按键/旋钮通过 pywebview js_api 调这里,pywebview 在各自的线程上调用每个方法;
本类把事件 push 进引擎的 SimpleQueue(与 midi_in 同一路径),引擎在 sounddevice 的音频回调
线程里消费并发声。UI 侧的调用只入队、改标量或整体替换对象,从不在回调线程上做事。
功能按面拆成 mixin:设备、神经渲染、取色、导出、会话。
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.engine import Event  # noqa: E402
from runtime.gesture import GestureInterpreter  # noqa: E402  演奏法判定的单一事实来源
from ui.bridge_capture import BridgeCaptureMixin  # noqa: E402
from ui.bridge_device import BridgeDeviceMixin  # noqa: E402
from ui.bridge_export import BridgeExportMixin  # noqa: E402
from ui.bridge_neural import BridgeNeuralMixin  # noqa: E402
from ui.bridge_session import BridgeSessionMixin  # noqa: E402

DEFAULT_SNAPSHOT = ROOT / "assets" / "snapshots" / "synth_pad.npz"  # 温暖持续 pad,开场默认音色
USER_TONES = ROOT / "assets" / "user_tones"  # 用户保存的取色音色(.npz)


class _RecQueue:
    """事件队列代理:所有音符源(屏幕键盘/电脑键盘/外接 MIDI)都经此入队;
    录制中则给每个事件打时间戳记进 log。引擎只调 get_nowait(),透明转发。"""

    def __init__(self):
        self._q: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self.rec = False
        self._t0 = 0.0
        self.log = []  # [(t_sec, Event)]

    def arm(self):
        self.log = []
        self._t0 = time.perf_counter()
        self.rec = True

    def disarm(self):
        self.rec = False
        out = self.log
        self.log = []  # 停录即清,录制数据由返回值带走
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
            hook = getattr(self._b, "on_art", None)  # MIDI 路径:判定结果推回 UI 标签
            if hook and art:
                try:
                    hook(art)
                except Exception:
                    pass
        elif ev.kind == "off":
            self._b.note_off(ev.note)
        else:
            self._b._q.put(ev)  # bend/vibrato/pedal 直通


class QuillBridge(
    BridgeDeviceMixin,
    BridgeNeuralMixin,
    BridgeCaptureMixin,
    BridgeExportMixin,
    BridgeSessionMixin,
):
    """实时引擎桥:演奏事件、旋钮、音色热切换;其余功能面在各 mixin。"""

    def __init__(
        self, snapshot=DEFAULT_SNAPSHOT, noise_level=0.02
    ):  # 噪声层默认 2%,Noise 旋钮可调
        import threading

        import numpy as np

        from runtime.engine import Engine
        from snapshot import load_snapshot
        from synth.instrument import build_instrument

        self._state_lock = threading.Lock()
        self._q = _RecQueue()  # 录制代理队列(所有音符源汇合处)
        snap = load_snapshot(str(snapshot))
        inst = build_instrument(snap, noise_level=noise_level)
        # 默认音色的稳态能量是取色音色音量平衡的基准
        self._ref_energy = float(np.sqrt((np.asarray(inst.H_steady, np.float64) ** 2).sum())) or 0.3
        self.engine = Engine(inst, eval_mode=False)
        self.params = {}
        self._running = False
        self._midi = None
        self._default_path = Path(snapshot)  # 音色库:默认音色快照
        self._last_snap = None  # 最近取色/加载的快照(用于保存)
        self._last_centroid = 6.0
        self._gi = GestureInterpreter()  # 阈值/时长的唯一来源(legato_max_st、力度→滑速…)
        self._held = []  # 现场按住的键(按下顺序)
        self._onsets = []  # 最近 onset 时刻(strum 窗口)
        self._sep_model = None  # Demucs 惰性加载缓存
        self._in_dev = None  # 选中的录音输入设备(None=系统默认)
        self._out_dev = None  # 选中的输出设备(None=系统默认)

    def _classify(self, midi):
        """现场演奏的实时判定,规则与 GestureInterpreter.process 一致,参数直接取自它:
        80 ms 窗口内 ≥3 音 → strum;有按住音时 |音程| ≤ legato_max_st → legato,更大 → slide;
        否则 pluck。"""
        now = time.perf_counter()
        self._onsets = [t for t in self._onsets if now - t < self._gi.strum_window]
        self._onsets.append(now)
        if len(self._onsets) >= self._gi.strum_min:
            return "strum", -1
        if getattr(self, "_voice_mode", "solo") == "poly":
            return "pluck", -1  # poly:重叠即和弦,不做连音/滑音
        frm = self._held[-1] if self._held else None
        if frm is not None and frm != midi:
            if abs(midi - frm) <= self._gi.legato_max_st:
                return "legato", frm
            return "slide", frm
        return "pluck", -1

    @staticmethod
    def _safe_name(name, default="tone"):
        """用户起的资产名 → 文件名安全形式(一条规则:音色、行为库、烘焙都用它,同名才真同名)。"""
        import re

        return re.sub(r"[^\w\- ]", "", str(name or "")).strip()[:40] or default

    @staticmethod
    def _auto_noise_level(Hs):
        """取色默认噪声底按谱质心定:近正弦源(擦玻璃/口哨,质心 ≤1.2 个
        谐波)质感全在噪声层 → 30%;有谐波结构的源(人声/吉他,质心 ≥1.8)→ 2%;之间线性。"""
        import numpy as np

        hs = np.abs(np.asarray(Hs, np.float64))
        cen = float((np.arange(1, len(hs) + 1) * hs).sum() / (hs.sum() + 1e-12))
        u = min(1.0, max(0.0, (1.8 - cen) / 0.6))
        return round(0.02 + 0.28 * u, 3)

    def _balance(self, inst):
        """把取色乐器的稳态谐波响度归一到基准(不同捕捉音量拉齐;起音颗粒同比缩放)。"""
        import numpy as np

        Hs = np.asarray(inst.H_steady, np.float64)
        norm = float(np.sqrt((Hs**2).sum()))
        if norm > 1e-6:
            g = self._ref_energy / norm
            inst.H_steady = (Hs * g).astype(np.float32)
            ag = np.asarray(getattr(inst, "attack_grain", []), np.float64)
            if ag.size:
                inst.attack_grain = (ag * g).astype(np.float32)
        return inst

    # ---- 生命周期 ----
    def start(self):
        """开实时音频流(sounddevice OutputStream,独立回调线程)+ 尝试接外接 MIDI。"""
        if not self._running:
            self.engine.start(queue=self._q)
            self._running = True
            self._open_midi()
        return self._running

    def stop(self):
        self._close_cap_stream()
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

    # ---- 演奏(JS / MIDI 调用)----
    def note_on(self, midi, vel=100, art="auto", prev=-1):
        """art="auto"(现场演奏)→ 引擎侧统一判定(gesture.py 语义);显式 art(卷帘回放)→ 照办。
        返回判定结果字符串,UI 用它点亮演奏法标签。"""
        midi, vel, prev = int(midi), int(vel), int(prev)
        if art == "auto":
            art, frm = self._classify(midi)
            if prev < 0:
                prev = frm  # 滑音/连音起点 = 当前按住的音
            if midi in self._held:
                self._held.remove(midi)
            self._held.append(midi)
        if getattr(self, "_live_behavior", None) and getattr(self, "_live_ready", False):
            return self._note_on_live(midi, vel, art, prev)
        if art == "slide" and prev >= 0:
            self._q.put(Event("slide", midi, vel, self._gi._slide_dur(vel), prev))  # 力度→滑速
        elif art == "legato" and prev >= 0:
            self._q.put(Event("legato", midi, vel, self._gi.legato_dur, prev))
        elif art == "strum" and len(self._onsets) >= 2:
            self._later(self._strum_delay(), Event("on", midi, vel, 0.0, -1))
        else:
            self._q.put(Event("on", midi, vel, 0.0, -1))  # pluck = 普通起音
        return art

    def _note_on_live(self, midi, vel, art, prev):
        """行为模式的现场起音:检索轨迹 → ctl / ctl_legato 事件;检索失败退回规则路径。"""
        try:
            ctl = self._live_ctl(midi, vel, art, prev)
        except Exception as e:  # 实时路径不能因检索异常哑掉
            print(f"[bridge] live behavior fallback: {str(e)[:80]}")
            self._q.put(Event("on", midi, vel, 0.0, -1))
            return art
        self._live_on_t[midi] = time.perf_counter()
        if art in ("slide", "legato") and prev >= 0:
            ev = Event("ctl_legato", midi, vel, ctl, prev)  # 沿用上一音的声部,不重起音
        else:
            ev = Event("ctl", midi, vel, ctl, -1)
        if art == "strum" and len(self._onsets) >= 2:
            self._later(self._strum_delay(), ev)
        else:
            self._q.put(ev)
        return art

    def _strum_delay(self):
        """现场扫弦错峰:簇内第 k 音落在首音之后 k×spread 处(STRUM 旋钮),已过时刻为 0。"""
        k = len(self._onsets) - 1
        return max(0.0, self._onsets[0] + k * self._gi.strum_spread - time.perf_counter())

    def note_off(self, midi):
        m = int(midi)
        if m in self._held:
            self._held.remove(m)
        self._q.put(Event("off", m, 0, 0.0, -1))
        t_on = getattr(self, "_live_on_t", {}).pop(m, None)
        if t_on is not None:  # 行为模式:用实际按键时长更新时长先验(EMA),下一个音的检索更准
            d = time.perf_counter() - t_on
            if 0.05 < d < 4.0:
                self._live_dur = 0.6 * float(self._live_dur) + 0.4 * d
        return True

    # ---- 卷帘回放:引擎采样级调度,与 click 同一时钟 ----
    def play_notes(self, notes, bpm=120.0, metro=False, bpb=4):
        """整段乐句写进 engine._sched(绝对采样时间戳,实时回调按块消费)。
        notes: [{midi,start(秒),len,vel,art}];metro=True 时 click 与首音同一基准采样对齐。"""
        eng = self.engine
        sr = eng.sr
        base = int(getattr(eng, "_stream_samples", 0)) + int(0.05 * sr)  # 50ms 起播余量
        plan, midis = [], set()
        for t, kind, m, v, value, note2 in self._gi.notes_to_events(notes):
            plan.append((base + int(t * sr), Event(kind, m, v, value, note2)))
            midis.add(m)
        plan.sort(key=lambda x: x[0])
        self._play_midis = midis
        eng._sched_i = 0  # 先归零再换 plan:旧 plan 过期事件会被跳过
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
            sd.stop()  # 神经渲染回放也一并停
        except Exception:
            pass
        return True

    def vibrato(self, depth01):
        # 行为模式下也照发:Voice 在库轨迹走完前不叠加 LFO,走完后由它接力长音的揉弦;
        # 同时记下最近深度,供行为模式决定下一个音的 vibrato 标签(CC1 与旋钮同权)
        self._live_vib = float(depth01)
        self._q.put(Event("vibrato", -1, 0, float(depth01), -1))
        return True

    # ---- 实时弹奏接行为库 ----
    def set_live_behavior(self, name=None, wait=False):
        """现场弹奏改由行为库驱动:按下键 → 按(音高, 上一音, 标签)检索轨迹 → 控制流驱动
        Voice;时长未知先按最近几个音的时长(初值=库中位数)拉伸,末帧保持,松键走释放。
        name 为空 = 关,回到规则路径(GestureInterpreter)。
        载入 + 预热(首次 ~3s:import、checkpoint、JIT)在后台线程,就绪前按键走规则路径,
        不卡 UI;`live_behavior_status()` 轮询,wait=True 则同步等完(测试用)。"""
        import threading

        name = str(name or "").strip()
        self._live_ready = False
        self._live_error = None
        if not name:
            self._live_behavior = None
            return {"ok": True, "name": None, "ready": True}
        self._live_behavior = name

        def _warm():
            import numpy as np

            try:
                nn, env = self._neural().resolve_behavior(name)  # 与 NeuralTake.render 同一解析
                if hasattr(nn, "clone"):
                    nn = nn.clone()  # 与离线渲染不共享 last_idx(MIDI 线程 vs UI 线程)
                feats = getattr(nn, "features", None)
                med = float(np.median(feats[:, 1])) if feats is not None and len(feats) else 0.5
                self._live_src = (nn, env)
                self._live_dur = min(2.0, max(0.25, med))  # 时长先验:库中位数,之后按演奏 EMA
                self._live_on_t = {}
                self._live_ctl(60, 100, "pluck", -1)  # 预热:首次检索/滤波的 import 与 JIT
                self._live_ctl(62, 100, "legato", 60)
                with self._state_lock:
                    if self._live_behavior == name:  # 期间没被切走
                        self._live_ready = True
            except Exception as e:
                with self._state_lock:
                    self._live_error = f"behavior {name!r} unusable: {str(e)[:60]}"
                    if self._live_behavior == name:
                        self._live_behavior = None

        t = threading.Thread(target=_warm, daemon=True, name="live-behavior-warmup")
        t.start()
        if wait:
            t.join()
            if self._live_error:
                return {"ok": False, "msg": self._live_error}
            return {"ok": True, "name": name, "ready": True, "dur_prior": round(self._live_dur, 3)}
        return {"ok": True, "name": name, "ready": False}

    def live_behavior_status(self):
        return {
            "name": getattr(self, "_live_behavior", None),
            "ready": bool(getattr(self, "_live_ready", False)),
            "error": getattr(self, "_live_error", None),
        }

    def _live_art_label(self, art, prev):
        """现场音的标签:slide 来自手势判定;vibrato 看最近一次揉弦深度(旋钮或 CC1)>0.15;
        其余 plain。"""
        if art == "slide" and prev >= 0:
            return 2
        depth = getattr(self, "_live_vib", None)
        if depth is None:
            depth = float(self.params.get("vibdepth", 0.0))
        return 1 if depth > 0.15 else 0

    def _live_ctl(self, midi, vel, art, prev):
        """一个现场音 → (f0_hz[T], amp[T], fps)。有上一音(连音/滑音)时按两音乐句检索,
        取第二音的帧,使检索上下文是"句内换音"而不是"句首";否则按单音(句首起音)。"""
        import numpy as np

        from neural.perform import FPS, control_from_behavior

        nn, env = self._live_src
        # 检索时长:最近几个音的 EMA,但不低于 0.8 s。短音被松键截断没有损失,长音则有足够长的
        # 库轨迹(揉弦/起伏)可走;走完后 Voice 把揉弦交给引擎 LFO 接力。
        dur = max(0.8, float(self._live_dur))
        art_i = self._live_art_label(art, prev)
        score = []
        if prev >= 0 and art in ("slide", "legato"):
            score.append({"midi": int(prev), "dur": 0.2, "art": 0})
        score.append({"midi": int(midi), "dur": dur + 0.12, "art": art_i})  # 多 0.12s 再切掉句尾收弓
        f0, loud = control_from_behavior(
            score, nn, pitch_glide=False, envelope=env, traj_gain=float(getattr(self, "_live_gain", 1.0))
        )[:2]
        f0 = f0.numpy()[0, :, 0].astype(np.float64)
        loud = loud.numpy()[0, :, 0].astype(np.float64)
        if len(score) == 2:
            start = int(0.2 * FPS) - 5  # 第二音在控制流里的起点(减 20ms crossfade 重叠)
            f0, loud = f0[start:], loud[start:]
        keep = max(8, int(dur * FPS))
        f0, loud = f0[:keep], loud[:keep]
        # 末 100ms 把音高收回标称值:末帧保持时不能冻在揉弦的某个相位上(最多 ±30 cents 跑调)
        f0_nom = 440.0 * 2.0 ** ((int(midi) - 69) / 12.0)
        r = min(len(f0), int(0.1 * FPS))
        if r > 1:
            w = np.linspace(0.0, 1.0, r)
            f0[-r:] = np.exp(np.log(f0[-r:]) * (1.0 - w) + np.log(f0_nom) * w)
        from runtime.engine import loud_to_amp

        return np.ascontiguousarray(f0), loud_to_amp(loud), float(FPS)

    # ---- 旋钮(JS 调用)----
    def set_param(self, name, value):
        """旋钮实时改声,每个旋钮都作用于引擎:
        master/noise/vibrate/vibdepth/brightness/artint 改标量即时生效;
        strum/gsens 改手势判定器的时长;pick 要重整起音颗粒 → 防抖重建乐器。"""
        from synth.voice import set_articulation_intensity, set_tone_tilt

        v = float(value)
        self.params[name] = v
        eng = self.engine
        if name == "master":
            eng.master_gain = max(0.0, v * 1.4)  # 输出乘子(见 process_block)
        elif name == "noise":
            if eng.instrument is not None:  # 噪声端比例:直接改,Voice 逐块读
                eng.instrument.noise_level = v
        elif name == "vibrate":
            eng.vib_rate_hz = 2.0 + v * 8.0  # 2–10 Hz
        elif name == "vibdepth":
            eng.vib_max_cents = v * 80.0
            self._live_vib = v
            # 让按住的音立刻揉起来;行为模式下轨迹走完后由 LFO 接力
            self._q.put(Event("vibrato", -1, 0, v, -1))
        elif name == "brightness":
            set_tone_tilt((v - 0.5) * 12.0)  # ±6 dB/oct,与 UI 标注一致
        elif name == "artint":
            set_articulation_intensity(v * 2.0)  # 0.5 为默认强度,1.0 为两倍摩擦/凹陷
        elif name == "strum":
            self._gi.strum_spread = (8.0 + v * 70.0) / 1000.0  # 弦序时差 8–78 ms(UI 同式)
        elif name == "gsens":
            glide = (20.0 + v * 80.0) / 1000.0  # 连音滑程 20–100 ms(UI 同式)
            self._gi.legato_dur = glide
            self._gi.slide_dur = glide * 2.2  # 保持默认 130/60 的比例
            self._gi.slide_dur_min = glide * 0.9
        elif name == "pick":
            self._rebuild_instrument_later()
        return True

    def _rebuild_instrument(self):
        """按当前快照 + pick/noise 旋钮重建乐器并热切换(起音颗粒形状住在 build 阶段)。"""
        from snapshot import load_snapshot
        from synth.instrument import build_instrument

        snap = self._last_snap or load_snapshot(str(self._default_path))
        tex = bool((snap.get("meta", {}) or {}).get("texture"))
        inst = build_instrument(
            snap, noise_level=0.6 if tex else 0.02, pick=float(self.params.get("pick", 0.5))
        )
        self._balance(inst)
        if not tex and "noise" in self.params:
            inst.noise_level = float(self.params["noise"])
        with self._state_lock:
            self.engine.set_instrument(inst)

    def _rebuild_instrument_later(self, delay_s=0.06):
        """旋钮拖动时每秒几十次事件,只在停手 60ms 后重建一次(最新值赢)。"""
        import threading

        t = getattr(self, "_rebuild_timer", None)
        if t is not None:
            t.cancel()
        t = threading.Timer(delay_s, self._rebuild_instrument)
        t.daemon = True
        self._rebuild_timer = t
        t.start()

    def _later(self, delay_s, ev):
        """延迟投递一个引擎事件(strum 错峰用;Timer 抖动 ~1ms,远小于弦序时差)。"""
        import threading

        if delay_s <= 0.0:
            self._q.put(ev)
            return
        t = threading.Timer(delay_s, self._q.put, args=(ev,))
        t.daemon = True
        t.start()

