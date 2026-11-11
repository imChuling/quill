"""实时音频引擎：块基声部池与事件调度。

`process_block` 是实时回调与离线渲染共用的唯一代码路径：`render_offline` 不开声卡，
喂事件列表跑同一函数，是单测、回归与试听刺激音的生成基座。

实时路径的约束：
  演奏模式下 gc.freeze() + gc.disable()，stop 时恢复；
  事件带采样时间戳，块内按采样偏移分段触发，strum 的 5–15 ms 时差经得起块边界；
  loudness/f0 块内线性斜坡、包络尾清零在 voice 内；
  母线 nan_to_num + tanh 软限幅；
  回调内不分配内存、不加锁、不做 IO，UI 与音频线程之间只走队列和原子引用替换；
  热切换乐器 = 原子重绑 instrument 引用；
  回调最外层 try/except，出错则静音并置错误标志；numba 预热防首音 xrun。

本模块只依赖 numpy 和 numba；mido 只在 midi_in.py。
"""

from __future__ import annotations

import gc
import os
from collections import namedtuple

import numpy as np

from synth.oscillator import warmup as _osc_warmup
from synth.voice import (
    _REL,
    Voice,
    regrain_legato,
    set_humanize,
    set_liveness,
    set_nyquist_ramp_hz,
    set_vibrato_shape,
)

try:
    from quill_config import CFG
except Exception:  # pragma: no cover
    CFG = None

# 事件：kind in {"on","off","bend","cc","slide","legato","vibrato","ctl","retrig"}
#   slide/legato: note=目标音, note2=起始音(从其声部续奏), value=时长秒
#   vibrato: value=深度 0-1
#   ctl: 起音并进入控制驱动, value=(f0_hz[T], amp[T], fps);retrig: 该声部重放起音颗粒
Event = namedtuple("Event", ["kind", "note", "vel", "value", "note2"], defaults=[-1])


LOUD_GAIN = 0.4  # z-score 响度 → 线性幅度的动态范围（±2σ ≈ ±7 dB）


def loud_to_amp(loud, gain: float = LOUD_GAIN):
    """z-score log-RMS 响度（neural/perform.py 的口径）→ 线性幅度，峰值归一到 1。

    离线 render_control 与实时行为模式共用；动态范围只在这一处定义。
    """
    loud = np.asarray(loud, np.float64)
    return np.exp(float(gain) * (loud - loud.max()))


def _runtime_cfg(cfg):
    cfg = cfg or {}
    rt = cfg.get("runtime", {})
    return {
        "blocksize": int(rt.get("blocksize", 256)),
        "max_voices": int(rt.get("max_voices", 16)),
        "channels": int(rt.get("channels", 1)),
        "noise_level": float(rt.get("noise_level", 0.5)),
        "attack_fade_ms": float(rt.get("attack_fade_ms", 20.0)),
        "steal_fade_ms": float(rt.get("steal_fade_ms", 4.0)),
        "nyquist_ramp_hz": float(rt.get("nyquist_ramp_hz", 600.0)),
    }


class Engine:
    def __init__(self, instrument, cfg=CFG, eval_mode: bool = False):
        self.instrument = instrument  # 热切换时原子重绑
        self.sr = int(instrument.sr)
        self.eval_mode = bool(eval_mode)
        c = cfg if cfg is not None else {}
        self.rt = _runtime_cfg(c)
        self.adsr = dict(
            c.get(
                "adsr", {"attack_ms": 8.0, "decay_ms": 120.0, "sustain": 0.7, "release_ms": 180.0}
            )
        )
        self.vel_cfg = dict(c.get("velocity", {"loud_exp": 1.5, "tilt_db_oct": 6.0}))

        self.blocksize = self.rt["blocksize"]
        self.max_voices = self.rt["max_voices"]
        K = instrument.K
        self.pool = [Voice(K, self.sr) for _ in range(self.max_voices)]
        self._note2voice = {}
        self._age = 0
        # 通道级弯音状态（半音）：新 note_on 必须继承，否则推轮时每个新音先从原始音高
        # chirp 到弯音位
        self._bend_st = 0.0
        # 通道级揉弦深度（0–1）：控制驱动的声部起音时继承，库轨迹走完后 LFO 接力
        self._vib_st = 0.0
        self.error_flag = False

        set_nyquist_ramp_hz(self.rt["nyquist_ramp_hz"])
        _syn = c.get("synth", {}) if isinstance(c, dict) else {}
        set_humanize(_syn.get("humanize_cents", 6.0), _syn.get("humanize_amp", 0.05))
        set_liveness(
            _syn.get("liveness_spec", 0.85),
            _syn.get("liveness_dark", 0.0),
            _syn.get("liveness_breath", 0.8),
        )
        _ges = c.get("gesture", {}) if isinstance(c, dict) else {}
        set_vibrato_shape(_ges.get("vibrato_onset_ms", 110.0), _ges.get("vibrato_ramp_ms", 220.0))

        # 预分配：回调内不分配内存
        self._buf = np.zeros(self.blocksize, dtype=np.float64)

        # 手势模板：slide 曲线与颤音速率，可由 .tplib 注入
        self.slide_curve = None  # g(τ)∈[0,1]，None 为线性
        # CC1 颤音速率优先取源片段实测值（snapshot.vibrato.rate_hz），否则默认
        self.vib_rate_hz = float(getattr(instrument, "vib_rate", None) or 5.5)
        self.vib_max_cents = 50.0  # CC1=1 时的最大颤音深度
        # CC64 延音踏板
        self._pedal_down = False
        self._pedal_sustained = set()

        # 事件调度
        self._sched = []  # 离线：[(t_samp, Event)]，升序
        self._sched_i = 0
        self._queue = None  # 实时：midi_in 注入的 SimpleQueue
        self._stream = None
        self._stream_samples = 0
        self._rng = np.random.default_rng(0)

        _osc_warmup()  # JIT 预热，防首音 xrun

    def set_instrument(self, inst):
        """热切换乐器：CPython 引用赋值是原子的，回调线程读到的要么是旧的要么是新的。"""
        self.instrument = inst

    def _alloc_voice(self) -> Voice:
        """空闲声部优先；否则偷最旧的释放中声部；再否则偷最旧的。"""
        for v in self.pool:
            if not v.active:
                return v
        released = [v for v in self.pool if v.stage == _REL]
        cand = released if released else self.pool
        return min(cand, key=lambda v: v.age)

    def _noise_pos0(self):
        if self.eval_mode:
            return 0
        tlen = len(self.instrument.noise_table)
        return int(self._rng.integers(0, tlen)) if tlen > 0 else 0

    def _apply_event(self, ev: Event):
        if ev.kind in ("on", "ctl"):
            v = self._alloc_voice()
            # 若偷的是仍在响的音，先解绑它的 note 映射
            if v.active and v.note in self._note2voice and self._note2voice[v.note] is v:
                self._note2voice.pop(v.note, None)
            self._age += 1
            v.note_on(
                ev.note,
                ev.vel,
                self.instrument,
                self.adsr,
                self.vel_cfg,
                self.rt,
                self._age,
                self._noise_pos0(),
            )
            v.init_bend(self._bend_st)  # 继承通道弯音，瞬时到位
            if ev.kind == "ctl":
                v.drive(*ev.value)
                if self._vib_st > 0.0:  # 库轨迹走完后由 LFO 接力揉弦，走完前 Voice 门控不叠加
                    v.set_vibrato(self._vib_st * self.vib_max_cents, self.vib_rate_hz)
            self._note2voice[ev.note] = v
        elif ev.kind == "retrig":
            v = self._note2voice.get(ev.note)
            if v is not None and v.active:
                v.retrigger_attack()
        elif ev.kind == "ctl_legato":
            # 连音换到新的控制流：沿用 note2 的声部（不重置相位/包络），低电平重触发颗粒
            v = self._note2voice.get(ev.note2)
            if v is None or not v.active:
                self._apply_event(Event("ctl", ev.note, ev.vel, ev.value, -1))
                return
            v.drive(*ev.value)
            v.retrigger_attack(regrain_legato())  # 与规则路径的连音颗粒同一倍率
            v.note = int(ev.note)
            self._note2voice.pop(ev.note2, None)
            self._note2voice[ev.note] = v
        elif ev.kind == "off":
            v = self._note2voice.pop(ev.note, None)
            if v is not None and v.note == ev.note:
                if self._pedal_down:  # 踏板按下：延音，推迟释放
                    self._pedal_sustained.add(v)
                else:
                    v.note_off()
        elif ev.kind == "pedal":  # CC64 延音踏板
            down = float(ev.value) >= 0.5
            if self._pedal_down and not down:  # 抬起：释放所有被踏板延音的声部
                for v in self._pedal_sustained:
                    if v.active:
                        v.note_off()
                self._pedal_sustained.clear()
            self._pedal_down = down
        elif ev.kind == "bend":
            self._bend_st = float(ev.value)  # 通道级状态，供新音继承
            tbd = getattr(self, "_trace_bend", None)
            if tbd is not None:  # 诊断：记录弯音事件（采样时刻，半音）
                tbd.append((self._stream_samples, float(ev.value)))
            for v in self.pool:
                if v.active:
                    v.set_bend(ev.value)
        elif ev.kind in ("slide", "legato"):
            self._glide_event(ev)
        elif ev.kind == "vibrato":  # CC1 → 颤音深度
            self._vib_st = max(0.0, float(ev.value))
            depth = self._vib_st * self.vib_max_cents
            for v in self.pool:
                if v.active:
                    v.set_vibrato(depth, self.vib_rate_hz)

    def _glide_event(self, ev):
        """slide/legato：从 note2 的声部续奏滑到 note，不重起音。"""
        v = self._note2voice.get(ev.note2)
        if v is None or not v.active:
            # 无可续奏的起始音，退化为普通起音
            self._apply_event(Event("on", ev.note, max(1, int(ev.vel or 90)), 0.0))
            return
        f0_tgt = 440.0 * 2.0 ** ((ev.note - 69) / 12.0)
        dur_s = float(ev.value) if ev.value else (0.04 if ev.kind == "legato" else 0.12)
        curve = self.slide_curve if ev.kind == "slide" else None
        v.glide_to(
            f0_tgt, int(dur_s * self.sr), curve=curve, kind=ev.kind, vel=int(ev.vel or 90)
        )
        v.note = int(ev.note)
        self._note2voice.pop(ev.note2, None)
        self._note2voice[ev.note] = v

    def _render_segment(self, buf, a, b):
        if b <= a:
            return
        seg = buf[a:b]
        nlen = b - a
        for v in self.pool:
            if v.active:
                v.render_into(seg, nlen)

    def _events_for_block(self, block_start, n):
        """返回 [(offset_in_block, Event)]，按 offset 升序。"""
        out = []
        if self._queue is not None:  # 实时：本块到达的事件即时触发
            while True:
                try:
                    ev = self._queue.get_nowait()
                except Exception:
                    break
                out.append((0, ev))
        # 离线：从有序调度里取落在 [start, start+n) 的
        while self._sched_i < len(self._sched):
            t, ev = self._sched[self._sched_i]
            if t < block_start:
                self._sched_i += 1
                continue
            if t >= block_start + n:
                break
            out.append((t - block_start, ev))
            self._sched_i += 1
        out.sort(key=lambda x: x[0])
        return out

    def process_block(self, out, n, block_start):
        """核心：渲染 n 个采样到 out[:n]（float32）。实时与离线共用。"""
        buf = self._buf
        try:
            buf[:n] = 0.0
            events = self._events_for_block(block_start, n)
            seg = 0
            for offset, ev in events:
                offset = 0 if offset < 0 else (n if offset > n else offset)
                if offset > seg:
                    self._render_segment(buf, seg, offset)
                self._apply_event(ev)  # 事件在块内偏移处生效，采样级精度
                seg = offset
            if seg < n:
                self._render_segment(buf, seg, n)
            np.nan_to_num(buf[:n], copy=False)
            buf[:n] *= getattr(self, "master_gain", 1.0)  # 总音量旋钮，默认 1.0
            ck = getattr(self, "_click", None)  # 节拍器，与音符同一采样时钟
            if ck is not None:
                self._render_click(buf, n, block_start, ck)
            np.tanh(buf[:n], out=buf[:n])  # 母线软限幅
            out[:n] = buf[:n]
        except Exception:  # 异常不杀流：静音并置错误标志
            out[:n] = 0.0
            self.error_flag = True

    # ---- 实时（需要声卡与 sounddevice）----
    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            self.error_flag = True
        self.process_block(outdata[:, 0], frames, self._stream_samples)
        if outdata.shape[1] > 1:
            outdata[:, 1:] = outdata[:, :1]
        tb = getattr(self, "_trace_buf", None)
        if tb is not None:  # 诊断：抓真实送声卡的音频，会破坏零分配，仅调试用
            tb.append(outdata[:, 0].copy())
        self._stream_samples += frames

    def start(self, queue=None, device=None):
        """打开输出流并进入演奏模式：gc.freeze() + gc.disable()，stop 时恢复。"""
        import sounddevice as sd

        self._queue = queue
        self._stream_samples = 0
        self.error_flag = False
        if os.environ.get("QUILL_BEND_TRACE"):  # 诊断模式：抓实机音频与弯音事件
            self._trace_buf = []
            self._trace_bend = []
            print("[BEND TRACE] armed — 关窗(或 Ctrl+C)后写 assets/user_tones/_bend_trace.*")
            if not getattr(self, "_trace_atexit", False):  # 非正常关闭也能存
                import atexit

                atexit.register(self._dump_trace)
                self._trace_atexit = True
        else:
            self._trace_buf = None
            self._trace_bend = None
        gc.collect()
        try:
            gc.freeze()
        except Exception:
            pass
        gc.disable()
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.blocksize,
            channels=self.rt["channels"],
            dtype="float32",
            callback=self._audio_callback,
            device=device,
        )
        self._stream.start()
        return self._stream

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._dump_trace()
        gc.enable()

    def _dump_trace(self):
        tb = getattr(self, "_trace_buf", None)
        if not tb:
            return
        try:
            import json

            y = np.concatenate(tb).astype(np.float32)
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "assets", "user_tones"
            )
            os.makedirs(out_dir, exist_ok=True)
            wav_path = os.path.join(out_dir, "_bend_trace.wav")
            try:
                import soundfile as sf

                sf.write(wav_path, y, self.sr)
            except Exception:  # 无 soundfile 时退回标准库，16-bit PCM
                import wave

                pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2")
                with wave.open(wav_path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self.sr)
                    w.writeframes(pcm.tobytes())
            with open(os.path.join(out_dir, "_bend_trace.json"), "w") as f:
                json.dump({"sr": self.sr, "blocksize": self.blocksize, "bend": self._trace_bend}, f)
            print(
                f"[BEND TRACE] {len(y) / self.sr:.1f}s audio + "
                f"{len(self._trace_bend)} bend events -> assets/user_tones/_bend_trace.*"
            )
        except Exception as e:
            print(f"[BEND TRACE] dump failed: {e}")
        finally:
            self._trace_buf = None
            self._trace_bend = None

    def stream_latency_ms(self):
        if self._stream is not None:
            return float(self._stream.latency) * 1000.0
        return float("nan")

    # ---- 离线渲染（不开声卡，与实时共用 process_block）----
    def reset(self):
        for v in self.pool:
            v.active = False
            v._reset_runtime()
        self._note2voice.clear()
        self._age = 0
        self._sched = []
        self._sched_i = 0
        self._stream_samples = 0
        self.error_flag = False
        self._pedal_down = False
        self._pedal_sustained = set()
        self._bend_st = 0.0
        self._vib_st = 0.0
        self._rng = np.random.default_rng(0)

    def set_gesture_templates(self, slide_curve=None, vib_rate_hz=None):
        """从 .tplib 注入手势模板：slide 曲线与颤音速率。"""
        if slide_curve is not None:
            self.slide_curve = np.asarray(slide_curve, dtype=np.float64)
        if vib_rate_hz:
            self.vib_rate_hz = float(vib_rate_hz)

    # ---- 节拍器：采样级 click，与音符同一时钟 ----
    def set_click(self, bpm=None, accent=4, gain=0.45, start=None):
        """开/关节拍器（bpm=None 关）。

        tick 是约 5 ms 的衰减正弦，波形在此预生成，回调内零分配；每 accent 拍的第 1 拍为
        重音。start 为起始绝对采样，None 为当下。
        """
        if not bpm:
            self._click = None
            return
        period = int(round(self.sr * 60.0 / float(bpm)))
        t = np.arange(int(self.sr * 0.005), dtype=np.float64) / self.sr
        burst = np.exp(-t / 0.0015)
        hi = (np.sin(2 * np.pi * 2093.0 * t) * burst * gain).astype(np.float32)
        lo = (np.sin(2 * np.pi * 1568.0 * t) * burst * gain * 0.7).astype(np.float32)
        now = int(getattr(self, "_stream_samples", 0))
        nxt = max(int(start), now) if start is not None else now
        self._click = {
            "period": period,
            "accent": max(1, int(accent)),
            "next": nxt,
            "beat": 0,
            "hi": hi,
            "lo": lo,
        }

    def _render_click(self, buf, n, block_start, ck):
        """把落在本块的 tick 加进 buf。tick 只有 5 ms，跨块尾部截断可忽略。"""
        end = block_start + n
        while ck["next"] < end:
            off = ck["next"] - block_start
            wav = ck["hi"] if ck["beat"] % ck["accent"] == 0 else ck["lo"]
            if off >= 0:
                L = min(len(wav), n - off)
                if L > 0:
                    buf[off : off + L] += wav[:L]
            ck["next"] += ck["period"]
            ck["beat"] += 1

    def render_offline(self, events_sec, dur_s: float) -> np.ndarray:
        """喂 (t_sec, kind, note, vel, value[, note2]) 事件列表，离线渲染 -> float32。

        与实时回调跑同一 process_block；eval_mode=True 时噪声指针复位，输出可复现。
        slide/legato 事件第 6 项 note2 为起始音。
        """
        self.reset()
        self._queue = None
        sched = []
        for e in events_sec:
            t_sec, kind = e[0], e[1]
            note = e[2] if len(e) > 2 else -1
            vel = e[3] if len(e) > 3 else 0
            value = e[4] if len(e) > 4 else 0.0
            note2 = e[5] if len(e) > 5 else -1
            sched.append((int(round(t_sec * self.sr)), Event(kind, note, vel, value, note2)))
        sched.sort(key=lambda x: x[0])
        self._sched = sched

        total = int(round(dur_s * self.sr))
        bs = self.blocksize
        nblocks = (total + bs - 1) // bs
        out = np.zeros(nblocks * bs, dtype=np.float32)
        for bi in range(nblocks):
            start = bi * bs
            self.process_block(out[start : start + bs], bs, start)
        return out[:total]


    def render_control(
        self,
        f0_hz,
        loud,
        fps: float = 250.0,
        onsets_s=(),
        vel: int = 100,
        loud_gain: float = LOUD_GAIN,
        tail_s: float = 0.3,
        bright=None,
        noise=None,
        motion: float = 1.0,
    ) -> np.ndarray:
        """(f0, loudness) 控制流 → 音频，与 DDSP 解码器同一接口，行为库可以直接驱动取色音色。

        loud 是演奏法层的 z-score log-RMS（见 neural/perform.py），按 amp = exp(loud_gain·loud)
        还原成线性幅度并归一到峰值 1；loud_gain 决定动态范围（0.4 时 ±2σ ≈ ±7 dB）。
        onsets_s 处重放起音颗粒。bright/noise 是行为库的第 3/4 维（log2 质心残差 oct、
        log10 平坦度残差）：质心残差 × 6 dB/oct 进谱倾斜，平坦度残差进噪声层倍率；motion
        整体缩放，0 为关，只用 f0/loudness。
        """
        f0_hz = np.asarray(f0_hz, np.float64)
        amp = loud_to_amp(loud, loud_gain)
        note = int(round(69.0 + 12.0 * np.log2(max(float(f0_hz[0]), 1e-3) / 440.0)))
        T = len(f0_hz)
        ctl = (f0_hz, amp, float(fps))
        if motion > 0 and bright is not None and noise is not None:
            tilt = np.asarray(bright, np.float64) * 6.0 * float(motion)  # oct → dB/oct
            nz = 10.0 ** (np.asarray(noise, np.float64) * float(motion))  # log10 → 倍率
            ctl = ctl + (tilt, nz)
        evs = [(0.0, "ctl", note, int(vel), ctl, -1)]
        evs += [(float(t), "retrig", note, 0, 0.0, -1) for t in onsets_s if float(t) > 1e-3]
        evs.append((T / float(fps), "off", note, 0, 0.0, -1))
        saved = self.adsr
        # 包络交给控制流：起音即到位、不衰减；note_off 后 80 ms 收尾
        self.adsr = {"attack_ms": 3.0, "decay_ms": 1.0, "sustain": 1.0, "release_ms": 80.0}
        try:
            return self.render_offline(evs, T / float(fps) + float(tail_s))
        finally:
            self.adsr = saved


def engine_from_snapshot_file(path, eval_mode=True, noise_level=None, cfg=CFG):
    """便捷：快照文件 -> Instrument -> Engine。"""
    from snapshot import load_snapshot
    from synth.instrument import build_instrument

    snap = load_snapshot(path)
    nl = (
        (CFG or {}).get("runtime", {}).get("noise_level", 0.5)
        if noise_level is None
        else noise_level
    )
    inst = build_instrument(snap, noise_level=nl)
    return Engine(inst, cfg=cfg, eval_mode=eval_mode)
