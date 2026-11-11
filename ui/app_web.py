"""Quill 桌面 app(pywebview):WebView 画 HTML UI,Python 进程内跑实时引擎。

Api 是 pywebview 的 js_api 对象:页面通过 window.pywebview.api.* 调它的方法,pywebview 在
各自的线程上调用每个方法;Api 只做参数转发与窗口管理,引擎逻辑在 ui/bridge.QuillBridge,
音频走 sounddevice。Python 反向推送走 main_window.evaluate_js,页面用 window.quill* 钩子
接收(quillShowArt、quillLiveBehavior、quillRefreshTones、quillTimbreSaved 等)。

运行:  python ui/app_web.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # 以脚本方式运行时把仓库根放进路径,ui/、runtime/ 等包才可导入
UI_HTML = ROOT / "web" / "quill.dc.html"  # 主窗页面,依赖同目录的 support.js
CAPTURE_HTML = ROOT / "web" / "capture.html"  # 取色独立小窗


class Api:
    """js_api 桥:UI 通过 window.pywebview.api.* 调这些 Python 方法。"""

    def __init__(self):
        self.bridge = None
        self.err = None
        self.main_window = None  # 主窗引用(取色后更新色卡)
        self.capture_window = None  # 取色窗引用(保证只开一个)
        try:
            from ui.bridge import QuillBridge

            self.bridge = QuillBridge()
            self.bridge.start()  # 开实时音频流(sounddevice)
            self.bridge.on_art = self._push_art  # 外接 MIDI 的演奏法判定 → 主窗标签
        except Exception as e:  # 音频起不来也别让窗口崩
            self.err = str(e)
            print("[Api] 引擎启动失败:", e)

    def _push_art(self, art):
        """外接 MIDI 判定结果推回主窗(mido 线程调用;evaluate_js 由 pywebview 编组)。"""
        if self.main_window is not None:
            import json

            try:
                self.main_window.evaluate_js(
                    f"window.quillShowArt&&window.quillShowArt({json.dumps(str(art))})"
                )
            except Exception:
                pass

    def note_on(self, midi, vel=110, art="auto", prev=-1):
        """返回判定出的演奏法字符串(auto 时),UI 拿它点亮标签。"""
        if self.bridge:
            return self.bridge.note_on(midi, vel, art, prev)
        return "pluck"

    def note_off(self, midi):
        if self.bridge:
            self.bridge.note_off(midi)
        return True

    def record_start(self):
        if self.bridge:
            self.bridge.record_start()
        return True

    def play_notes(self, notes, bpm=120.0, metro=False, bpb=4):
        """卷帘回放:引擎采样级调度(音符与 click 同一时钟)。"""
        return self.bridge.play_notes(notes, bpm, metro, bpb) if self.bridge else False

    def stop_playback(self):
        return self.bridge.stop_playback() if self.bridge else False

    def play_notes_neural(
        self,
        notes,
        bpm=120.0,
        instrument="violin",
        behavior=None,
        traj_gain=1.0,
        vibrato_source="retrieval",
        randomness=0.3,
        timing=0.0,
        continuity=0.0,
    ):
        """卷帘回放(神经音色):art 字段 = 演奏法开关(vibrato/slide/plain);
        instrument = timbre, behavior = style source, traj_gain = trajectory intensity。"""
        return (
            self.bridge.play_notes_neural(
                notes, bpm, instrument, behavior, traj_gain, vibrato_source, randomness,
                timing, continuity,
            )
            if self.bridge
            else {"ok": False, "msg": "Engine not started"}
        )

    def try_demo(self):
        """一键试听:glass × violin behavior,预渲染优先,回退到实时神经渲染。"""
        from pathlib import Path

        demo = Path(__file__).resolve().parent.parent / "assets" / "demos" / "ab_l0ctrl" / "B2b_l0_glass_x_violin_fixed.wav"
        if demo.exists():
            try:
                import sounddevice as sd
                import soundfile as sf

                y, sr = sf.read(str(demo), dtype="float32")
                out = getattr(self.bridge, "_out_dev", None) if self.bridge else None
                sd.play(y, sr, device=out)
                return {"ok": True, "source": "demo_wav", "dur": round(len(y) / sr, 2)}
            except Exception:
                pass
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        notes = [
            {"midi": 72, "start": 0, "len": 0.4, "vel": 100},
            {"midi": 74, "start": 0.5, "len": 0.3, "vel": 90},
            {"midi": 76, "start": 0.9, "len": 0.5, "vel": 95},
            {"midi": 79, "start": 1.5, "len": 0.8, "vel": 100},
            {"midi": 76, "start": 2.5, "len": 0.4, "vel": 85},
            {"midi": 74, "start": 3.0, "len": 0.3, "vel": 80},
            {"midi": 72, "start": 3.4, "len": 1.0, "vel": 100},
        ]
        return self.bridge.play_notes_neural(notes, 120.0, "glass", "violin")

    def list_instruments(self):
        """本机可用的神经音色名(UI 的 Neural 按钮据此循环)。"""
        if not self.bridge:
            return []
        try:
            return self.bridge.list_instruments()
        except Exception:
            return []

    def list_behavior_sources(self):
        """列出可用行为源(内置乐器 + 用户导入的自定义库)。"""
        if not self.bridge:
            return []
        try:
            return self.bridge.list_behavior_sources()
        except Exception:
            return []

    def import_style_wav(self):
        """弹文件对话框选 wav → extract_style → 导入为行为源。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            import webview as wv

            win = self.main_window
            paths = win.create_file_dialog(
                wv.OPEN_DIALOG,
                allow_multiple=False,
                file_types=("Audio (*.wav;*.mp3;*.m4a;*.flac)",),
            )
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}
        if not paths:
            return {"ok": False, "msg": "cancelled"}
        p = paths[0] if isinstance(paths, (list, tuple)) else paths
        return self.bridge.import_style_from_wav(p)

    def delete_behavior_source(self, name):
        if not self.bridge:
            return {"ok": False}
        return self.bridge.delete_behavior_source(name)

    def export_take(self, notes, bpm=120.0, fmt="wav", neural_opts=None):
        """卷帘导出:存盘对话框 → MIDI(.mid 标准文件)或 WAV(当前音色离线渲染)。
        neural_opts(仅 wav_neural):{behavior, traj_gain, vibrato_source, randomness}。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        if not notes:
            return {"ok": False, "msg": "Nothing to export — record or draw notes first"}
        try:
            import webview as wv

            fname = "quill_take." + ("mid" if fmt == "midi" else "wav")
            p = self.main_window.create_file_dialog(wv.SAVE_DIALOG, save_filename=fname)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:50]}
        if not p:
            return {"ok": False, "msg": "cancelled"}
        path = p if isinstance(p, str) else (p[0] if p else None)
        return self.bridge.export_take(notes, bpm, fmt, path, neural_opts)

    def click_on(self, bpm, bpb=4):
        return self.bridge.click_on(bpm, bpb) if self.bridge else False

    def click_off(self):
        return self.bridge.click_off() if self.bridge else False

    def list_devices(self):
        return (
            self.bridge.list_devices()
            if self.bridge
            else {"input": [], "output": [], "midi": [], "engine": {}}
        )

    def set_input_device(self, idx):
        return self.bridge.set_input_device(idx) if self.bridge else {"ok": False}

    def set_output_device(self, idx):
        return self.bridge.set_output_device(idx) if self.bridge else {"ok": False}

    def set_midi_input(self, name):
        return self.bridge.set_midi_input(name) if self.bridge else {"ok": False}

    def probe_input(self, ms=350):
        return self.bridge.probe_input(ms) if self.bridge else {"ok": False}

    def save_session(self, state):
        """UI 状态(旋钮/ADSR/模式/行为库/卷帘)落盘;UI 负责去抖,这里只校验+原子写。"""
        return self.bridge.save_session(state) if self.bridge else {"ok": False, "msg": "Engine not started"}

    def load_session(self):
        """上次会话;缺失/损坏返回 {}。"""
        return self.bridge.load_session() if self.bridge else {}

    def get_status(self):
        """状态栏数据:MIDI 设备名与实测的输出流延迟。"""
        if not self.bridge:
            return {"midi": None, "latency_ms": None}
        try:
            lat = self.bridge.engine.stream_latency_ms()
            lat = round(float(lat), 1) if lat == lat else None  # NaN 防护
        except Exception:
            lat = None
        return {"midi": self.bridge.midi_name(), "latency_ms": lat}

    def record_poll(self):
        """录制中实时取已录音符 + 时长(卷帘轮询用,实现录制跟随滚动)。"""
        return self.bridge.record_poll() if self.bridge else {"notes": [], "t": 0.0}

    def record_stop(self):
        """停录,返回录到的音符 [{midi,start,len,vel}](秒)给卷帘显示。"""
        return self.bridge.record_stop() if self.bridge else []

    def set_param(self, name, value):
        if self.bridge:
            self.bridge.set_param(name, value)
        return True

    def set_adsr(self, a, d, s, r):
        if self.bridge:
            self.bridge.set_adsr(a, d, s, r)
        return True

    # ---- 取色(独立小窗)----
    def open_capture(self, mode=None):
        """主界面点 Capture → 弹出独立取色小窗。已开着就不再新建(只允许一个)。
        mode='song':打开后直接进从歌曲提取(选文件 → 分离),对应主窗的 FROM A SONG。"""
        if self.capture_window is not None:  # 已有取色窗:提到顶上,不新建
            try:
                self.capture_window.restore()  # 若最小化先还原
                self.capture_window.show()  # 置顶/聚焦
            except Exception:
                pass
            if mode == "song":
                try:
                    self.capture_window.evaluate_js("typeof fromSong==='function'&&fromSong()")
                except Exception:
                    pass
            return True
        win = webview.create_window(
            "Capture",
            str(CAPTURE_HTML),
            js_api=self,
            width=400,
            height=612,
            resizable=False,  # 612:容纳 NOISE 行与选轨区
            background_color="#15130e",
        )
        self.capture_window = win

        def _closed():
            self.capture_window = None  # 关掉后清引用,可再次打开

        try:
            win.events.closed += _closed
        except Exception:
            pass
        if mode == "song":  # 等页面加载完再触发选歌

            def _loaded():
                try:
                    win.evaluate_js("typeof fromSong==='function'&&fromSong()")
                except Exception:
                    pass

            try:
                win.events.loaded += _loaded
            except Exception:
                pass
        return True

    def capture_start(self):
        return self.bridge.capture_start() if self.bridge else False

    def set_voice_mode(self, mode="solo"):
        return self.bridge.set_voice_mode(mode) if self.bridge else "solo"

    def set_live_behavior(self, name=None):
        """现场弹奏用哪个行为库驱动(空=关,规则路径)。载入在后台,就绪/失败时推给主窗
        (`window.quillLiveBehavior(name, ready, error)`),期间按键走规则路径。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            r = self.bridge.set_live_behavior(name)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:80]}
        if r.get("ok") and not r.get("ready"):
            import json
            import threading
            import time

            def _watch():
                for _ in range(300):  # 最多等 30s
                    st = self.bridge.live_behavior_status()
                    if st["ready"] or st["error"] or st["name"] != name:
                        break
                    time.sleep(0.1)
                if not st["ready"] and not st["error"] and st["name"] == name:
                    # 预热线程卡住(checkpoint 读不出来 / import 挂起):回到规则路径,
                    # 把超时当错误推回页面。
                    try:
                        self.bridge.set_live_behavior(None)
                    except Exception:
                        pass
                    st = {"name": name, "ready": False, "error": f"behavior {name!r} did not load in 30 s — back to rules"}
                self._refresh_main(
                    "window.quillLiveBehavior&&window.quillLiveBehavior("
                    f"{json.dumps(st['name'])},{json.dumps(bool(st['ready']))},{json.dumps(st['error'])})"
                )

            threading.Thread(target=_watch, daemon=True, name="live-behavior-watch").start()
        return r

    def live_behavior_status(self):
        return self.bridge.live_behavior_status() if self.bridge else {"name": None, "ready": False}

    def set_capture_mode(self, texture=False):
        """Texture 模式开关(雨/风/机器等非乐音取色)。"""
        return self.bridge.set_capture_mode(texture) if self.bridge else False

    def capture_level(self):
        """录制中实时电平(取色小窗电平表轮询)。"""
        return self.bridge.capture_level() if self.bridge else 0.0

    def play_take(self, t0=None, t1=None):
        """取色窗试听走带:播 take/选区(波形坐标对齐)。"""
        return self.bridge.play_take(t0, t1) if self.bridge else {"ok": False}

    def stop_take(self):
        return self.bridge.stop_take() if self.bridge else True

    def capture_meter(self):
        """主窗 SOURCE 实时波形:电平 + 是否在录(共享 bridge,取色小窗录音时主窗同步)。"""
        return self.bridge.capture_meter() if self.bridge else {"lvl": 0.0, "active": False}

    def _push_tone_name(self, r, captured=True):
        """取色/换音色成功后同步主窗:TONE·COLOUR 名 + 命名框 + 顶部预设名 + SOURCE 真 f0。

        captured=True 表示引擎现在弹的是一段未保存的新取色,页面的配对要跟过去;
        载入已保存音色时传 False,否则配对会被错标成 "(unsaved)"。"""
        if r.get("ok") and self.main_window is not None:
            import json
            import math

            nm = json.dumps(r.get("disp") or r.get("name") or "Capture")
            js = (
                f"var e=document.getElementById('quillToneName'); if(e) e.textContent={nm};"
                f"var i=document.getElementById('quillToneInput'); if(i) i.value={nm};"
            )
            f0 = r.get("f0")
            if f0:
                midi = int(round(69 + 12 * math.log2(float(f0) / 440.0)))
                names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                nn = f"{names[midi % 12]}{midi // 12 - 1}"
                js += (
                    f"var s=document.getElementById('quillSrcF0');"
                    f" if(s) s.textContent='f\u2080 {float(f0):.1f} Hz \u00b7 {nn}';"
                )
            wave = r.get("wave")
            if wave:
                js += f"window._quillWave={json.dumps(wave)};window._quillWaveT0=performance.now();"
            if r.get("noise") is not None:  # 自动噪声底 → Air/Noise 旋钮跟过去
                js += f"window.quillSetKnob&&window.quillSetKnob('noise',{float(r['noise']):.3f});"
            # 动效:名字渐显与音色卡光扫,与取色动作同拍
            js += (
                "var tn=document.getElementById('quillToneName');"
                " if(tn){tn.classList.remove('nameSwap'); void tn.offsetWidth; tn.classList.add('nameSwap');}"
                "var tc=document.getElementById('quillToneCard');"
                " if(tc){tc.classList.remove('sweepGo'); void tc.offsetWidth; tc.classList.add('sweepGo');}"
            )
            if captured:  # the engine now plays the (unsaved) capture: let the page's pair follow it
                js += f"window.quillCaptured&&window.quillCaptured({nm});"
            try:
                self.main_window.evaluate_js(js)
            except Exception:
                pass

    def capture_stop(self):
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            r = self.bridge.capture_stop()
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        self._push_tone_name(r)
        return r

    def open_file_capture(self):
        """OPEN FILE… 按钮:文件对话框 → 直接取色(单音/单乐器素材用;整歌走 pick_song)。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            import webview as wv

            win = self.capture_window or self.main_window
            paths = win.create_file_dialog(wv.OPEN_DIALOG, allow_multiple=False)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}
        if not paths:
            return {"ok": False, "msg": "No file chosen"}
        p = paths[0] if isinstance(paths, (list, tuple)) else paths
        r = self.capture_from_file(p)
        self._push_tone_name(r)
        return r

    # ---- 从歌曲提取乐器(Demucs 分离 → 挑轨取色)----
    def pick_song(self):
        """文件对话框选歌 → 分离 → 返回可选轨列表。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            import webview as wv

            win = self.capture_window or self.main_window
            paths = win.create_file_dialog(wv.OPEN_DIALOG, allow_multiple=False)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:60]}
        if not paths:
            return {"ok": False, "msg": "No file chosen"}
        p = paths[0] if isinstance(paths, (list, tuple)) else paths
        return self.bridge.separate_stems(p)

    def capture_stem(self, path):
        """从分离出的某一轨取色(键盘立刻弹它)。"""
        r = self.capture_from_file(path)
        self._push_tone_name(r)
        return r

    def capture_from_file(self, path):
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            return self.bridge.capture_from_file(path)
        except Exception as e:  # 非音频文件、解码失败:页面要拿到 msg,不能收到异常
            return {"ok": False, "msg": str(e)[:70]}

    def refine_capture(
        self, f0_hint=None, lo_hz=None, hi_hz=None, t0=None, t1=None, denoise="auto"
    ):
        """精修最近捕捉:标准音 / 低切高切 / 拖选段 / 降噪档 → 重分析 → 热切换 + 推动效。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            r = self.bridge.refine_capture(f0_hint, lo_hz, hi_hz, t0, t1, denoise)
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}
        self._push_tone_name(r)
        return r

    def capture_path(self, path):
        """拖拽文件取色(pywebview 给真实路径):分析 → 热切换 → 推动效。"""
        r = self.capture_from_file(path)
        self._push_tone_name(r)
        return r

    def capture_bytes(self, b64, name="drop"):
        """拖拽降级:前端传 base64 字节 → 写临时文件 → 取色。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        import base64
        import os
        import tempfile

        try:
            data = base64.b64decode(b64)
            ext = os.path.splitext(name)[1] or ".wav"
            fd, tmp = tempfile.mkstemp(suffix=ext, prefix="quill_drop_")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            try:
                r = self.capture_from_file(tmp)
            finally:
                os.unlink(tmp)
            self._push_tone_name(r)
            return r
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}

    # ---- 音色库(保存/列表/加载/删除)----
    def list_tones(self):
        return self.bridge.list_tones() if self.bridge else []

    def _refresh_main(self, js):
        if self.main_window is not None:
            try:
                self.main_window.evaluate_js(js)
            except Exception:
                pass

    def save_timbre(self, name):
        """保存音色;成功后通知主窗刷新 SAVED TONES 与卷帘 Neural 列表(新 tone: 入口)。"""
        r = self.bridge.save_timbre(name) if self.bridge else {"ok": False}
        if r.get("ok"):
            import json

            self._refresh_main("window.quillRefreshTones&&window.quillRefreshTones()")
            self._refresh_main("window.quillRefreshInstruments&&window.quillRefreshInstruments()")
            # saved from the capture window → the main window's pair follows the new tone
            self._refresh_main(f"window.quillTimbreSaved&&window.quillTimbreSaved({json.dumps(str(r.get('name')))},{json.dumps(bool((r.get('behavior') or {}).get('state') == 'extracting'))})")
            self._beh_announced = False
        return r

    def behavior_status(self):
        """保存音色后同名行为库的后台提取进度;完成时刷新卷帘的行为源列表。"""
        if not self.bridge:
            return {"state": "idle"}
        d = self.bridge.behavior_status()
        if d.get("state") == "done" and not getattr(self, "_beh_announced", False):
            import json

            self._beh_announced = True
            res = d.get("result") or {}
            n = int(res.get("n_notes", 0) or 0) if res.get("ok") else 0
            self._refresh_main("window.quillBehaviorsChanged&&window.quillBehaviorsChanged()")
            # one recording → two assets: tell the main window so it can offer the cross-use right away
            self._refresh_main(f"window.quillAssetsNew&&window.quillAssetsNew({json.dumps(str(d.get('name') or ''))},{n})")
        return d

    def bake_ddsp(self, name, base=None, steps=1000):
        """取色后烘焙 DDSP 音色(后台子进程);进度用 bake_status 轮询。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        return self.bridge.bake_ddsp(name, base or None, steps)

    def bake_status(self):
        if not self.bridge:
            return {"state": "idle"}
        d = self.bridge.bake_status()
        if d.get("state") == "done" and not getattr(self, "_bake_announced", False):
            self._bake_announced = True
            self._refresh_main("window.quillRefreshTones&&window.quillRefreshTones()")
            self._refresh_main("window.quillRefreshInstruments&&window.quillRefreshInstruments()")
        elif d.get("state") != "done":
            self._bake_announced = False
        return d

    def rename_tone(self, name, new_name):
        return self.bridge.rename_tone(name, new_name) if self.bridge else {"ok": False}

    def load_tone(self, name):
        if not self.bridge:
            return {"ok": False}
        r = self.bridge.load_tone(name)
        self._push_tone_name(r, captured=False)  # 名字/预设/SOURCE f0 一站式同步;不是新取色
        return r

    def delete_tone(self, name):
        return self.bridge.delete_tone(name) if self.bridge else {"ok": False}


def _set_dock_icon():
    """macOS Dock 图标 = 品牌 Q 印章(assets/brand);非 macOS / 无 pyobjc 时静默跳过。"""
    icon = ROOT.parent / "assets" / "brand" / "quill_icon_1024.png"
    if not icon.exists():
        return
    try:
        from AppKit import NSApplication, NSImage

        img = NSImage.alloc().initWithContentsOfFile_(str(icon))
        if img:
            NSApplication.sharedApplication().setApplicationIconImage_(img)
    except Exception:
        pass


def main():
    if not UI_HTML.exists():
        print(f"UI 缺失: {UI_HTML}")
        return 1
    api = Api()
    # 页面按固定 1440×900 设计并整体等比缩放:内容不变形,非 1.6 比例的窗口留同色暗边。
    api.main_window = webview.create_window(
        "Quill",
        str(UI_HTML),
        js_api=api,
        width=1280,
        height=800,
        min_size=(960, 600),
        background_color="#15130e",
    )
    webview.start(_set_dock_icon)
    if api.bridge:
        api.bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
