"""Quill 桌面 app(pywebview)—— M2:UI 键盘/旋钮 → 真实 Quill 引擎。

架构见 docs/app_plan.md:WebView 画 Claude Design 的 UI,Python 后端=真实实时引擎,
进程内 js_api 桥直连,音频走 sounddevice(无浏览器)。

M1:原生窗口加载 UI ✅
M2(本文件):Api 接 ui/bridge.QuillBridge,UI 的 noteOn/noteOff 调 api.note_on/off →
  真实 L0 取色音色实时发声;旋钮 → api.set_param。

运行:  python ui/app_web.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                          # 让 bridge 可导入
UI_HTML = ROOT / "web" / "quill.dc.html"               # 可编辑源(已接桥;需同目录 support.js)
CAPTURE_HTML = ROOT / "web" / "capture.html"           # 取色独立小窗(M4)


class Api:
    """js_api 桥 —— UI 通过 window.pywebview.api.* 调这些 Python 方法。"""

    def __init__(self):
        self.bridge = None
        self.err = None
        self.main_window = None                         # 主窗引用(取色后更新色卡)
        self.capture_window = None                       # 取色窗引用(保证只开一个)
        try:
            from bridge import QuillBridge
            self.bridge = QuillBridge()
            self.bridge.start()                        # 开实时音频流(sounddevice)
            self.bridge.on_art = self._push_art        # 外接 MIDI 的演奏法判定 → 主窗标签
        except Exception as e:                         # 音频起不来也别让窗口崩
            self.err = str(e)
            print("[Api] 引擎启动失败:", e)

    def _push_art(self, art):
        """外接 MIDI 判定结果推回主窗(mido 线程调用;evaluate_js 由 pywebview 编组)。"""
        if self.main_window is not None:
            import json
            try:
                self.main_window.evaluate_js(
                    f"window.quillShowArt&&window.quillShowArt({json.dumps(str(art))})")
            except Exception:
                pass

    def ping(self):
        return "quill-backend-ok" if self.bridge else f"backend-error:{self.err}"

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

    def play_notes_neural(self, notes, bpm=120.0, instrument="violin",
                          behavior=None, traj_gain=1.0,
                          vibrato_source="retrieval", randomness=0.3):
        """卷帘回放(神经音色):art 字段 = 演奏法开关(vibrato/slide/plain);
        instrument = timbre, behavior = style source, traj_gain = trajectory intensity。"""
        return self.bridge.play_notes_neural(
            notes, bpm, instrument, behavior, traj_gain,
            vibrato_source, randomness) if self.bridge \
            else {"ok": False, "msg": "Engine not started"}

    def list_behavior_sources(self):
        """列出可用行为源(内置乐器 + 用户导入的自定义库)。"""
        if not self.bridge:
            return []
        try:
            return self.bridge.list_behavior_sources()
        except Exception as e:
            return []

    def import_style_wav(self):
        """弹文件对话框选 wav → extract_style → 导入为行为源。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        try:
            import webview as wv
            win = self.main_window
            paths = win.create_file_dialog(wv.OPEN_DIALOG, allow_multiple=False,
                                           file_types=("Audio (*.wav;*.mp3;*.m4a;*.flac)",))
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

    def stage_take(self, notes, bpm=120.0):
        """SEND TO AUDIOTOOL:登记 take 到 Local Companion API(后台渲染),
        Nexus Companion(127.0.0.1:5173)轮询到 ready 即可上传+插入工程。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        return self.bridge.stage_take(notes, bpm=bpm)

    def export_take(self, notes, bpm=120.0, fmt="wav"):
        """卷帘导出:存盘对话框 → MIDI(.mid 标准文件)或 WAV(当前音色离线渲染)。"""
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
        return self.bridge.export_take(notes, bpm, fmt, path)

    def click_on(self, bpm, bpb=4):
        return self.bridge.click_on(bpm, bpb) if self.bridge else False

    def click_off(self):
        return self.bridge.click_off() if self.bridge else False

    def list_devices(self):
        return self.bridge.list_devices() if self.bridge else {"input": [], "output": [], "midi": [], "engine": {}}

    def set_input_device(self, idx):
        return self.bridge.set_input_device(idx) if self.bridge else {"ok": False}

    def set_output_device(self, idx):
        return self.bridge.set_output_device(idx) if self.bridge else {"ok": False}

    def set_midi_input(self, name):
        return self.bridge.set_midi_input(name) if self.bridge else {"ok": False}

    def probe_input(self, ms=350):
        return self.bridge.probe_input(ms) if self.bridge else {"ok": False}

    def get_status(self):
        """状态栏真数据:MIDI 设备名 + 实测输出流延迟(评委截图要求真数字,拒绝装饰值)。"""
        if not self.bridge:
            return {"midi": None, "latency_ms": None}
        try:
            lat = self.bridge.engine.stream_latency_ms()
            lat = round(float(lat), 1) if lat == lat else None      # NaN 防护
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

    # ---- 取色(M4:独立小窗)---- #
    def open_capture(self, mode=None):
        """主界面点 Capture → 弹出独立取色小窗。已开着就不再新建(只允许一个)。
        mode='song':打开后直接进 Mode 3(整歌选文件→分离),主窗 FROM A SONG 直达。"""
        if self.capture_window is not None:              # 已有取色窗:提到顶上,不新建
            try:
                self.capture_window.restore()            # 若最小化先还原
                self.capture_window.show()               # 置顶/聚焦
            except Exception:
                pass
            if mode == "song":
                try:
                    self.capture_window.evaluate_js("typeof fromSong==='function'&&fromSong()")
                except Exception:
                    pass
            return True
        win = webview.create_window(
            "Capture", str(CAPTURE_HTML),
            js_api=self, width=400, height=612, resizable=False,   # 612:容纳 NOISE 行 + Mode3 选轨区
            background_color="#15130e",
        )
        self.capture_window = win

        def _closed():
            self.capture_window = None                   # 关掉后清引用,可再次打开

        try:
            win.events.closed += _closed
        except Exception:
            pass
        if mode == "song":                               # 等页面加载完再触发选歌
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

    def _push_tone_name(self, r):
        """取色/换音色成功后同步主窗:TONE·COLOUR 名 + 命名框 + 顶部预设名 + SOURCE 真 f0。"""
        if r.get("ok") and self.main_window is not None:
            import json
            import math
            nm = json.dumps(r.get("disp") or r.get("name") or "Capture")
            js = (f"var e=document.getElementById('quillToneName'); if(e) e.textContent={nm};"
                  f"var i=document.getElementById('quillToneInput'); if(i) i.value={nm};"
                  f"var p=document.getElementById('quillPresetName'); if(p) p.textContent={nm};")
            f0 = r.get("f0")
            if f0:
                midi = int(round(69 + 12 * math.log2(float(f0) / 440.0)))
                names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                nn = f"{names[midi % 12]}{midi // 12 - 1}"
                js += (f"var s=document.getElementById('quillSrcF0');"
                       f" if(s) s.textContent='f\u2080 {float(f0):.1f} Hz \u00b7 {nn}';")
            wave = r.get("wave")
            if wave:
                js += (f"window._quillWave={json.dumps(wave)};"
                       f"window._quillWaveT0=performance.now();")
            # 动效触发:名字渐显 + 音色卡光扫(与取色动作同拍,连贯感的来源)
            js += ("var tn=document.getElementById('quillToneName');"
                   " if(tn){tn.classList.remove('nameSwap'); void tn.offsetWidth; tn.classList.add('nameSwap');}"
                   "var tc=document.getElementById('quillToneCard');"
                   " if(tc){tc.classList.remove('sweepGo'); void tc.offsetWidth; tc.classList.add('sweepGo');}")
            try:
                self.main_window.evaluate_js(js)
            except Exception:
                pass

    def capture_stop(self):
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        r = self.bridge.capture_stop()
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

    # ---- Mode 3:从编曲提取乐器(Demucs 分离 → 挑轨取色)---- #
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
        return self.bridge.capture_from_file(path) if self.bridge else {"ok": False}

    def refine_capture(self, f0_hint=None, lo_hz=None, hi_hz=None, t0=None, t1=None, denoise="auto"):
        """精修最近捕捉:标准音 / 低切高切 / 拖选段 / 降噪档 → 重分析 → 热切换 + 推动效。"""
        if not self.bridge:
            return {"ok": False, "msg": "Engine not started"}
        r = self.bridge.refine_capture(f0_hint, lo_hz, hi_hz, t0, t1, denoise)
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
        import base64, tempfile, os
        try:
            data = base64.b64decode(b64)
            ext = os.path.splitext(name)[1] or ".wav"
            fd, tmp = tempfile.mkstemp(suffix=ext, prefix="quill_drop_")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            r = self.capture_from_file(tmp)
            self._push_tone_name(r)
            return r
        except Exception as e:
            return {"ok": False, "msg": str(e)[:70]}

    # ---- 音色库(保存/列表/加载/删除)---- #
    def list_tones(self):
        return self.bridge.list_tones() if self.bridge else []

    def save_timbre(self, name):
        """保存音色;成功后通知主窗刷新 SAVED TONES(取色小窗保存时主窗同步)。"""
        r = self.bridge.save_timbre(name) if self.bridge else {"ok": False}
        if r.get("ok") and self.main_window is not None:
            try:
                self.main_window.evaluate_js("window.quillRefreshTones&&window.quillRefreshTones()")
            except Exception:
                pass
        return r

    def rename_tone(self, name, new_name):
        return self.bridge.rename_tone(name, new_name) if self.bridge else {"ok": False}

    def load_tone(self, name):
        if not self.bridge:
            return {"ok": False}
        r = self.bridge.load_tone(name)
        self._push_tone_name(r)                          # 名字/预设/SOURCE f0 一站式同步
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
        print(f"UI 缺失: {UI_HTML}"); return 1
    api = Api()
    # 设计是固定 1440×900 + 整体等比缩放(scaler):内容永远等比不变形,非 1.6 窗口留暗边(同背景色)。
    # 暂解锁让用户拉到顺眼尺寸;定了像素再改成 resizable=False 锁死。
    api.main_window = webview.create_window(
        "Quill", str(UI_HTML),
        js_api=api,
        width=1280, height=800, min_size=(960, 600),
        background_color="#15130e",
    )
    webview.start(_set_dock_icon)
    if api.bridge:
        api.bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
