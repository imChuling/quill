"""TakeStore + Local Companion API —— Sprint 1(Nexus MVP)。

契约 = docs/take_schema_v1.md(冻结 2026-07-29):
  GET /status                     Quill 版本/当前音色/引擎状态
  GET /takes                      可发送 takes(元数据,新→旧)
  GET /takes/{id}                 单个 take 元数据
  GET /takes/{id}/audio.wav       WAV 字节(ready 前 409)

铁律(冲刺本子 §4):只监听 127.0.0.1;客户端永不提交本地路径;OAuth token
不进本进程;渲染在后台线程的引擎副本上跑(eval_mode),不碰实时 callback;
Companion 崩溃/断网不影响 Quill 演奏(本模块是纯供给侧)。

幂等:stage 的 take_id = sha1(notes+bpm+timbre+title)[:12] —— 重复点击返回同一 take。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_ALLOWED_ORIGINS = {"http://127.0.0.1:5173", "http://localhost:5173"}
API_PORT = 8723                      # 冻结;companion 侧同值


class TakeStore:
    """落盘 take 仓库:assets/takes/{id}.json + {id}.wav。线程安全(渲染线程写/HTTP 线程读)。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- #
    @staticmethod
    def take_id_for(notes, bpm, timbre_name, title, variant="") -> str:
        """内容哈希。variant 收纳一切影响音频但不在上述字段里的渲染参数
        (如合成旋钮)——漏掉它会让改参数后的重渲误命中幂等缓存。"""
        payload = json.dumps([notes, float(bpm), timbre_name, title, variant],
                             sort_keys=True, separators=(",", ":"))
        return "tk_" + hashlib.sha1(payload.encode()).hexdigest()[:12]

    def stage(self, *, title, notes, bpm, timbre, duration_s, sr,
              provenance, kind="performance", variant="") -> dict:
        """登记一个 take(audio.state=pending)。幂等:同内容返回既有 take。"""
        tid = self.take_id_for(notes, bpm, timbre.get("name", ""), title, variant)
        with self._lock:
            existing = self._read(tid)
            if existing is not None:
                return existing
            take = {
                "take_id": tid,
                "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "kind": kind,
                "title": title,
                "duration_s": round(float(duration_s), 3),
                "sample_rate": int(sr),
                "bpm": float(bpm),
                "timbre": timbre,
                "behavior": {"source": None, "provenance": None},
                "notes": notes,
                "provenance": provenance,
                "audio": {"state": "pending", "wav_bytes": 0, "peak_dbfs": None},
            }
            self._write(tid, take)
            return take

    def attach_audio(self, take_id: str, wav_bytes: bytes, peak_dbfs: float):
        with self._lock:
            take = self._read(take_id)
            if take is None:
                return None
            # 原子落盘:take_id 是内容哈希,并发的同内容渲染会写同一路径,
            # 而下载走的是无锁读 —— 非原子写会让下载方拿到截断的 WAV。
            dst = self.root / f"{take_id}.wav"
            tmp = dst.with_suffix(f".wav.tmp{os.getpid()}")
            tmp.write_bytes(wav_bytes)
            os.replace(tmp, dst)
            take["audio"] = {"state": "ready", "wav_bytes": len(wav_bytes),
                             "peak_dbfs": round(float(peak_dbfs), 1)}
            self._write(take_id, take)
            return take

    def set_duration(self, take_id: str, seconds: float):
        """渲染后回填真实时间轴长度(companion 据此折算 musicDurationTicks)。"""
        with self._lock:
            take = self._read(take_id)
            if take is None:
                return None
            take["duration_s"] = round(float(seconds), 3)
            self._write(take_id, take)
            return take

    def fail(self, take_id: str, msg: str):
        with self._lock:
            take = self._read(take_id)
            if take is None:
                return
            take["audio"] = {"state": "failed", "wav_bytes": 0,
                             "peak_dbfs": None, "error": str(msg)[:120]}
            self._write(take_id, take)

    # ---------------------------------------------------------------- #
    def list(self) -> list:
        with self._lock:
            metas = [self._read(p.stem) for p in sorted(
                self.root.glob("tk_*.json"),
                key=lambda p: p.stat().st_mtime, reverse=True)]
        return [m for m in metas if m]

    def get(self, take_id: str):
        with self._lock:
            return self._read(take_id)

    def wav_path(self, take_id: str) -> Path:
        return self.root / f"{take_id}.wav"

    def _read(self, tid):
        p = self.root / f"{tid}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _write(self, tid, take):
        (self.root / f"{tid}.json").write_text(json.dumps(take, ensure_ascii=False, indent=1))


# -------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    store: TakeStore = None           # serve() 注入
    status_fn = staticmethod(lambda: {})
    render_fn = None                  # serve() 注入:render_fn(notes,bpm,title)->{ok,take_id,state}

    def log_message(self, *a):        # 静音默认访问日志(音频 app 的 stdout 要干净)
        pass

    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin in _ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        """POST /render(take_schema_v1 §4):notes+bpm+title → 登记渲染,返回 take_id。
        校验:notes 非空/midi∈[0,127]/len>0/总时长≤120s;不接受任何路径字段。"""
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if parts != ["render"]:
            return self._json(404, {"ok": False, "code": "not_found"})
        if self.render_fn is None:
            return self._json(503, {"ok": False, "code": "busy", "msg": "renderer not attached"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            notes = body.get("notes") or []
            bpm = float(body.get("bpm", 120.0))
            title = str(body.get("title", "") or "Inked region")[:80]
            if not notes:
                return self._json(400, {"ok": False, "code": "bad_request", "msg": "empty notes"})
            end = 0.0
            for nt in notes:
                m = int(nt["midi"]); s = float(nt["start"]); ln = float(nt["len"])
                if not (0 <= m <= 127) or ln <= 0 or s < 0:
                    return self._json(400, {"ok": False, "code": "bad_request",
                                            "msg": f"bad note midi={m}"})
                end = max(end, s + ln)
            if end > 120.0:
                return self._json(400, {"ok": False, "code": "bad_request", "msg": ">120s"})
            clean = [{"midi": int(nt["midi"]), "start": float(nt["start"]),
                      "len": float(nt["len"]), "vel": int(nt.get("vel", 100)),
                      "art": str(nt.get("art", "pluck"))} for nt in notes]
            r = self.render_fn(clean, bpm, title)
            return self._json(200 if r.get("ok") else 500, r)
        except (KeyError, TypeError, ValueError) as e:
            return self._json(400, {"ok": False, "code": "bad_request", "msg": str(e)[:60]})
        except Exception as e:
            return self._json(500, {"ok": False, "code": "render_failed", "msg": str(e)[:60]})

    def do_GET(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        try:
            if parts == ["status"]:
                return self._json(200, self.status_fn())
            if parts == ["takes"]:
                return self._json(200, {"takes": self.store.list()})
            if len(parts) == 2 and parts[0] == "takes":
                take = self.store.get(parts[1])
                return self._json(200, take) if take else self._json(404, {"ok": False, "code": "not_found"})
            if len(parts) == 3 and parts[0] == "takes" and parts[2] == "audio.wav":
                take = self.store.get(parts[1])
                if take is None:
                    return self._json(404, {"ok": False, "code": "not_found"})
                if take["audio"]["state"] != "ready":
                    return self._json(409, {"ok": False, "code": "busy",
                                            "msg": take["audio"]["state"]})
                data = self.store.wav_path(parts[1]).read_bytes()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self._json(404, {"ok": False, "code": "not_found"})
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._json(500, {"ok": False, "code": "internal", "msg": str(e)[:80]})
            except Exception:
                pass


def serve(store: TakeStore, status_fn, port: int = API_PORT, render_fn=None):
    """127.0.0.1 起本地 API(daemon 线程),返回 server(可 .shutdown())。端口占用返回 None。"""
    handler = type("H", (_Handler,), {"store": store, "status_fn": staticmethod(status_fn),
                                      "render_fn": staticmethod(render_fn) if render_fn else None})
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        return None
    threading.Thread(target=srv.serve_forever, daemon=True,
                     name="quill-companion-api").start()
    return srv
