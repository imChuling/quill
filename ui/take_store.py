"""On-disk repository of rendered takes: JSON metadata plus a WAV per take.

Used by ui/render_service.py: POST /render stages a take, renders it, attaches the WAV;
GET /takes/{id}/audio.wav serves it. Idempotent: take_id is a content hash of
(notes, bpm, timbre, title, variant), so re-rendering identical input returns the existing
take. Metadata writes are serialised by one lock; the WAV is written atomically because the
download path reads it without the lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


class TakeStore:
    """落盘 take 仓库:assets/takes/{id}.json + {id}.wav。线程安全(渲染线程写/HTTP 线程读)。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    @staticmethod
    def take_id_for(notes, bpm, timbre_name, title, variant="") -> str:
        """内容哈希。variant 收纳一切影响音频但不在上述字段里的渲染参数(如合成旋钮),
        否则改参数后的重渲会误命中幂等缓存。"""
        payload = json.dumps(
            [notes, float(bpm), timbre_name, title, variant], sort_keys=True, separators=(",", ":")
        )
        return "tk_" + hashlib.sha1(payload.encode()).hexdigest()[:12]

    def stage(
        self,
        *,
        title,
        notes,
        bpm,
        timbre,
        duration_s,
        sr,
        provenance,
        kind="performance",
        variant="",
    ) -> dict:
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
            # 而下载走的是无锁读,非原子写会让下载方拿到截断的 WAV。
            dst = self.root / f"{take_id}.wav"
            tmp = dst.with_suffix(f".wav.tmp{os.getpid()}")
            tmp.write_bytes(wav_bytes)
            os.replace(tmp, dst)
            take["audio"] = {
                "state": "ready",
                "wav_bytes": len(wav_bytes),
                "peak_dbfs": round(float(peak_dbfs), 1),
            }
            self._write(take_id, take)
            return take

    def set_duration(self, take_id: str, seconds: float):
        """渲染后回填乐句摆位后的真实时间轴长度(DAW 侧据此折算 musicDurationTicks)。"""
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
            take["audio"] = {
                "state": "failed",
                "wav_bytes": 0,
                "peak_dbfs": None,
                "error": str(msg)[:120],
            }
            self._write(take_id, take)

    # ----------------------------------------------------------------
    def list(self) -> list:
        with self._lock:
            metas = [
                self._read(p.stem)
                for p in sorted(
                    self.root.glob("tk_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
                )
            ]
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
