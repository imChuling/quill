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
    """On-disk take store: assets/takes/{id}.json + {id}.wav. Thread-safe (render thread writes / HTTP thread reads)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    @staticmethod
    def take_id_for(notes, bpm, timbre_name, title, variant="") -> str:
        """Content hash. variant collects every render parameter that affects the audio but is not in the fields above (e.g. synth knobs),
        otherwise a rerender after changing parameters would wrongly hit the idempotent cache."""
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
        """Register a take (audio.state=pending). Idempotent: the same content returns the existing take."""
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
            # atomic write: take_id is a content hash, concurrent renders of the same content write the same path,
            # and downloads use lock-free reads, so a non-atomic write could give the downloader a truncated WAV.
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
        """After rendering, fill in the real timeline length after phrase placement (the DAW side derives musicDurationTicks from it)."""
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
