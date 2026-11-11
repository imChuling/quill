"""Render service: FastAPI implementation of the take_schema_v1 contract.

Render backend for the web page (web_demo/); can run locally or hosted:

  GET  /status                 version/available timbres/behaviour sources
  POST /capture                one recording -> timbre snapshot + behaviour library (two assets)
  GET  /assets                 list of captured assets
  POST /render                 notes+bpm+title(+timbre/behavior/synth_params) -> synchronous render -> take_id
  GET  /takes/{id}             take metadata
  GET  /takes/{id}/audio.wav   WAV bytes

Rendering uses the neural path (ConditionedNN retrieval + DDSP timbre) and does not depend on the real-time engine; capture timbres use the offline L0 engine.
POST /render completes synchronously (seconds for a short region) and returns state=ready, so a stateless container has no background threads or polling
races. NeuralTake's voice cache and the L0 engine cache are shared across requests, and all rendering happens under _render_lock.
CORS is controlled by the QUILL_ALLOWED_ORIGINS environment variable (default *).

Local trial run: QUILL_WEB_DIR=$PWD/web_demo uvicorn ui.render_service:app --port 8726
"""

from __future__ import annotations

import collections
import io
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

QUILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUILL))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from neural.neural_take import (  # noqa: E402  same timbre table/renderer as the desktop version
    DISTILLED_BEHAVIORS,
    INSTRUMENTS,
    UI_RENDER_DEFAULTS,
    available_instruments,
    get_neural_take,
)
from runtime.gesture import GestureInterpreter  # noqa: E402
from ui.take_store import TakeStore  # noqa: E402  on-disk take store

TAKES_DIR = Path(os.environ.get("QUILL_TAKES_DIR", "/tmp/quill_takes"))
ASSETS = Path(os.environ.get("QUILL_ASSETS_DIR", "/tmp/quill_assets"))
TONES_DIR, BEHS_DIR = ASSETS / "tones", ASSETS / "behaviors"
VERSION = "render-service/2.1"

# ---- Render/input contract constants ----
SR_OUT = 16000  # DDSP output sample rate
TAIL_S = 0.05  # tail kept after the last note: musicDurationTicks written back to the DAW is derived from it, so it must hug the timeline
MAX_RENDER_S = 120.0  # timeline cap per render
MAX_UPLOAD_BYTES = 30_000_000  # upload cap per /capture
CAPTURE_MIN_S, CAPTURE_MAX_S = 1.0, 90.0
BEH_CACHE_MAX = 8  # behaviour library LRU cap (container memory is limited)
MAX_STORED_TONES = int(os.environ.get("QUILL_MAX_ASSETS", "60"))  # cap on the total number of assets


def _persist(warnings: list | None = None):
    """Explicit Modal Volume commit (no-op when running locally).

    Failures must be visible: silently swallowing them would let /capture return ok while the asset vanishes in the next container.
    """
    if not os.environ.get("QUILL_VOLUME_NAME"):
        return
    try:
        import modal

        modal.Volume.from_name(os.environ["QUILL_VOLUME_NAME"]).commit()
    except Exception as e:
        msg = f"volume commit failed: {str(e)[:80]}"
        print(f"[quill] {msg}", file=sys.stderr)
        if warnings is not None:
            warnings.append(msg)


def _evict_assets(warnings: list | None = None):
    """Evict the oldest assets in the asset directory by mtime, preventing unbounded growth of the persistent volume (/capture has no auth)."""
    for d in (TONES_DIR, BEHS_DIR):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        for p in files[:-MAX_STORED_TONES] if len(files) > MAX_STORED_TONES else []:
            try:
                p.unlink()
                if warnings is not None:
                    warnings.append(f"evicted oldest asset {p.stem}")
            except OSError:
                pass


# only these knobs act on rendering; any key not listed is dropped.
#   noise->DDSP noise side  master->output gain  reverb->algorithmic reverb
#   brightness->spectral tilt  artint->traj_gain (trajectory strength)
#   vibmodel->vibrato source (≥0.5 uses the learned vibrato model, otherwise retrieval)  randomness->vibrato randomness
#   timing->expressive timing strength  continuity->concatenation-cost reranking λ
SYNTH_KEYS = {
    "noise", "master", "reverb", "brightness", "artint",
    "vibmodel", "randomness", "timing", "continuity",
}
DISTILLED_NAME = "mididdsp_distilled"  # key in neural_take; the external contract still calls it "distilled"
_render_lock = threading.Lock()  # NeuralTake's voice cache is not thread-safe


def _render_tone_l0(notes: list, eng):
    """Capture timbre -> offline rendering with the L0 engine (the same chain as the desktop version).

    Engine events carry absolute time, so the timeline is preserved by construction; slide/legato keep the desktop semantics.
    """
    import io as _io

    import soundfile as sf

    evs = _GI.notes_to_events(notes)  # same function as the desktop piano roll: velocity -> slide speed, strum staggering
    dur = max(t for t, kind, *_ in evs if kind == "off") + 0.5
    with _render_lock:  # the cached engine is shared across requests, same lock
        y = np.asarray(eng.render_offline(evs, dur), np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = (y / peak * 0.9).astype(np.float32)
    out_peak = float(np.max(np.abs(y))) if y.size else 0.0
    buf = _io.BytesIO()
    sf.write(buf, y, eng.sr, format="WAV")
    return (
        buf.getvalue(),
        (20.0 * np.log10(out_peak) if out_peak > 0 else -120.0),
        round(len(y) / eng.sr, 3),
    )


def _render_audio(
    notes: list, timbre: str, behavior: str | None = None, synth_params: dict | None = None
) -> tuple[bytes, float, float]:
    """notes({midi,start,len,vel,art}) -> NeuralTake.render -> WAV.

    Returns (wav_bytes, peak_dbfs, duration_s).

    Chord/overlap monophonisation, phrase splitting, timeline placement, edge fades and post-processing all happen in
    NeuralTake.render (the same function as the desktop version); this layer only resolves names and maps knobs.
    behavior values:
      built-in timbre name   that timbre's trajectory library (cross-instrument transfer, e.g. violin timbre × guitar behaviour)
      "beh:<id>"     behaviour library obtained from /capture
      "distilled"    distilled residual student model (MIDI-DDSP teacher -> Quill behaviour residual)
    """
    import soundfile as sf

    sp = synth_params or {}
    if timbre.startswith("tone:"):
        eng = _l0_engine(timbre[5:])
        if "noise" in sp and not eng._texture:  # the Air/Noise knob also acts on capture timbres (consistent with the desktop)
            eng.instrument.noise_level = float(sp["noise"])
        if not behavior:
            return _render_tone_l0(notes, eng)  # no behaviour library: event path
        timbre = eng  # capture timbre × behaviour library: L0 control-driven (same path as the desktop)
    if behavior == "distilled":
        behav = DISTILLED_NAME
    elif behavior and behavior.startswith("beh:"):
        behav = _load_behavior_nn(behavior[4:])
    else:
        behav = behavior or None

    with _render_lock:
        y, sr = get_neural_take().render(
            notes,
            instrument=timbre,
            behavior=behav,
            noise_gain=float(sp.get("noise", 0.6)),
            # INTENSITY knob -> traj_gain: strength of the retrieved trajectory (0.3 subdued ... 2.0 exaggerated)
            traj_gain=0.3 + float(sp.get("artint", 0.41)) * 1.7,
            vibrato_source="model" if float(sp.get("vibmodel", 0.0)) >= 0.5 else "retrieval",
            randomness=float(sp.get("randomness", 0.3)),
            **{k: float(sp.get(k, v)) for k, v in UI_RENDER_DEFAULTS.items()},
            postfx={"brightness": sp.get("brightness", 0.5), "reverb": sp.get("reverb", 0.0)},
            tail_s=TAIL_S,
        )
    out_peak = float(np.max(np.abs(y))) if y.size else 0.0
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return (
        buf.getvalue(),
        (20.0 * np.log10(out_peak) if out_peak > 0 else -120.0),
        round(len(y) / sr, 3),
    )


_beh_cache: collections.OrderedDict[str, object] = collections.OrderedDict()
_l0_cache: dict = {}
_GI = GestureInterpreter()  # single semantic source for piano-roll/DAW notes -> engine events (same class as the desktop)


def _l0_engine(tone_id: str):
    """captured tone -> offline L0 engine (control-driven), one cached by file mtime."""
    from runtime.engine import engine_from_snapshot_file
    from snapshot import load_snapshot

    p = TONES_DIR / f"{tone_id}.npz"
    key = (tone_id, p.stat().st_mtime)
    if key not in _l0_cache:
        meta = load_snapshot(str(p)).get("meta", {}) or {}
        tex = bool(meta.get("texture"))
        _l0_cache.clear()
        _l0_cache[key] = engine_from_snapshot_file(
            str(p), eval_mode=True, noise_level=0.6 if tex else float(meta.get("noise_level") or 0.02)
        )
        _l0_cache[key]._texture = tex  # the noise layer of texture timbres is not overridden by the Air/Noise knob (consistent with the desktop)
    return _l0_cache[key]


def _load_behavior_nn(beh_id: str):
    """captured behavior npz -> ConditionedNN (LRU cache, container memory is limited)."""
    if beh_id in _beh_cache:
        _beh_cache.move_to_end(beh_id)
        return _beh_cache[beh_id]
    from neural.neural_take import ConditionedNN

    d = np.load(str(BEHS_DIR / f"{beh_id}.npz"))
    nn = ConditionedNN(d["features"], d["trajectories"])
    _beh_cache[beh_id] = nn
    while len(_beh_cache) > BEH_CACHE_MAX:
        _beh_cache.popitem(last=False)
    return nn


def _slug(name: str, blob: bytes) -> str:
    import hashlib
    import re

    base = (
        re.sub(r"[^\w\-]+", "-", (name or "capture").strip().lower()).strip("-")[:24] or "capture"
    )
    return f"{base}-{hashlib.sha1(blob).hexdigest()[:8]}"


def _decode_audio(blob: bytes):
    """Any common format -> mono float32. Writes a temp file so the ffmpeg backend works properly."""
    import tempfile

    import librosa

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(blob)
        tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=None, mono=True)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return y.astype(np.float32), int(sr)


_extract_mod = None

def _get_extract_mod():
    global _extract_mod
    if _extract_mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "quill_extract_style", str(QUILL / "tools" / "extract_style.py")
        )
        _extract_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_extract_mod)
    return _extract_mod

def _extract_behavior(blob: bytes, out_npz: Path):
    """extract_style pipeline (paths in and out); returns (n_notes, art_counts) or None."""
    import tempfile

    mod = _get_extract_mod()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        import soundfile as _sf

        y, sr = _decode_audio(blob)
        _sf.write(f.name, y, sr)
        tmp = f.name
    try:
        mod.extract(tmp, str(out_npz), verbose=False)
    finally:
        Path(tmp).unlink(missing_ok=True)
    if not out_npz.exists():
        return None
    d = np.load(str(out_npz))
    n = int(d["features"].shape[0])
    if n < 2:
        out_npz.unlink(missing_ok=True)
        return None
    arts = d["features"][:, 6].astype(int)
    return n, {
        "plain": int((arts == 0).sum()),
        "vibrato": int((arts == 1).sum()),
        "slide": int((arts == 2).sum()),
    }


# --------------------------------------------------------------------
app = FastAPI(title="Quill Render Service")
_origins = [o for o in os.environ.get("QUILL_ALLOWED_ORIGINS", "*").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
_takes = TakeStore(TAKES_DIR)


@app.middleware("http")
async def _no_cache_html(request: Request, call_next):
    """Disable caching for the entry HTML so browsers do not keep the old page after a redeploy; static assets are cached as usual."""
    resp = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/status")
def status():
    builtin = available_instruments()
    behaviors = list(builtin)
    if DISTILLED_BEHAVIORS[DISTILLED_NAME].exists():
        behaviors.append("distilled")
    return {
        "ok": True,
        "version": VERSION,
        "engine": "neural-ddsp",
        "timbres": builtin,
        "behaviors": behaviors,  # timbre name = that timbre's trajectory library; "distilled" = distilled student model
    }


@app.post("/capture")
async def capture(request: Request, name: str = "capture"):
    """One recording -> two assets: timbre snapshot (tone) + articulation style library (behavior)."""
    blob = await request.body()
    if not blob or len(blob) < 1000:
        raise HTTPException(400, "empty audio")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "audio too large (30MB max)")
    try:
        y, sr = _decode_audio(blob)
    except Exception as e:
        raise HTTPException(400, f"cannot decode audio: {str(e)[:50]}") from None
    dur = len(y) / sr
    if not (CAPTURE_MIN_S <= dur <= CAPTURE_MAX_S):
        raise HTTPException(400, f"audio must be 1-90s (got {dur:.1f}s)")
    TONES_DIR.mkdir(parents=True, exist_ok=True)
    BEHS_DIR.mkdir(parents=True, exist_ok=True)
    aid = _slug(name, blob)
    out = {
        "ok": True,
        "id": aid,
        "duration_s": round(dur, 2),
        "tone": None,
        "behavior": None,
        "warnings": [],
    }
    try:
        from analysis.timbre import analyze_timbre
        from snapshot import save_snapshot

        snap = analyze_timbre(y, sr, name=name)
        snap.setdefault("meta", {})["disp"] = name  # display name for the asset list
        save_snapshot(str(TONES_DIR / aid), snap)
        f0 = float(snap.get("f0_ref", 0.0))
        midi = 69 + 12 * np.log2(f0 / 440.0) if f0 > 0 else 0
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note = f"{names[int(round(midi)) % 12]}{int(round(midi)) // 12 - 1}" if f0 > 0 else "—"
        h = snap.get("H_env")
        out["tone"] = {
            "id": aid,
            "name": name,
            "f0_hz": round(f0, 1),
            "note": note,
            "partials": (int(np.asarray(h).shape[0]) if h is not None else 0),
        }
    except Exception as e:
        out["warnings"].append(f"tone: {str(e)[:60]}")
    try:
        r = _extract_behavior(blob, BEHS_DIR / f"{aid}.npz")
        if r:
            n, arts = r
            out["behavior"] = {"id": aid, "name": name, "n_notes": n, "arts": arts}
        else:
            out["warnings"].append("behavior: fewer than 2 notes detected")
    except Exception as e:
        out["warnings"].append(f"behavior: {str(e)[:60]}")
    if not out["tone"] and not out["behavior"]:
        raise HTTPException(422, "; ".join(out["warnings"]) or "capture failed")
    _evict_assets(out["warnings"])
    _persist(out["warnings"])
    return out


def _reload_vol():
    if not os.environ.get("QUILL_VOLUME_NAME"):
        return
    try:
        import modal

        modal.Volume.from_name(os.environ["QUILL_VOLUME_NAME"]).reload()
    except Exception:
        pass


@app.get("/assets")
def assets():
    """List of captured assets (tones + behaviors)."""
    _reload_vol()
    tones, behs = [], []
    if TONES_DIR.is_dir():
        for p in sorted(TONES_DIR.glob("*.npz"), key=lambda x: x.stat().st_mtime):
            try:
                from snapshot import load_snapshot

                disp = (load_snapshot(str(p)).get("meta", {}) or {}).get("disp") or p.stem
            except Exception:
                disp = p.stem
            tones.append({"id": p.stem, "name": str(disp)})
    if BEHS_DIR.is_dir():
        for p in sorted(BEHS_DIR.glob("*.npz"), key=lambda x: x.stat().st_mtime):
            try:
                d = np.load(str(p))
                behs.append(
                    {
                        "id": p.stem,
                        "name": p.stem.rsplit("-", 1)[0],
                        "n_notes": int(d["features"].shape[0]),
                    }
                )
            except Exception:
                continue
    # the distilled behaviour student sits alongside captured behaviour libraries and can be swapped in directly for comparison
    if DISTILLED_BEHAVIORS[DISTILLED_NAME].exists():
        behs.insert(
            0, {"id": "distilled", "name": "MIDI-DDSP distilled", "n_notes": 0, "kind": "model"}
        )
    return {"tones": tones, "behaviors": behs}


@app.post("/render")
def render(body: dict):
    """notes + timbre + behavior → staged take, rendered synchronously."""
    _reload_vol()
    notes = body.get("notes") or []
    bpm = float(body.get("bpm", 120.0))
    title = str(body.get("title", "") or "Inked region")[:80]
    timbre = str(body.get("timbre", "violin"))
    behavior = str(body.get("behavior", "") or "")
    _safe_id = lambda s: all(c.isalnum() or c in "-_" for c in s) and len(s) <= 80
    if timbre.startswith("tone:"):
        tid = timbre[5:]
        if not _safe_id(tid):
            raise HTTPException(400, "invalid tone id")
        if not (TONES_DIR / f"{tid}.npz").exists():
            raise HTTPException(400, f"unknown captured tone {tid!r}")
    elif timbre not in available_instruments():
        raise HTTPException(400, f"unknown timbre {timbre!r}")
    if behavior in ("distilled", "beh:distilled"):
        behavior = "distilled"
        if not DISTILLED_BEHAVIORS[DISTILLED_NAME].exists():
            raise HTTPException(400, "distilled behavior model not deployed")
    elif behavior in INSTRUMENTS:
        if behavior not in available_instruments():
            raise HTTPException(400, f"behavior library for {behavior!r} not deployed")
    elif behavior.startswith("beh:"):
        bid = behavior[4:]
        if not _safe_id(bid):
            raise HTTPException(400, "invalid behavior id")
        if not (BEHS_DIR / f"{bid}.npz").exists():
            raise HTTPException(400, f"unknown behavior {bid!r}")
    elif behavior:
        raise HTTPException(400, f"unknown behavior {behavior!r}")
    if not notes:
        raise HTTPException(400, "empty notes")
    end = 0.0
    try:
        for nt in notes:
            m, s, ln = int(nt["midi"]), float(nt["start"]), float(nt["len"])
            if not (0 <= m <= 127) or ln <= 0 or s < 0:
                raise HTTPException(400, f"bad note midi={m}")
            end = max(end, s + ln)
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as e:  # missing fields/wrong types are 400, not 500
        raise HTTPException(400, f"malformed notes: {str(e)[:50]}") from None
    if end > MAX_RENDER_S:
        raise HTTPException(400, f">{MAX_RENDER_S:.0f}s")

    synth_params = {
        k: max(0.0, min(1.0, float(v)))
        for k, v in (body.get("synth_params") or {}).items()
        if k in SYNTH_KEYS
    }
    clean = [
        {
            "midi": int(n["midi"]),
            "start": float(n["start"]),
            "len": float(n["len"]),
            "vel": int(n.get("vel", 100)),
            "art": str(n.get("art", "pluck")),
        }
        for n in notes
    ]
    take = _takes.stage(
        title=title,
        notes=clean,
        bpm=bpm,
        timbre={
            "name": timbre + (f" × {behavior}" if behavior else ""),
            "f0_ref_hz": 0.0,
            "noise_level": 0.0,
        },
        duration_s=end,
        sr=SR_OUT,
        provenance={
            "quill_version": VERSION,
            "engine": "neural-ddsp",
            "atlas": None,
            "source_region": None,
        },
        # knobs affect the audio but are not in the other fields, so they must be part of the take identity, otherwise after changing a knob
        # a rerender would wrongly hit the idempotent cache and return the old audio (the knob would seem to "do nothing").
        variant=json.dumps(synth_params, sort_keys=True, separators=(",", ":")),
    )
    if take["audio"]["state"] == "ready":  # idempotent hit
        return {"ok": True, "take_id": take["take_id"], "state": "ready"}
    try:
        wav, peak_db, audio_s = _render_audio(clean, timbre, behavior or None, synth_params)
    except HTTPException:
        raise
    except Exception as e:
        _takes.fail(take["take_id"], str(e))
        raise HTTPException(500, f"render_failed: {str(e)[:60]}") from None
    _takes.attach_audio(take["take_id"], wav, peak_db)
    # duration_s becomes the real timeline length after phrase placement; the DAW side derives musicDurationTicks from it,
    # so it matches the source region's rhythm exactly and is not stretched.
    _takes.set_duration(take["take_id"], audio_s)
    return {"ok": True, "take_id": take["take_id"], "state": "ready"}


@app.get("/takes")
def takes_list():
    return {"takes": _takes.list()}


def _valid_tid(tid: str) -> bool:
    """take_id is a content hash tk_+12hex; validate the format before path joining (defence in depth)."""
    import re

    return bool(re.fullmatch(r"tk_[0-9a-f]{12}", tid))


@app.get("/takes/{tid}")
def take_meta(tid: str):
    take = _takes.get(tid) if _valid_tid(tid) else None
    if take is None:
        raise HTTPException(404, "not_found")
    return take


@app.get("/takes/{tid}/audio.wav")
def take_audio(tid: str):
    take = _takes.get(tid) if _valid_tid(tid) else None
    if take is None:
        raise HTTPException(404, "not_found")
    if take["audio"]["state"] != "ready":
        raise HTTPException(409, take["audio"]["state"])
    return FileResponse(_takes.wav_path(tid), media_type="audio/wav")


# ---- Same-origin hosting of the web page (QUILL_WEB_DIR points to web_demo/) ----
# the mount is registered after the routes, so /status and other APIs match first; page and API share an origin -> zero CORS.
_web = os.environ.get("QUILL_WEB_DIR", "")
if _web and Path(_web).is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
