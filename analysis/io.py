"""Audio I/O and gain normalisation.

  * load(path)   : read wav/mp3 -> (y float32 mono, sr=44100)
  * record(sec)  : microphone recording -> y (sounddevice, blocking)
  * normalize(y) : peak normalisation

Stereo input is downmixed automatically; unified to 44.1k / mono.
"""

from __future__ import annotations

import numpy as np

try:  # central config; defaults when this module is imported outside the repo
    from quill_config import CFG

    _SR = int(CFG["audio"]["sr"])
    _NORM_PEAK = float(CFG["audio"]["norm_peak"])
except Exception:  # pragma: no cover
    _SR = 44100
    _NORM_PEAK = 0.97


def normalize(y: np.ndarray, peak: float = _NORM_PEAK) -> np.ndarray:
    """Peak-normalise to `peak`.

    Silent/all-zero signals are returned as is, avoiding division by zero that would amplify the noise floor.
    """
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m < 1e-9:
        return y
    return (y * (peak / m)).astype(np.float32)


_MAX_SECONDS = 60.0  # upper limit for a performance clip


def load(
    path: str,
    sr: int = _SR,
    mono: bool = True,
    do_normalize: bool = True,
    max_seconds: float = _MAX_SECONDS,
) -> tuple[np.ndarray, int]:
    """Read an audio file -> (y, sr). Supports wav/mp3 (librosa via soundfile/audioread).

    librosa resamples to `sr` (default 44100) and downmixes to mono; truncates beyond `max_seconds`
    (default 60s). Returns float32, range about [-1, 1]; peak-normalises when `do_normalize=True`.
    """
    import librosa  # local import: keep librosa out of everything outside analysis

    dur = float(max_seconds) if max_seconds else None
    y, sr_out = librosa.load(path, sr=sr, mono=mono, duration=dur)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if do_normalize:
        y = normalize(y)
    return y, int(sr_out)


def record(
    sec: float, sr: int = _SR, do_normalize: bool = True, max_seconds: float = _MAX_SECONDS
) -> np.ndarray:
    """Blocking recording of `sec` seconds from the default input device -> y (mono, float32).

    `sec` is capped at `max_seconds`. Raises RuntimeError when no input device is available / permission is denied
    (macOS asks for microphone permission the first time).
    """
    import sounddevice as sd  # local import, unit tests/offline paths need no sound card

    sec = min(float(sec), float(max_seconds)) if max_seconds else float(sec)
    n = int(round(sec * sr))
    try:
        buf = sd.rec(n, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:  # missing device/permission/driver
        raise RuntimeError(f"Recording failed (check microphone permission/input device): {exc}") from exc

    y = np.ascontiguousarray(buf.reshape(-1), dtype=np.float32)
    if do_normalize:
        y = normalize(y)
    return y


def trim_silence(
    y: np.ndarray, sr: int = _SR, thresh_db: float = -42.0, pad_ms: float = 60.0
) -> np.ndarray:
    """Trim leading/trailing silence by an energy threshold: breaths/noise/hiss in casual recordings sit mostly in the silent ends; trimming them
    cleans up f0 tracking and the noise envelope. The threshold is relative to peak RMS; pad is kept at the start to preserve the natural attack."""
    y = np.asarray(y, dtype=np.float32)
    hop = 512
    n = len(y) // hop
    if n < 4:
        return y
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    act = np.where(rms > rms.max() * 10 ** (thresh_db / 20.0))[0]
    if not act.size:
        return y
    pad = int(pad_ms / 1000.0 * sr)
    a = max(0, int(act[0]) * hop - pad)
    b = min(len(y), (int(act[-1]) + 1) * hop + pad)
    return y[a:b]
