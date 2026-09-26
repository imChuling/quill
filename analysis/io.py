"""音频 I/O 与增益归一化 —— FR-1 雏形（阶段1 Day1 下午）。

职责：
  * load(path)   : 读 wav/mp3 -> (y float32 mono, sr=44100)
  * record(sec)  : 麦克风录音 -> y（sounddevice 阻塞式）
  * normalize(y) : 自动增益（峰值归一化）

立体声输入自动 downmix（NFR-5）；统一 44.1k / mono。
"""
from __future__ import annotations

import numpy as np

try:  # 集中配置；脱离仓库单独导入本模块时给默认值
    from quill_config import CFG
    _SR = int(CFG["audio"]["sr"])
    _NORM_PEAK = float(CFG["audio"]["norm_peak"])
except Exception:  # pragma: no cover - 兜底默认
    _SR = 44100
    _NORM_PEAK = 0.97


def normalize(y: np.ndarray, peak: float = _NORM_PEAK) -> np.ndarray:
    """峰值归一化到 `peak`（FR-1 自动增益）。

    静音/全零信号原样返回，避免除零放大底噪。
    """
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m < 1e-9:
        return y
    return (y * (peak / m)).astype(np.float32)


_MAX_SECONDS = 60.0   # NFR-6 / FR-1：演奏片段 ≤60s


def load(path: str, sr: int = _SR, mono: bool = True,
         do_normalize: bool = True, max_seconds: float = _MAX_SECONDS
         ) -> tuple[np.ndarray, int]:
    """读取音频文件 -> (y, sr)。支持 wav/mp3（librosa via soundfile/audioread）。

    librosa 重采样到 `sr`（默认 44100）、downmix 到单声道（NFR-5）；
    超过 `max_seconds`（默认 60s，FR-1）截断。
    返回 float32，范围约 [-1, 1]；`do_normalize=True` 时做峰值归一化。
    """
    import librosa  # 局部导入：保持 analysis 之外无 librosa 依赖（铁律 #12）

    dur = float(max_seconds) if max_seconds else None
    y, sr_out = librosa.load(path, sr=sr, mono=mono, duration=dur)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if do_normalize:
        y = normalize(y)
    return y, int(sr_out)


def record(sec: float, sr: int = _SR, do_normalize: bool = True,
           max_seconds: float = _MAX_SECONDS) -> np.ndarray:
    """从默认输入设备阻塞式录音 `sec` 秒 -> y（mono, float32）。

    `sec` 上限 `max_seconds`（FR-1 麦克风录音 ≤60s）。无可用输入设备 / 权限被拒时
    抛 RuntimeError（macOS 首次会弹麦克风授权，见实施计划阶段1 坑位）。
    """
    import sounddevice as sd  # 局部导入，单测/离线路径不需要声卡

    sec = min(float(sec), float(max_seconds)) if max_seconds else float(sec)
    n = int(round(sec * sr))
    try:
        buf = sd.rec(n, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:  # 设备缺失/权限/驱动
        raise RuntimeError(
            f"录音失败（检查麦克风权限/输入设备）：{exc}"
        ) from exc

    y = np.ascontiguousarray(buf.reshape(-1), dtype=np.float32)
    if do_normalize:
        y = normalize(y)
    return y


def trim_silence(y: np.ndarray, sr: int = _SR, thresh_db: float = -42.0,
                 pad_ms: float = 60.0) -> np.ndarray:
    """能量门限修剪首尾静音(取色地板 P3):随手录音的呼吸/杂音/底噪多在首尾静默段,
    修掉后 f0 跟踪与噪声包络都干净。阈值相对峰值 RMS;首端留 pad 保住自然起音。"""
    y = np.asarray(y, dtype=np.float32)
    hop = 512
    n = len(y) // hop
    if n < 4:
        return y
    rms = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    act = np.where(rms > rms.max() * 10 ** (thresh_db / 20.0))[0]
    if not act.size:
        return y
    pad = int(pad_ms / 1000.0 * sr)
    a = max(0, int(act[0]) * hop - pad)
    b = min(len(y), (int(act[-1]) + 1) * hop + pad)
    return y[a:b]
