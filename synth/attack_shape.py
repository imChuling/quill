"""拨片厚度 / 起音软硬 —— 可演奏参数（FR-12 合成器面板旋钮，numpy-only 引擎纯净）。

用户洞察（2026-06）：木/电/古典/电声吉他的**起音**差异极大；"厚拨片"听感=重、亮、尖。
把它做成一个可解释、可弹的参数 `pick`∈[0,1]：
  0 = 指弹/薄拨片：起音颗粒轻、暗（低通）、圆（更长起音 fade）；
  1 = 厚拨片：起音颗粒重、亮（近全通）、尖（无 fade）。
映射三处：① 起音残差颗粒增益 ∝ pick；② 颗粒亮度（低通 cutoff 随 pick 升）；③ 起音 fade ∝ (1-pick)。

只依赖 numpy（铁律 #12：synth/ 不引 scipy/librosa）。低通用 rfft 砖墙（颗粒短，振铃可忽略）。
"""
from __future__ import annotations

import numpy as np

PICK_DEFAULT = 0.5


def _lowpass_fft(x, cutoff_hz, sr):
    """numpy 砖墙低通（颗粒级，<3k 样本，振铃无碍）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[f > cutoff_hz] = 0.0
    return np.fft.irfft(X, n=len(x))


def shape_attack(grain, pick: float, sr: int = 44100):
    """按拨片厚度整形起音残差颗粒。pick∈[0,1] -> 整形后颗粒（float32）。

    pick=0 返回零（纯指弹无瞬态）；pick=1 原样（最尖最亮）；中间=低通+衰减。
    """
    grain = np.asarray(grain, dtype=np.float64)
    pick = float(np.clip(pick, 0.0, 1.0))
    if grain.size == 0 or pick <= 0.0:
        return np.zeros_like(grain, dtype=np.float32)
    if pick >= 1.0:
        return grain.astype(np.float32)
    cutoff = min(600.0 * 2.0 ** (pick * 4.0), sr * 0.49)   # 软 600Hz → 硬 ~9.6kHz
    g = _lowpass_fft(grain, cutoff, sr)
    return (pick * g).astype(np.float32)


def onset_fade(n_samples_hint, pick: float, sr: int = 44100, max_ms: float = 40.0):
    """返回起音 fade-in 包络（升余弦平方根，长度随 (1-pick)）；pick=1 时长度 0。"""
    pick = float(np.clip(pick, 0.0, 1.0))
    fade = int(sr * (max_ms / 1000.0) * (1.0 - pick))
    if fade <= 1:
        return None
    return np.sqrt(0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))).astype(np.float32)


def apply_pick(body, attack_res, pick: float, sr: int = 44100):
    """把整形后的起音叠加到无硬起音的琴体上，并按 pick 施加起音 fade。

    body: 不含 attack 的渲染（render(..., add_attack=False)）；attack_res: 快照起音残差。
    返回带"拨片厚度"起音的音频（float32）。
    """
    y = np.asarray(body, dtype=np.float64).copy()
    g = shape_attack(attack_res, pick, sr)
    if g.size:
        m = min(len(g), len(y))
        y[:m] += g[:m]
    fade = onset_fade(len(y), pick, sr)
    if fade is not None:
        m = min(len(fade), len(y))
        y[:m] *= fade[:m]
    return y.astype(np.float32)
