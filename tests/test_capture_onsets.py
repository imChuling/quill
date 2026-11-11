"""取色结果里的 n_onsets:合成三音乐句 → ≈3;单音 → ≤1;太短 → 0。驱动取色页的"行为库需要 2+ 音"提示。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

from bridge_capture import BridgeCaptureMixin  # noqa: E402

SR = 22050


def _tone(hz, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    env = np.minimum(1.0, t / 0.01) * np.exp(-t * 2.0)
    return (0.4 * np.sin(2 * np.pi * hz * t) * env).astype(np.float32)


def test_three_separated_notes_count_about_three():
    gap = np.zeros(int(0.25 * SR), np.float32)
    y = np.concatenate([_tone(330, 0.6), gap, _tone(392, 0.6), gap, _tone(440, 0.8)])
    n = BridgeCaptureMixin._count_onsets(y, SR)
    assert n is not None and 2 <= n <= 4, n


def test_single_sustained_note_counts_at_most_one():
    n = BridgeCaptureMixin._count_onsets(_tone(440, 3.0), SR)
    assert n is not None and n <= 1, n


def test_too_short_take_is_zero_and_stereo_is_accepted():
    assert BridgeCaptureMixin._count_onsets(np.zeros(100, np.float32), SR) == 0
    st = np.stack([_tone(440, 1.0), _tone(440, 1.0)], axis=1)
    assert BridgeCaptureMixin._count_onsets(st, SR) is not None
