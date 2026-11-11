"""实时引擎自然幅度包络（拨弦衰减 / 持续保持）—— build_instrument amp_env + Voice。"""

import numpy as np
import pytest

from runtime.engine import Engine
from synth.instrument import build_instrument
from tests.helpers import SR, make_test_snapshot


def _snap(decay: bool, f0=220.0, T=200):
    return make_test_snapshot(f0=f0, T=T, decay=decay)


def test_amp_env_decays_for_plucked():
    inst = build_instrument(_snap(decay=True))
    assert inst.amp_env is not None
    assert inst.amp_env[-1] < 0.2  # 衰减到接近 0


def test_amp_env_flat_for_sustained():
    inst = build_instrument(_snap(decay=False))
    # 持续类包络≈1（不引入衰减 -> 无回归）
    assert inst.amp_env is None or float(np.min(inst.amp_env)) > 0.9


def _tail_vs_head(inst, hold_s=1.5, sr=SR):
    eng = Engine(inst, eval_mode=True)
    y = np.asarray(
        eng.render_offline(
            [(0.05, "on", 57, 100, 0.0), (hold_s, "off", 57, 0, 0.0)], dur_s=hold_s + 0.3
        )
    )
    head = float(np.sqrt(np.mean(y[int(sr * 0.1) : int(sr * 0.3)] ** 2)) + 1e-12)
    tail = float(np.sqrt(np.mean(y[int(sr * 1.0) : int(sr * 1.2)] ** 2)) + 1e-12)
    return head, tail


def test_engine_plucked_note_decays():
    head, tail = _tail_vs_head(build_instrument(_snap(decay=True)))
    assert head > 3 * tail  # 拨弦：尾远小于起


def test_engine_sustained_note_holds():
    head, tail = _tail_vs_head(build_instrument(_snap(decay=False)))
    assert tail > 0.5 * head  # 持续：尾不塌（无回归）


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
