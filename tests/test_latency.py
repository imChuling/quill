"""Latency regression gate: compute time of process_block at 16-voice full load (@256/44.1k, block budget 5.8 ms).

Two assertions, each covering one thing:
  * median < 3.0 ms: catches code slowdowns. The median is insensitive to system load; measured ~2.65 ms on a quiet machine
    (2026-10-29, M2), 1.68 ms half a year earlier.
  * p95 < 5.8 ms (block budget): catches xrun risk. p95 is strongly affected by OS scheduling jitter,
    and under heavy load drifts from 2.7 ms to 13 ms, so it only guards the budget and is not held tighter.
Three independent measurements, the median of each, reduce single-run jitter.

Marked slow, not run in CI (no sound card in the cloud, and machine performance is not representative of this machine); locally `pytest tests/test_latency.py`.
"""

import statistics

import pytest

from eval.metrics import measure_render_compute_ms
from runtime.engine import Engine
from synth.instrument import build_instrument
from tests.helpers import make_test_snapshot

pytestmark = pytest.mark.slow

BLOCK_BUDGET_MS = 256 / 44100 * 1000  # 5.80 ms
MEDIAN_GATE_MS = 3.0  # code regression gate: ~2.65 ms measured on a quiet machine, 13% headroom
P95_GATE_MS = BLOCK_BUDGET_MS  # xrun gate: p95 must not exceed one block's budget


def _measure(n_runs=3, n_blocks=200):
    meds, p95s = [], []
    for _ in range(n_runs):
        eng = Engine(build_instrument(make_test_snapshot()), eval_mode=True)
        med, p95 = measure_render_compute_ms(eng, n_voices=16, n_blocks=n_blocks)
        meds.append(med)
        p95s.append(p95)
    return statistics.median(meds), statistics.median(p95s)


def test_render_median_has_not_regressed():
    med, _ = _measure()
    assert med < MEDIAN_GATE_MS, f"median(3 runs) = {med:.3f} ms, exceeds {MEDIAN_GATE_MS} ms (code got slower?)"


def test_render_p95_within_block_budget():
    _, p95 = _measure()
    assert p95 < P95_GATE_MS, f"p95(median of 3) = {p95:.3f} ms, exceeds block budget {P95_GATE_MS:.2f} ms (xrun risk)"
