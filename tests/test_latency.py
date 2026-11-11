"""延迟回归门:16 复音满载 process_block 的计算耗时(@256/44.1k,块预算 5.8 ms)。

两条断言,各管一件事:
  * 中位数 < 3.0 ms:抓代码变慢。中位数对系统负载不敏感,安静机器实测 ~2.65 ms
    (2026-10-29,M2),半年前 1.68 ms。
  * p95 < 5.8 ms(块预算):抓 xrun 风险。p95 受操作系统调度抖动影响大,
    高负载下会从 2.7 ms 飘到 13 ms,所以只按预算兜底,不卡更紧。
三次独立测量各取中位,降低单次抖动。

标记 slow,CI 不跑(云端无声卡、机器性能不代表本机);本地 `pytest tests/test_latency.py`。
"""

import statistics

import pytest

from eval.metrics import measure_render_compute_ms
from runtime.engine import Engine
from synth.instrument import build_instrument
from tests.helpers import make_test_snapshot

pytestmark = pytest.mark.slow

BLOCK_BUDGET_MS = 256 / 44100 * 1000  # 5.80 ms
MEDIAN_GATE_MS = 3.0  # 代码退化门:安静机器实测 ~2.65 ms,留 13% 余量
P95_GATE_MS = BLOCK_BUDGET_MS  # xrun 门:p95 不得超过一个块的预算


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
