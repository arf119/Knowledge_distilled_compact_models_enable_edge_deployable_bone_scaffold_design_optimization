from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyReport:
    median_ms: float
    interquartile_range_ms: float
    samples: int


def profile_latency(
    operation: Callable[[], object], warmup: int = 100, runs: int = 1000
) -> LatencyReport:
    for _ in range(warmup):
        operation()
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        operation()
        timings.append((time.perf_counter_ns() - start) / 1.0e6)
    quartiles = statistics.quantiles(timings, n=4)
    return LatencyReport(statistics.median(timings), quartiles[2] - quartiles[0], runs)
