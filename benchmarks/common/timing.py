"""Latency sampling and percentile statistics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LatencyStats:
    count: int
    min_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    stdev_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "min_ms": self.min_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
            "stdev_ms": self.stdev_ms,
        }


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def summarize_latencies(samples_ms: Sequence[float]) -> LatencyStats:
    """Compute descriptive latency statistics from millisecond samples."""
    if not samples_ms:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ordered = sorted(samples_ms)
    stdev = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    return LatencyStats(
        count=len(ordered),
        min_ms=ordered[0],
        mean_ms=statistics.mean(ordered),
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        p99_ms=_percentile(ordered, 99),
        max_ms=ordered[-1],
        stdev_ms=stdev,
    )
