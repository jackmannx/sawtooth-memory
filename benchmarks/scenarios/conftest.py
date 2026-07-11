"""Scenario benchmark fixtures."""

from __future__ import annotations

import pytest

from benchmarks.common.config import benchmark_config


@pytest.fixture
def scenario_config():
    return benchmark_config(
        soft_limit_tokens=250,
        hard_limit_tokens=600,
        chunk_size=4,
        enable_events=False,
    )
