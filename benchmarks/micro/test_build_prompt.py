"""build_prompt microbenchmarks."""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.common.config import MockedContextManager
from sawtooth_memory.state import MemoryState


@pytest.mark.benchmark(group="build_prompt")
class TestBuildPromptBenchmark:
    def test_build_prompt_populated_state(
        self, benchmark, bench_config, populated_state: MemoryState
    ) -> None:
        async def _run() -> None:
            async with MockedContextManager(
                "You are a physics expert.",
                bench_config,
                enable_events=False,
            ) as cm:
                cm._state = populated_state
                await cm.build_prompt()

        benchmark(lambda: asyncio.run(_run()))
