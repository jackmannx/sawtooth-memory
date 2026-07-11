"""add_message microbenchmarks."""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.common.config import MockedContextManager


@pytest.mark.benchmark(group="add_message")
class TestAddMessageBenchmark:
    def test_add_message_short(self, benchmark, bench_config) -> None:
        async def _run() -> None:
            async with MockedContextManager(
                "You are a physics expert.",
                bench_config,
                enable_events=False,
            ) as cm:
                await cm.add_message("user", "Tell me about quantum entanglement.")

        benchmark(lambda: asyncio.run(_run()))

    def test_add_message_with_ner_entities(self, benchmark, bench_config) -> None:
        content = (
            "Remember txn_998877_alpha_omega at /etc/nginx/sites-enabled/api.conf "
            "and https://internal.corp/runbooks/inc-4421"
        )

        async def _run() -> None:
            async with MockedContextManager(
                "You are a physics expert.",
                bench_config,
                enable_events=False,
            ) as cm:
                await cm.add_message("user", content)

        benchmark(lambda: asyncio.run(_run()))
