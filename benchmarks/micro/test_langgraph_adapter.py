"""LangGraph adapter microbenchmarks."""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from benchmarks.common.config import MockedContextManager
from sawtooth_memory.integrations.langgraph.adapter import SawtoothLangGraphAdapter


def _tool_rich_messages(count: int) -> list:
    messages = []
    for i in range(count):
        call_id = f"call_{i}"
        messages.append(HumanMessage(content=f"Run tool {i}", id=f"human_{i}"))
        messages.append(
            AIMessage(
                content="",
                id=f"ai_{i}",
                tool_calls=[{"id": call_id, "name": "lookup", "args": {"q": str(i)}}],
            )
        )
        messages.append(
            ToolMessage(
                content=json.dumps({"status": "ok", "value": i}),
                tool_call_id=call_id,
                id=f"tool_{i}",
            )
        )
    return messages


@pytest.mark.benchmark(group="langgraph_adapter")
class TestLangGraphAdapterBenchmark:
    def test_sync_state_dedup(self, benchmark, bench_config) -> None:
        messages = _tool_rich_messages(20)

        async def _run() -> None:
            async with MockedContextManager(
                "You are a physics expert.",
                bench_config,
                enable_events=False,
            ) as cm:
                adapter = SawtoothLangGraphAdapter(cm)
                await adapter.sync_state(messages)
                await adapter.sync_state(messages)

        benchmark(lambda: asyncio.run(_run()))

    def test_get_compiled_prompt_sanitization(self, benchmark, bench_config) -> None:
        async def _run() -> None:
            async with MockedContextManager(
                "You are a physics expert.",
                bench_config,
                enable_events=False,
            ) as cm:
                adapter = SawtoothLangGraphAdapter(cm)
                await adapter.sync_state(_tool_rich_messages(10))
                await adapter.get_compiled_prompt()

        benchmark(lambda: asyncio.run(_run()))
