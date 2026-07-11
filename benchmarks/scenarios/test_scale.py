"""Scale benchmarks across conversation length and message size."""

from __future__ import annotations

import pytest

from benchmarks.common.config import MockedContextManager
from benchmarks.common.fixtures import generate_conversation
from benchmarks.common.timing import summarize_latencies
from sawtooth_memory.monitor import TokenMonitor


async def _run_scale_scenario(turns: int, message_size: str) -> dict[str, object]:
    import time

    from benchmarks.common.config import benchmark_config

    config = benchmark_config(enable_events=False)
    conversation = generate_conversation(turns=turns, message_size=message_size)
    add_latencies_ms: list[float] = []

    session = MockedContextManager(
        "You are a physics expert.",
        config,
        mock_delay_ms=0.5,
        enable_events=False,
    )
    async with session as cm:
        for msg in conversation:
            start = time.perf_counter()
            await cm.add_message(msg["role"], msg["content"])  # type: ignore[arg-type]
            add_latencies_ms.append((time.perf_counter() - start) * 1000.0)

        await cm.stop()
        final_prompt = await cm.build_prompt()
        prompt_string = "\n".join(m["content"] for m in final_prompt)
        monitor = TokenMonitor(model="gpt-4o")

    add_stats = summarize_latencies(add_latencies_ms)
    return {
        "turns": turns,
        "message_size": message_size,
        "messages_total": len(conversation),
        "add_message_p95_ms": add_stats.p95_ms,
        "final_prompt_tokens": monitor.count_text(prompt_string),
        "compression_calls": session.compressor.call_count,
    }


@pytest.mark.integration_benchmark
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turns,message_size",
    [
        (10, "small"),
        (25, "medium"),
        (50, "medium"),
        (100, "small"),
    ],
)
async def test_add_message_latency_stable_at_scale(turns: int, message_size: str) -> None:
    results = await _run_scale_scenario(turns, message_size)
    assert results["add_message_p95_ms"] < 100.0
    assert results["final_prompt_tokens"] > 0


@pytest.mark.integration_benchmark
@pytest.mark.asyncio
async def test_prompt_size_grows_sublinearly_with_turns() -> None:
    small = await _run_scale_scenario(10, "medium")
    large = await _run_scale_scenario(50, "medium")

    # Hierarchical compression should keep prompt growth bounded.
    ratio = large["final_prompt_tokens"] / small["final_prompt_tokens"]  # type: ignore[operator]
    assert ratio < 5.0
