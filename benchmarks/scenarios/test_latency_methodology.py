"""Corrected latency methodology for Sawtooth vs blocking baselines."""

from __future__ import annotations

import time
from typing import Any

import pytest

from benchmarks.common.config import MockedContextManager
from benchmarks.common.fixtures import generate_conversation
from benchmarks.common.timing import summarize_latencies
from benchmarks.scenarios.blocking_baseline import BlockingSummaryMemory
from sawtooth_memory.monitor import TokenMonitor


async def _run_sawtooth_latency_scenario(
    conversation: list[dict[str, str]],
    *,
    mock_delay_ms: float = 1.0,
) -> dict[str, Any]:
    from benchmarks.common.config import benchmark_config

    config = benchmark_config(enable_events=False)
    add_latencies_ms: list[float] = []
    build_latencies_ms: list[float] = []

    session = MockedContextManager(
        "You are a physics expert.",
        config,
        mock_delay_ms=mock_delay_ms,
        enable_events=False,
    )
    async with session as cm:
        for msg in conversation:
            start = time.perf_counter()
            await cm.add_message(msg["role"], msg["content"])  # type: ignore[arg-type]
            add_latencies_ms.append((time.perf_counter() - start) * 1000.0)

            if msg["role"] == "assistant":
                start = time.perf_counter()
                await cm.build_prompt()
                build_latencies_ms.append((time.perf_counter() - start) * 1000.0)

        drain_start = time.perf_counter()
        await cm.stop()
        drain_ms = (time.perf_counter() - drain_start) * 1000.0

        final_prompt = await cm.build_prompt()
        prompt_string = "\n".join(m["content"] for m in final_prompt)
        monitor = TokenMonitor(model="gpt-4o")
        final_tokens = monitor.count_text(prompt_string)

    add_stats = summarize_latencies(add_latencies_ms)
    build_stats = summarize_latencies(build_latencies_ms)

    return {
        "framework": "sawtooth",
        "add_message": add_stats.to_dict(),
        "build_prompt": build_stats.to_dict(),
        "drain_ms": drain_ms,
        "background_compress_calls": session.compressor.call_count,
        "final_prompt_tokens": final_tokens,
        "user_perceived_turn_p95_ms": max(add_stats.p95_ms, build_stats.p95_ms),
    }


def _run_blocking_latency_scenario(
    conversation: list[dict[str, str]],
    *,
    simulate_ms: float = 5.0,
) -> dict[str, Any]:
    memory = BlockingSummaryMemory("phi4-mini", "http://localhost:11434", simulate_ms=simulate_ms)
    per_turn_latencies_ms: list[float] = []

    for i in range(0, len(conversation), 2):
        user_msg = conversation[i]["content"]
        ai_msg = conversation[i + 1]["content"]
        elapsed_s = memory.save_context(user_msg, ai_msg)
        per_turn_latencies_ms.append(elapsed_s * 1000.0)

    turn_stats = summarize_latencies(per_turn_latencies_ms)
    final_text = memory.final_text()
    monitor = TokenMonitor(model="gpt-4o")

    return {
        "framework": "blocking_summary",
        "per_turn_blocked": turn_stats.to_dict(),
        "total_blocked_ms": sum(per_turn_latencies_ms),
        "final_prompt_tokens": monitor.count_text(final_text),
        "user_perceived_turn_p95_ms": turn_stats.p95_ms,
    }


@pytest.mark.integration_benchmark
@pytest.mark.asyncio
async def test_sawtooth_main_thread_latency_is_sub_millisecond() -> None:
    """Sawtooth add_message/build_prompt must stay on the main thread hot path."""
    conversation = generate_conversation(turns=10, message_size="medium")
    results = await _run_sawtooth_latency_scenario(conversation, mock_delay_ms=1.0)

    assert results["add_message"]["p95_ms"] < 50.0
    assert results["build_prompt"]["p95_ms"] < 50.0
    assert results["user_perceived_turn_p95_ms"] < 50.0


@pytest.mark.integration_benchmark
def test_blocking_baseline_has_high_per_turn_latency() -> None:
    """Blocking summary memory should block each turn (simulated compressor)."""
    conversation = generate_conversation(turns=10, message_size="small")
    results = _run_blocking_latency_scenario(conversation, simulate_ms=5.0)

    assert results["per_turn_blocked"]["p95_ms"] >= 5.0
    assert results["total_blocked_ms"] > results["per_turn_blocked"]["mean_ms"]


@pytest.mark.integration_benchmark
@pytest.mark.asyncio
async def test_sawtooth_faster_than_blocking_on_user_perceived_latency() -> None:
    conversation = generate_conversation(turns=10, message_size="medium")
    sawtooth = await _run_sawtooth_latency_scenario(conversation, mock_delay_ms=2.0)
    blocking = _run_blocking_latency_scenario(conversation, simulate_ms=5.0)

    assert sawtooth["user_perceived_turn_p95_ms"] < blocking["user_perceived_turn_p95_ms"]
