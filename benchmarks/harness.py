"""Benchmark harness runner for comparative E2E reports."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any

from benchmarks.common.config import MockedContextManager, recall_benchmark_config
from benchmarks.common.fixtures import all_needle_values, generate_conversation
from benchmarks.common.reporting import collect_environment, merge_report, write_report
from benchmarks.common.timing import summarize_latencies
from benchmarks.scenarios.blocking_baseline import BlockingSummaryMemory
from benchmarks.scenarios.test_latency_methodology import (
    _run_blocking_latency_scenario,
    _run_sawtooth_latency_scenario,
)
from benchmarks.scenarios.test_recall_suite import _recall_score
from sawtooth_memory.config import OllamaConfig
from sawtooth_memory.middleware import ContextManager
from sawtooth_memory.monitor import TokenMonitor


def _run_langchain_live_benchmark(
    conversation: list[dict[str, str]],
    *,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    memory = BlockingSummaryMemory(model, base_url, simulate_ms=0.0)
    per_turn_latencies_ms: list[float] = []

    for i in range(0, len(conversation), 2):
        user_msg = conversation[i]["content"]
        ai_msg = conversation[i + 1]["content"]
        elapsed_s = memory.save_context(user_msg, ai_msg)
        per_turn_latencies_ms.append(elapsed_s * 1000.0)
        print(
            f"  [Blocking] Turn {i // 2 + 1}: blocked {elapsed_s:.2f}s"
        )

    turn_stats = summarize_latencies(per_turn_latencies_ms)
    final_text = memory.final_text()
    monitor = TokenMonitor(model="gpt-4o")
    needles = all_needle_values()
    needle_retained = all(n in final_text for n in needles)

    return {
        "framework": "blocking_summary_live",
        "per_turn_blocked": turn_stats.to_dict(),
        "total_blocked_ms": sum(per_turn_latencies_ms),
        "final_prompt_tokens": monitor.count_text(final_text),
        "golden_needle_recall_pct": 100.0 if needle_retained else 0.0,
        "user_perceived_turn_p95_ms": turn_stats.p95_ms,
    }


async def _run_sawtooth_mock_benchmark(
    conversation: list[dict[str, str]],
    *,
    mock_delay_ms: float,
) -> dict[str, Any]:
    results = await _run_sawtooth_latency_scenario(
        conversation,
        mock_delay_ms=mock_delay_ms,
    )
    needles = all_needle_values()
    config = recall_benchmark_config()

    session = MockedContextManager(
        "You are a physics expert.",
        config,
        mock_delay_ms=mock_delay_ms,
        enable_events=False,
    )
    async with session as cm:
        for msg in conversation:
            await cm.add_message(msg["role"], msg["content"])  # type: ignore[arg-type]
        final_prompt = await cm.build_prompt()
        prompt_string = "\n".join(m["content"] for m in final_prompt)
        searchable = (
            f"{prompt_string}\n"
            f"{cm.state.l1_5_entities.to_json_str()}\n"
            f"{cm.state.l2_archival.narrative}"
        )

    needle_retained = all(n in searchable for n in needles)
    results["golden_needle_recall_pct"] = 100.0 if needle_retained else 0.0
    return results


async def _run_sawtooth_live_benchmark(
    conversation: list[dict[str, str]],
    *,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    config = recall_benchmark_config()
    config.ollama = OllamaConfig(base_url=base_url, model=model)

    add_latencies_ms: list[float] = []
    build_latencies_ms: list[float] = []

    async with ContextManager("You are a physics expert.", config) as cm:
        for i, msg in enumerate(conversation):
            start = time.perf_counter()
            await cm.add_message(msg["role"], msg["content"])  # type: ignore[arg-type]
            add_latencies_ms.append((time.perf_counter() - start) * 1000.0)

            if msg["role"] == "assistant":
                start = time.perf_counter()
                await cm.build_prompt()
                build_latencies_ms.append((time.perf_counter() - start) * 1000.0)
                print(
                    f"  [Sawtooth] Turn {i // 2 + 1}: "
                    f"main-thread {(add_latencies_ms[-1]):.2f}ms"
                )

        drain_start = time.perf_counter()
        await cm.stop()
        drain_ms = (time.perf_counter() - drain_start) * 1000.0

        final_prompt = await cm.build_prompt()
        prompt_string = "\n".join(m["content"] for m in final_prompt)
        monitor = TokenMonitor(model="gpt-4o")
        searchable = (
            f"{prompt_string}\n"
            f"{cm.state.l1_5_entities.to_json_str()}\n"
            f"{cm.state.l2_archival.narrative}"
        )
        needles = all_needle_values()
        needle_retained = all(n in searchable for n in needles)

    add_stats = summarize_latencies(add_latencies_ms)
    build_stats = summarize_latencies(build_latencies_ms)

    return {
        "framework": "sawtooth_live",
        "add_message": add_stats.to_dict(),
        "build_prompt": build_stats.to_dict(),
        "drain_ms": drain_ms,
        "final_prompt_tokens": monitor.count_text(prompt_string),
        "golden_needle_recall_pct": 100.0 if needle_retained else 0.0,
        "user_perceived_turn_p95_ms": max(add_stats.p95_ms, build_stats.p95_ms),
    }


async def run_harness(
    *,
    turns: int = 10,
    message_size: str = "medium",
    mode: str = "mock",
    mock_delay_ms: float = 1.0,
    blocking_simulate_ms: float = 5.0,
    output: Path | None = None,
) -> dict[str, Any]:
    conversation = generate_conversation(turns=turns, message_size=message_size)
    env = collect_environment()
    env["scenario"] = {
        "turns": turns,
        "messages": len(conversation),
        "message_size": message_size,
        "mode": mode,
    }

    print("=" * 64)
    print(" SAWTOOTH MEMORY BENCHMARK HARNESS")
    print("=" * 64)
    print(f"Messages: {len(conversation)} | Mode: {mode} | Size: {message_size}")

    if mode == "live":
        model = os.getenv("BENCHMARK_LOCAL_MODEL", "phi4-mini")
        base_url = os.getenv("BENCHMARK_OLLAMA_URL", "http://localhost:11434")
        blocking = _run_langchain_live_benchmark(
            conversation,
            model=model,
            base_url=base_url,
        )
        sawtooth = await _run_sawtooth_live_benchmark(
            conversation,
            model=model,
            base_url=base_url,
        )
    else:
        blocking = _run_blocking_latency_scenario(
            conversation,
            simulate_ms=blocking_simulate_ms,
        )
        blocking["golden_needle_recall_pct"] = 0.0
        sawtooth = await _run_sawtooth_mock_benchmark(
            conversation,
            mock_delay_ms=mock_delay_ms,
        )

    recall = await _recall_score(turns, message_size=message_size)

    report = merge_report(
        env,
        comparison={
            "blocking_summary": blocking,
            "sawtooth": sawtooth,
        },
        recall_suite=recall,
    )

    if blocking["user_perceived_turn_p95_ms"] > 0:
        speedup = round(
            blocking["user_perceived_turn_p95_ms"]
            / max(sawtooth["user_perceived_turn_p95_ms"], 0.001),
            1,
        )
        report["speedup_user_perceived_p95"] = speedup

    _print_summary(blocking, sawtooth, recall)

    if output is not None:
        path = write_report(output, report)
        print(f"\nReport written to {path}")

    return report


def _print_summary(
    blocking: dict[str, Any],
    sawtooth: dict[str, Any],
    recall: dict[str, object],
) -> None:
    print("\n" + "=" * 64)
    print(f"{'Metric':<34} | {'Blocking':<12} | {'Sawtooth':<12}")
    print("-" * 64)
    print(
        f"{'User-perceived turn p95 (ms)':<34} | "
        f"{blocking['user_perceived_turn_p95_ms']:<12.2f} | "
        f"{sawtooth['user_perceived_turn_p95_ms']:<12.2f}"
    )
    print(
        f"{'Final prompt tokens':<34} | "
        f"{blocking['final_prompt_tokens']:<12} | "
        f"{sawtooth['final_prompt_tokens']:<12}"
    )
    print(
        f"{'Golden needle recall (%)':<34} | "
        f"{blocking.get('golden_needle_recall_pct', 0.0):<12} | "
        f"{sawtooth.get('golden_needle_recall_pct', 0.0):<12}"
    )
    print(
        f"{'Needle suite recall (%)':<34} | "
        f"{'n/a':<12} | "
        f"{recall['recall_pct']:<12}"
    )
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sawtooth benchmark harness")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument(
        "--message-size",
        choices=["small", "medium", "large"],
        default="medium",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default=os.getenv("BENCHMARK_MODE", "mock"),
    )
    parser.add_argument("--mock-delay-ms", type=float, default=1.0)
    parser.add_argument("--blocking-simulate-ms", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/latest.json"),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    asyncio.run(
        run_harness(
            turns=args.turns,
            message_size=args.message_size,
            mode=args.mode,
            mock_delay_ms=args.mock_delay_ms,
            blocking_simulate_ms=args.blocking_simulate_ms,
            output=args.output,
        )
    )
    elapsed = time.perf_counter() - start
    print(f"\nHarness completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
