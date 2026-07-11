"""In-memory storage adapter benchmarks."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from benchmarks.common.config import MockedContextManager, benchmark_config
from benchmarks.common.mock_compressor import MockCompressor
from benchmarks.common.timing import summarize_latencies
from sawtooth_memory.sync_wrapper import SawtoothSyncWrapper
from tests.l3_helpers import InMemorySemanticStorage


@pytest.mark.integration_benchmark
@pytest.mark.asyncio
async def test_storage_save_load_latency() -> None:
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = benchmark_config(
        storage_adapter=storage,
        session_id="bench-session",
        enable_events=False,
    )
    save_latencies_ms: list[float] = []

    session = MockedContextManager(
        "You are a physics expert.",
        config,
        enable_events=False,
    )
    async with session as cm:
        for i in range(50):
            start = time.perf_counter()
            await cm.add_message("user", f"Persisted message {i} with enough tokens.")
            save_latencies_ms.append((time.perf_counter() - start) * 1000.0)

    stats = summarize_latencies(save_latencies_ms)
    assert storage.sessions["bench-session"] is not None
    assert stats.p95_ms < 150.0


@pytest.mark.integration_benchmark
def test_sync_wrapper_add_message_overhead() -> None:
    config = benchmark_config(enable_events=False)
    latencies_ms: list[float] = []
    mock = MockCompressor(delay_ms=0.0)

    with patch("sawtooth_memory.middleware.OllamaCompressor", return_value=mock):
        with SawtoothSyncWrapper(
            "You are a physics expert.",
            config,
            enable_events=False,
        ) as wrapper:
            for i in range(20):
                start = time.perf_counter()
                wrapper.add_message("user", f"Sync message {i}")
                latencies_ms.append((time.perf_counter() - start) * 1000.0)

    stats = summarize_latencies(latencies_ms)
    assert stats.p95_ms < 200.0
