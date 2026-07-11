"""TokenMonitor microbenchmarks."""

from __future__ import annotations

import pytest

from sawtooth_memory.monitor import TokenMonitor
from sawtooth_memory.state import MemoryState


@pytest.mark.benchmark(group="token_monitor")
class TestTokenMonitorBenchmark:
    def test_count_text_short(self, benchmark, token_monitor: TokenMonitor) -> None:
        text = "Tell me about quantum entanglement and wave-function collapse."
        benchmark(token_monitor.count_text, text)

    def test_count_text_long(self, benchmark, token_monitor: TokenMonitor) -> None:
        text = "Quantum physics lecture. " * 2000
        benchmark(token_monitor.count_text, text)

    def test_recount_working_memory(
        self, benchmark, token_monitor: TokenMonitor, populated_state: MemoryState
    ) -> None:
        benchmark(token_monitor.recount_working_memory, populated_state)

    def test_should_trigger_compression(
        self, benchmark, token_monitor: TokenMonitor, populated_state: MemoryState
    ) -> None:
        populated_state.l1_working.token_count = 500
        benchmark(token_monitor.should_trigger_compression, populated_state)
