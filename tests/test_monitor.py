"""tests/test_monitor.py — Unit tests for TokenMonitor."""

import pytest

from sawtooth_memory.monitor import TokenMonitor
from sawtooth_memory.state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    Message,
    SystemPrompt,
    WorkingMemory,
)


@pytest.fixture
def monitor():
    return TokenMonitor(model="gpt-4o", soft_limit=100, hard_limit=200)


class TestTokenCounting:
    def test_count_text_nonempty(self, monitor):
        count = monitor.count_text("Hello, world!")
        assert count > 0

    def test_count_text_empty(self, monitor):
        assert monitor.count_text("") == 0

    def test_count_message_includes_overhead(self, monitor):
        msg = Message(role="user", content="Hi")
        count = monitor.count_message(msg)
        text_only = monitor.count_text("Hi")
        assert count == text_only + 4

    def test_longer_content_more_tokens(self, monitor):
        short = monitor.count_text("Hi")
        long = monitor.count_text("Hi " * 50)
        assert long > short


class TestThresholds:
    def test_exceeds_soft_limit_false(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.token_count = 50
        assert not monitor.exceeds_soft_limit(state)

    def test_exceeds_soft_limit_true(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.token_count = 100
        assert monitor.exceeds_soft_limit(state)

    def test_exceeds_hard_limit_false(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.token_count = 150
        assert not monitor.exceeds_hard_limit(state)

    def test_exceeds_hard_limit_true(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.token_count = 200
        assert monitor.exceeds_hard_limit(state)


class TestRecount:
    def test_recount_working_memory(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.messages = [
            Message(role="user", content="Hello", token_count=999),
            Message(role="assistant", content="Hi there", token_count=999),
        ]
        state.l1_working.token_count = 999

        monitor.recount_working_memory(state)

        expected = sum(monitor.count_message(m) for m in state.l1_working.messages)
        assert state.l1_working.token_count == expected
        assert state.l1_working.token_count != 999


def create_dummy_state() -> MemoryState:
    """Helper to safely initialize Pydantic MemoryState with required kwargs."""
    return MemoryState(
        l0_system=SystemPrompt(content="System", token_count=1),
        l1_working=WorkingMemory(),
        l1_5_entities=EntityLedger(),
        l2_archival=ArchivalMemory(),
    )


@pytest.mark.asyncio
async def test_monitor_debouncing():
    """Test that the monitor locks the queue to prevent API flooding."""
    state = create_dummy_state()
    monitor = TokenMonitor(soft_limit=10)

    msg = Message(role="user", content="This is a very long message to trigger limits.")
    state.l1_working.append(msg)
    monitor.recount_working_memory(state)

    # First check should trigger compression and lock the debouncer
    assert monitor.should_trigger_compression(state) is True
    assert monitor._is_compression_queued is True

    # Second check should return False (Debounced!) even though limits are still exceeded
    assert monitor.should_trigger_compression(state) is False

    # Simulate background worker finishing
    await monitor._on_compression_done(None)

    # Lock should be released
    assert monitor._is_compression_queued is False


def test_monitor_turn_based_batching():
    """Test that max_unsummarized_turns triggers compression before token limits."""
    state = create_dummy_state()
    monitor = TokenMonitor(soft_limit=5000, max_unsummarized_turns=3)

    # Add 2 short messages (Tokens are low, Turns = 2)
    state.l1_working.append(Message(role="user", content="Hi"))
    state.l1_working.append(Message(role="assistant", content="Hello"))
    assert monitor.should_trigger_compression(state) is False

    # Add 3rd message (Hits the turn batching limit)
    state.l1_working.append(Message(role="user", content="Test"))
    assert monitor.should_trigger_compression(state) is True
