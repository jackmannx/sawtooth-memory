"""
tests/test_sync_wrapper.py

Tests the synchronous blocking portal wrapper to ensure it safely
bridges standard sync execution with the async background worker.
"""

import pytest
from pathlib import Path
from sawtooth_memory.config import ContextManagerConfig
from sawtooth_memory.sync_wrapper import SawtoothSyncWrapper


def test_sync_wrapper_out_of_context():
    """Verify that calling methods outside the context manager safely raises a RuntimeError."""
    config = ContextManagerConfig(soft_limit_tokens=1000)
    wrapper = SawtoothSyncWrapper(system_prompt="Test", config=config)

    with pytest.raises(RuntimeError, match="must be used within a 'with' context"):
        wrapper.add_message("user", "Hello")

    with pytest.raises(RuntimeError, match="must be used within a 'with' context"):
        wrapper.build_prompt()


def test_sync_wrapper_lifecycle_and_health():
    """Verify that the wrapper initializes, connects to the portal, and tears down cleanly."""
    config = ContextManagerConfig(soft_limit_tokens=1000)

    with SawtoothSyncWrapper(
        system_prompt="Test Agent", config=config, enable_events=False
    ) as memory:
        health = memory.health_check()
        assert health["status"] == "healthy"
        assert health["checks"]["configuration"] == "OK"


def test_sync_wrapper_core_pipeline(tmp_path: Path):
    """Test the full synchronous addition, state mutation, and retrieval pipeline."""
    config = ContextManagerConfig(soft_limit_tokens=1000)
    journal_file = tmp_path / "sync_audit.jsonl"

    with SawtoothSyncWrapper(
        system_prompt="You are a helpful assistant.",
        config=config,
        enable_events=True,
        journal_path=journal_file,
    ) as memory:
        # 1. Add messages synchronously (blocks until safe)
        memory.add_message("user", "Hello world!")
        memory.add_message("assistant", "Greetings! How can I help?")

        # 2. Check stats across the thread boundary
        stats = memory.get_stats()
        assert stats["l1_message_count"] == 2
        assert stats["l0_tokens"] > 0

        # 3. Build prompt and verify compilation
        prompt = memory.build_prompt()
        assert len(prompt) == 3  # 1 system message + 2 L1 messages

        # Verify L0 injection
        assert prompt[0]["role"] == "system"
        assert "You are a helpful assistant." in prompt[0]["content"]

        # Verify L1 injection
        assert prompt[1]["role"] == "user"
        assert prompt[1]["content"] == "Hello world!"
        assert prompt[2]["role"] == "assistant"
        assert "Greetings!" in prompt[2]["content"]


def test_sync_wrapper_explainability(tmp_path: Path):
    """Test that explain_prompt correctly fetches and formats the trace synchronously."""
    config = ContextManagerConfig(soft_limit_tokens=1000)
    journal_file = tmp_path / "sync_audit_explain.jsonl"

    with SawtoothSyncWrapper(
        system_prompt="Explainability Test Sync",
        config=config,
        enable_events=True,
        journal_path=journal_file,
    ) as memory:
        trace = memory.explain_prompt()

        assert "l0_system" in trace
        assert trace["l0_system"]["content"] == "Explainability Test Sync"
        assert trace["l1_working_messages"] == 0
        assert isinstance(trace["l1_5_entities"], list)
