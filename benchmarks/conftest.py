"""Shared pytest configuration for the benchmark suite."""

from __future__ import annotations

import pytest

from sawtooth_memory.events.bus import reset_event_bus


@pytest.fixture(autouse=True)
def isolated_event_bus():
    """Reset the global EventBus singleton between benchmark tests."""
    reset_event_bus()
    yield
    reset_event_bus()
