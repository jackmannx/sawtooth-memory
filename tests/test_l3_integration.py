"""Integration and regression tests for L3 semantic archival storage."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sawtooth_memory.middleware import ContextManager
from sawtooth_memory.state import MemoryState, SemanticVectorMemory, SystemPrompt
from tests.l3_helpers import InMemorySemanticStorage, make_l3_config


@pytest.mark.asyncio
async def test_l3_semantic_metadata_restored_on_session_reload():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    saved = MemoryState(
        l0_system=SystemPrompt(content="sys"),
        l3_semantic=SemanticVectorMemory(
            chunk_count=7,
            last_indexed_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        ),
    )
    await storage.save_state("test-session", saved)

    config = make_l3_config(storage)
    async with ContextManager("sys", config=config, enable_events=False) as cm:
        assert cm.state.l3_semantic.chunk_count == 7
        assert cm.state.l3_semantic.last_indexed_at is not None
