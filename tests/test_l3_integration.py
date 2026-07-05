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


@pytest.mark.asyncio
async def test_hard_truncate_indexes_l3():
    from unittest.mock import patch

    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage, hard_limit_tokens=30, chunk_size=1)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        with patch.object(cm._monitor, "exceeds_hard_limit", return_value=True):
            with patch.object(
                cm._monitor, "should_trigger_compression", return_value=False
            ):
                await cm.add_message(
                    "user",
                    "Critical router diagnostics payload that must survive truncation.",
                )

    assert len(storage.vectors) == 1
    assert storage.sessions["test-session"].l3_semantic.chunk_count == 1
    assert "router" in storage.vectors[0][1].lower()


@pytest.mark.asyncio
async def test_embedder_closed_on_context_manager_stop():
    from unittest.mock import AsyncMock, patch

    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage)

    mock_close = AsyncMock()
    with patch(
        "sawtooth_memory.middleware.create_embedding_provider",
    ) as mock_factory:
        embedder = AsyncMock()
        embedder.close = mock_close
        mock_factory.return_value = embedder

        async with ContextManager("sys", config=config, enable_events=False):
            pass

    mock_close.assert_awaited_once()
