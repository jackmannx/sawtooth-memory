"""Integration and regression tests for L3 semantic archival storage."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

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


@pytest.mark.asyncio
async def test_l3_end_to_end_compression_indexes_vectors():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage, soft_limit_tokens=10, chunk_size=2)

    mock_compress = AsyncMock(
        return_value={
            "narrative_summary": "User asked about routers.",
            "extracted_entities": {},
        }
    )

    with patch(
        "sawtooth_memory.middleware.OllamaCompressor.compress",
        mock_compress,
    ):
        async with ContextManager("sys", config=config, enable_events=False) as cm:
            await cm.add_message("user", "Router firmware v2.4.1 drops packets.")
            await cm.add_message("assistant", "Let's inspect the logs.")
            await cm.add_message("user", "Still failing after reboot.")
            await cm._worker.stop()

    assert len(storage.vectors) >= 1
    assert storage.sessions["test-session"].l3_semantic.chunk_count >= 1
    assert "router" in storage.vectors[0][1].lower()


@pytest.mark.asyncio
async def test_search_semantic_archive_returns_stored_chunks():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        await cm._worker.index_l3_semantic(
            cm.state,
            "USER: Router firmware v2.4.1 drops packets nightly.",
            "manual-cycle",
        )
        results = await cm.search_semantic_archive("router firmware", top_k=3)

    assert len(results) == 1
    assert "router" in results[0].text.lower()
    assert results[0].similarity > 0


@pytest.mark.asyncio
async def test_l3_chunk_count_from_storage():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        await cm._worker.index_l3_semantic(cm.state, "First chunk.", "c1")
        await cm._worker.index_l3_semantic(cm.state, "Second chunk.", "c2")
        count = await cm.l3_chunk_count()

    assert count == 2
    assert cm.state.l3_semantic.chunk_count == 2


@pytest.mark.asyncio
async def test_health_check_reports_l3_status():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage)

    with patch(
        "sawtooth_memory.middleware.OllamaCompressor.ping",
        new_callable=AsyncMock,
    ):
        async with ContextManager("sys", config=config, enable_events=False) as cm:
            report = await cm.health_check()

    assert report["checks"]["l3_semantic_storage"] == "ENABLED"
    assert report["checks"]["l3_embedding_backend"] == "hash"


@pytest.mark.asyncio
async def test_explain_prompt_includes_l3_metadata():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        cm.state.l3_semantic.chunk_count = 3
        trace = cm.explain_prompt()

    assert trace["l3_semantic"]["chunk_count"] == 3
    assert trace["l3_semantic"]["in_prompt"] is False

@pytest.mark.asyncio
async def test_build_prompt_injects_l3_when_enabled():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        await cm._worker.index_l3_semantic(
            cm.state,
            "USER: Router firmware v2.4.1 drops packets nightly.",
            "manual-cycle",
        )
        await cm.add_message("user", "What do we know about routers?")
        prompt = await cm.build_prompt()

        system_content = prompt[0]["content"]
        assert "[ARCHIVE_L3]" in system_content
        assert "Router firmware" in system_content

        trace = cm.explain_prompt()
        assert trace["l3_semantic"]["in_prompt"] is True
        assert len(trace["l3_semantic"]["retrieved_chunks"]) == 1

@pytest.mark.asyncio
async def test_build_prompt_skips_l3_when_opt_out():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage, enable_l3_prompt_retrieval=False)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        await cm._worker.index_l3_semantic(
            cm.state,
            "USER: Router firmware v2.4.1 drops packets nightly.",
            "manual-cycle",
        )
        await cm.add_message("user", "What do we know about routers?")
        prompt = await cm.build_prompt()

        system_content = prompt[0]["content"]
        assert "[ARCHIVE_L3]" not in system_content

@pytest.mark.asyncio
async def test_build_prompt_uses_explicit_retrieval_query():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        await cm._worker.index_l3_semantic(
            cm.state,
            "USER: Router firmware v2.4.1 drops packets nightly.",
            "manual-cycle",
        )
        # Empty L1, but explicit query
        prompt = await cm.build_prompt(retrieval_query="routers")

        system_content = prompt[0]["content"]
        assert "[ARCHIVE_L3]" in system_content
        assert "Router firmware" in system_content

@pytest.mark.asyncio
async def test_build_prompt_respects_token_budget():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage, l3_retrieval_max_tokens=10, l3_retrieval_top_k=5)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        await cm._worker.index_l3_semantic(
            cm.state,
            "USER: A very long chunk about router firmware that drops packets nightly.",
            "c1",
        )
        await cm._worker.index_l3_semantic(
            cm.state,
            "USER: Another long chunk about network switches and firewalls.",
            "c2",
        )
        await cm.add_message("user", "routers and switches")
        prompt = await cm.build_prompt()

        system_content = prompt[0]["content"]
        assert "[ARCHIVE_L3]" in system_content
        # Should only have room for the first chunk
        assert "1." in system_content
        assert "2." not in system_content

@pytest.mark.asyncio
async def test_build_prompt_no_l3_without_query():
    storage = InMemorySemanticStorage(embedding_dimension=64)
    config = make_l3_config(storage)

    async with ContextManager("sys", config=config, enable_events=False) as cm:
        # L3 enabled, empty L1, no retrieval_query
        prompt = await cm.build_prompt()
        system_content = prompt[0]["content"]
        assert "[ARCHIVE_L3]" not in system_content
