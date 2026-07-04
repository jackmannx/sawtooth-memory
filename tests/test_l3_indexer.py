"""Tests for L3 text chunking and SemanticIndexer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sawtooth_memory.embeddings.hash import HashEmbeddingProvider
from sawtooth_memory.l3_indexer import SemanticIndexer, chunk_text
from sawtooth_memory.state import MemoryState, SystemPrompt
from sawtooth_memory.storage.semantic import SemanticChunkResult


class TestChunkText:
    def test_empty_text_returns_empty(self):
        assert chunk_text("", 100) == []
        assert chunk_text("   ", 100) == []

    def test_short_text_single_chunk(self):
        assert chunk_text("Hello world", 100) == ["Hello world"]

    def test_paragraph_merging(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, 30)
        assert len(chunks) >= 2
        assert all(len(c) <= 30 for c in chunks)

    def test_oversized_paragraph_hard_split(self):
        text = "word " * 500
        chunks = chunk_text(text.strip(), 200)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)


@pytest.mark.asyncio
async def test_semantic_indexer_batch_index():
    storage = AsyncMock()
    storage.upsert_vector_chunks_batch = AsyncMock(return_value=2)

    embedder = HashEmbeddingProvider(dimension=64)
    indexer = SemanticIndexer(storage, embedder, chunk_max_chars=50)

    state = MemoryState(l0_system=SystemPrompt(content="sys"))
    text = "USER: First message.\n\nASSISTANT: First reply.\n\nUSER: Second message."

    inserted = await indexer.index("session-1", text, state)

    assert inserted == 2
    assert state.l3_semantic.chunk_count == 2
    assert state.l3_semantic.last_indexed_at is not None
    storage.upsert_vector_chunks_batch.assert_awaited_once()
    batch = storage.upsert_vector_chunks_batch.call_args[0][1]
    assert len(batch) == 2
    assert all(len(emb) == 64 for _, emb in batch)


@pytest.mark.asyncio
async def test_semantic_indexer_search():
    storage = AsyncMock()
    storage.search_similar = AsyncMock(
        return_value=[SemanticChunkResult(text="found chunk", similarity=0.92)]
    )

    indexer = SemanticIndexer(
        storage, HashEmbeddingProvider(dimension=32), chunk_max_chars=100
    )
    results = await indexer.search("session-1", "router troubleshooting", top_k=3)

    assert len(results) == 1
    assert results[0].text == "found chunk"
    storage.search_similar.assert_awaited_once()


@pytest.mark.asyncio
async def test_semantic_indexer_empty_text():
    storage = AsyncMock()
    indexer = SemanticIndexer(
        storage, HashEmbeddingProvider(dimension=32), chunk_max_chars=100
    )
    state = MemoryState(l0_system=SystemPrompt(content="sys"))

    assert await indexer.index("session-1", "  ", state) == 0
    storage.upsert_vector_chunks_batch.assert_not_called()
