"""Tests for L3 integration in CompressionWorker."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sawtooth_memory.embeddings.hash import HashEmbeddingProvider
from sawtooth_memory.exceptions import CompressionError
from sawtooth_memory.l3_indexer import SemanticIndexer
from sawtooth_memory.state import MemoryState, Message, SystemPrompt
from sawtooth_memory.worker import CompressionTask, CompressionWorker


@pytest.fixture
def mock_compressor():
    compressor = AsyncMock()
    compressor.compress = AsyncMock(
        return_value={
            "narrative_summary": "User discussed network issues.",
            "extracted_entities": {"ticket_id": "INC-42"},
        }
    )
    compressor.close = AsyncMock()
    return compressor


@pytest.fixture
def semantic_storage():
    storage = AsyncMock()
    storage.upsert_vector_chunks_batch = AsyncMock(return_value=1)
    storage.save_state = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_worker_indexes_l3_on_compression(mock_compressor, semantic_storage):
    embedder = HashEmbeddingProvider(dimension=64)
    indexer = SemanticIndexer(semantic_storage, embedder, chunk_max_chars=500)

    event_bus = MagicMock()
    event_bus.emit = AsyncMock()

    worker = CompressionWorker(
        compressor=mock_compressor,
        event_bus=event_bus,
        l3_indexer=indexer,
        storage_adapter=semantic_storage,
        session_id="agent-1",
        embedding_backend="hash",
        embedding_model="hash-local",
        enable_deterministic_ner=False,
    )

    state = MemoryState(l0_system=SystemPrompt(content="sys"))
    messages = [
        Message(role="user", content="My router keeps dropping packets."),
        Message(role="assistant", content="Let's check the firmware version."),
    ]
    task = CompressionTask(messages=messages, state=state, cycle_id="cycle-1")

    await worker._process(task)

    semantic_storage.upsert_vector_chunks_batch.assert_awaited_once()
    assert state.l3_semantic.chunk_count == 1
    semantic_storage.save_state.assert_awaited_once()

    emitted_events = [call.args[0] for call in event_bus.emit.call_args_list]
    l3_events = [e for e in emitted_events if e.event_type == "l3.vector_indexed"]
    assert len(l3_events) == 1
    assert l3_events[0].chunks_indexed == 1


@pytest.mark.asyncio
async def test_worker_skips_l3_when_indexer_disabled(mock_compressor):
    worker = CompressionWorker(
        compressor=mock_compressor,
        enable_deterministic_ner=False,
    )
    state = MemoryState(l0_system=SystemPrompt(content="sys"))
    task = CompressionTask(
        messages=[Message(role="user", content="hello")],
        state=state,
        cycle_id="cycle-2",
    )

    await worker._process(task)

    assert state.l3_semantic.chunk_count == 0


@pytest.mark.asyncio
async def test_worker_indexes_l3_on_compression_fallback(mock_compressor, semantic_storage):
    mock_compressor.compress = AsyncMock(side_effect=CompressionError("provider down"))

    embedder = HashEmbeddingProvider(dimension=64)
    indexer = SemanticIndexer(semantic_storage, embedder, chunk_max_chars=500)

    worker = CompressionWorker(
        compressor=mock_compressor,
        fallback_truncate=True,
        l3_indexer=indexer,
        storage_adapter=semantic_storage,
        session_id="agent-1",
        enable_deterministic_ner=False,
    )

    state = MemoryState(l0_system=SystemPrompt(content="sys"))
    task = CompressionTask(
        messages=[Message(role="user", content="Salvage this text on fallback.")],
        state=state,
        cycle_id="fallback-1",
    )

    await worker._process(task)

    semantic_storage.upsert_vector_chunks_batch.assert_awaited_once()
    assert state.l3_semantic.chunk_count == 1
