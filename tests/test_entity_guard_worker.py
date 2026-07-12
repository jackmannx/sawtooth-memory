"""Integration tests for the entity guard compression pipeline."""

from unittest.mock import AsyncMock

import pytest

from sawtooth_memory.state import EntityLedger, MemoryState, Message, SystemPrompt
from sawtooth_memory.worker import CompressionTask, CompressionWorker


@pytest.fixture
def mock_compressor():
    compressor = AsyncMock()
    compressor.compress = AsyncMock(
        return_value={
            "narrative_summary": "The incident was escalated.",
            "extracted_entities": {},
        }
    )
    compressor.close = AsyncMock()
    return compressor


@pytest.mark.asyncio
async def test_worker_salvages_salience_entity_dropped_by_llm(mock_compressor):
    """Salience-discovered entities survive when the LLM returns nothing."""
    worker = CompressionWorker(
        compressor=mock_compressor,
        enable_deterministic_ner=True,
        enable_salience_extractor=True,
        salience_threshold=0.4,
        enable_entity_verifier=True,
    )
    await worker.start()

    state = MemoryState(
        l0_system=SystemPrompt(content="test"),
        l1_5_entities=EntityLedger(),
    )
    messages = [
        Message(
            role="user",
            content="Escalate ticket INC-4421 to on-call immediately.",
        )
    ]
    task = CompressionTask(messages=messages, state=state, cycle_id="test-cycle")
    await worker._process(task)

    values = [
        v for history in state.l1_5_entities.entities.values() for v in history
    ]
    assert "INC-4421" in values

    call_kwargs = mock_compressor.compress.call_args
    protected = call_kwargs.kwargs.get("protected_entities") or call_kwargs[1].get(
        "protected_entities"
    )
    assert protected is not None
    assert any(v == "INC-4421" for v in protected.values())

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_regex_multi_match(mock_compressor):
    mock_compressor.compress = AsyncMock(
        return_value={
            "narrative_summary": "Two UUIDs were discussed.",
            "extracted_entities": {},
        }
    )
    worker = CompressionWorker(
        compressor=mock_compressor,
        enable_salience_extractor=False,
    )
    await worker.start()

    state = MemoryState(
        l0_system=SystemPrompt(content="test"),
        l1_5_entities=EntityLedger(),
    )
    text = (
        "First 550e8400-e29b-41d4-a716-446655440000 and "
        "second 6ba7b810-9dad-11d1-80b4-00c04fd430c8."
    )
    task = CompressionTask(
        messages=[Message(role="user", content=text)],
        state=state,
        cycle_id="uuid-cycle",
    )
    await worker._process(task)

    assert state.l1_5_entities.get_latest("uuid") == (
        "550e8400-e29b-41d4-a716-446655440000"
    )
    assert state.l1_5_entities.get_latest("uuid_2") == (
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    )
    await worker.stop()
