"""Unit tests for compression_core shared cycle logic."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sawtooth_memory.compression_core import (
    CompressionCycleInput,
    CompressionEngineConfig,
    fallback_merge_into_state,
    merge_compression_into_state,
    run_compression_cycle_async,
    run_compression_cycle_sync,
)
from sawtooth_memory.exceptions import CompressionError
from sawtooth_memory.ner import NERPipeline
from sawtooth_memory.state import ArchivalMemory, EntityLedger, MemoryState, Message, SystemPrompt


@pytest.fixture
def memory_state():
    return MemoryState(
        l0_system=SystemPrompt(content="test"),
        l1_5_entities=EntityLedger(),
        l2_archival=ArchivalMemory(),
    )


@pytest.fixture
def engine():
    return CompressionEngineConfig(
        ner_pipeline=NERPipeline.from_config(
            enable=True,
            enable_salience=True,
            salience_threshold=0.4,
        ),
        enable_ner=True,
        fallback_truncate=True,
        enable_entity_verifier=True,
    )


def test_merge_compression_into_state(memory_state):
    merge_compression_into_state(
        memory_state,
        {
            "narrative_summary": "Summary text.",
            "extracted_entities": {"ticket_id": "INC-4421"},
        },
    )
    assert "Summary text." in memory_state.l2_archival.narrative
    values = [
        v for history in memory_state.l1_5_entities.entities.values() for v in history
    ]
    assert "INC-4421" in values


def test_fallback_merge_preserves_salience_entities(memory_state, engine):
    messages = [
        Message(role="user", content="Escalate ticket INC-4421 immediately.")
    ]
    fallback_merge_into_state(
        memory_state,
        messages,
        engine.ner_pipeline,
        enable_ner=True,
    )
    values = [
        v for history in memory_state.l1_5_entities.entities.values() for v in history
    ]
    assert "INC-4421" in values
    assert "COMPRESSION UNAVAILABLE" in memory_state.l2_archival.narrative


def test_run_compression_cycle_sync_success(memory_state, engine):
    compressor = MagicMock()
    compressor.compress.return_value = {
        "narrative_summary": "Incident escalated.",
        "extracted_entities": {},
    }

    messages = [
        Message(role="user", content="Escalate ticket INC-4421 immediately.")
    ]
    outcome = run_compression_cycle_sync(
        CompressionCycleInput(messages=messages, state=memory_state),
        compressor,
        engine,
    )

    assert outcome.success is True
    assert "Incident escalated." in memory_state.l2_archival.narrative
    values = [
        v for history in memory_state.l1_5_entities.entities.values() for v in history
    ]
    assert "INC-4421" in values
    compressor.compress.assert_called_once()


def test_run_compression_cycle_sync_fallback(memory_state, engine):
    compressor = MagicMock()
    compressor.compress.side_effect = CompressionError("backend down")

    messages = [Message(role="user", content="Ticket INC-4421")]
    outcome = run_compression_cycle_sync(
        CompressionCycleInput(messages=messages, state=memory_state),
        compressor,
        engine,
    )

    assert outcome.success is False
    assert outcome.fallback_used is True
    assert "COMPRESSION UNAVAILABLE" in memory_state.l2_archival.narrative


@pytest.mark.asyncio
async def test_run_compression_cycle_async_success(memory_state, engine):
    compressor = AsyncMock()
    compressor.compress = AsyncMock(
        return_value={
            "narrative_summary": "Async summary.",
            "extracted_entities": {},
        }
    )

    messages = [
        Message(role="user", content="Escalate ticket INC-4421 immediately.")
    ]
    outcome = await run_compression_cycle_async(
        CompressionCycleInput(messages=messages, state=memory_state),
        compressor,
        engine,
    )

    assert outcome.success is True
    assert "Async summary." in memory_state.l2_archival.narrative
