"""Integration tests for ingest-time entity scanning and pin_entity."""

import pytest

from sawtooth_memory.config import ContextManagerConfig
from sawtooth_memory.middleware import ContextManager


@pytest.mark.asyncio
async def test_ingest_scan_captures_salience_entity():
    config = ContextManagerConfig(
        soft_limit_tokens=100_000,
        enable_ingest_entity_scan=True,
        enable_deterministic_ner=True,
        enable_salience_extractor=True,
        salience_threshold=0.4,
    )
    async with ContextManager("Test agent.", config=config) as cm:
        await cm.add_message(
            "user",
            "Please escalate ticket INC-4421 to the on-call engineer.",
        )
        values = [
            v
            for history in cm.state.l1_5_entities.entities.values()
            for v in history
        ]
        assert "INC-4421" in values


@pytest.mark.asyncio
async def test_ingest_scan_disabled():
    config = ContextManagerConfig(
        soft_limit_tokens=100_000,
        enable_ingest_entity_scan=False,
        enable_deterministic_ner=True,
    )
    async with ContextManager("Test agent.", config=config) as cm:
        await cm.add_message("user", "Escalate ticket INC-4421 now.")
        assert len(cm.state.l1_5_entities.entities) == 0


@pytest.mark.asyncio
async def test_pin_entity(tmp_path):
    config = ContextManagerConfig(soft_limit_tokens=100_000)
    journal_file = tmp_path / "audit.jsonl"
    async with ContextManager(
        "Test agent.", config=config, journal_path=journal_file
    ) as cm:
        await cm.pin_entity("tracking_code", "ALPHA-991")
        assert cm.state.l1_5_entities.get_latest("tracking_code") == "ALPHA-991"

    trace = cm.explain_prompt()
    entity = next(
        e for e in trace["l1_5_entities"] if e["entity_key"] == "tracking_code"
    )
    assert entity["confidence"] == "100% (Pinned)"
