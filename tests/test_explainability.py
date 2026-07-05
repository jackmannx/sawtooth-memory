import pytest
from sawtooth_memory.config import ContextManagerConfig
from sawtooth_memory.middleware import ContextManager, _extract_entity_event


# ---------------------------------------------------------------------------
# Integration Test: End-to-End Explainability Trace
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_explain_prompt_generates_trace(tmp_path):
    config = ContextManagerConfig(
        cloud={"api_key": "dummy", "provider": "openai", "model": "dummy-model"}
    )
    journal_file = tmp_path / "audit.jsonl"

    async with ContextManager(
        system_prompt="You are a data agent.",
        config=config,
        enable_events=True,
        journal_path=journal_file,
    ) as cm:
        # 1. Trigger an L1.5 mutation
        cm._state.l1_5_entities.upsert({"user_transaction_id": "txn_998877"})

    # --- MAGIC HAPPENS HERE ---
    # Exiting the `async with` block automatically calls `cm.stop()`.
    # This natively calls `bus.drain()` AND `journal.stop()`, guaranteeing
    # all events are 100% flushed and saved to the JSONL file on the hard drive!

    # 2. Generate the trace NOW that the file is safely closed and flushed
    trace = cm.explain_prompt()

    # 3. Assert the L0 and L2 origins are mapped
    assert trace["l0_system"]["origin"] == "Hardcoded System Initialization"
    assert trace["l2_archival"]["origin"] == "Background Ollama Compression (L1 -> L2)"

    # 4. Assert the L1.5 deterministic trace successfully read the journal
    assert len(trace["l1_5_entities"]) == 1
    entity_trace = trace["l1_5_entities"][0]

    assert entity_trace["entity_key"] == "user_transaction_id"

    # Asserting safely regardless of whether the ledger uses string or list schemas
    actual_val = entity_trace["entity_value"]
    if isinstance(actual_val, list):
        assert actual_val == ["txn_998877"]
    else:
        assert actual_val == "txn_998877"

    assert entity_trace["confidence"] == "100% (Deterministic)"

    # With the OOP schema parser in place, this will now successfully find 'insert' or 'upsert'
    assert "insert" in entity_trace["origin"] or "upsert" in entity_trace["origin"]
    assert entity_trace["timestamp"] != "unknown"


# ---------------------------------------------------------------------------
# Unit Tests for the OOP Schema Extractor (_extract_entity_event)
# ---------------------------------------------------------------------------
def test_extract_entity_event_v1_flat_schema():
    record = {
        "channel": "l1_5.entity_anchored",
        "entity_key": "user_id",
        "operation": "insert",
        "timestamp": "12345",
    }
    key, op, ts = _extract_entity_event(record)
    assert key == "user_id"
    assert op == "insert"
    assert ts == "12345"


def test_extract_entity_event_v2_payload_schema():
    record = {
        "event_type": "entity_anchored",
        "payload": {"entity_key": "session_id", "operation": "update"},
        "timestamp": "67890",
    }
    key, op, ts = _extract_entity_event(record)
    assert key == "session_id"
    assert op == "update"
    assert ts == "67890"  # Properly falls back to the root timestamp


def test_extract_entity_event_v3_data_schema():
    record = {
        "channel": "l1_5.entity_anchored",
        "data": {"entity_key": "api_key", "operation": "delete", "timestamp": "99999"},
    }
    key, op, ts = _extract_entity_event(record)
    assert key == "api_key"
    assert op == "delete"
    assert ts == "99999"


def test_extract_entity_event_unknown_schema():
    record = {"unrelated_key": "unrelated_value"}
    key, op, ts = _extract_entity_event(record)
    assert key is None
    assert op == "unknown"
    assert ts == "unknown"


@pytest.mark.asyncio
async def test_explain_prompt_structure():
    """Verify that explain_prompt returns the correct schema."""
    config = ContextManagerConfig(soft_limit_tokens=1000)

    async with ContextManager(system_prompt="You are an AI.", config=config) as cm:
        cm.state.l2_archival.append_narrative("Old summary")
        # CHANGED: 'set' to 'upsert'
        cm.state.l1_5_entities.upsert({"user_id": "12345"})
        trace = cm.explain_prompt()

        assert "l0_system" in trace
        assert trace["l0_system"]["content"] == "You are an AI."
        assert "l2_archival" in trace
        assert trace["l2_archival"]["content"] == "Old summary"
        assert "l1_5_entities" in trace
        assert len(trace["l1_5_entities"]) == 1
        assert trace["l1_5_entities"][0]["entity_key"] == "user_id"
        assert trace["l1_5_entities"][0]["entity_value"] == ["12345"]
        assert "l3_semantic" in trace
        assert trace["l3_semantic"]["in_prompt"] is False
