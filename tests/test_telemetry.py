import pytest

from sawtooth_memory.config import ContextManagerConfig
from sawtooth_memory.events.types import EntityAnchoredEvent, L1EvictionEvent
from sawtooth_memory.middleware import ContextManager

pytestmark = pytest.mark.asyncio


async def test_entity_ledger_emits_telemetry(tmp_path):
    config = ContextManagerConfig(
        cloud={"api_key": "dummy", "provider": "openai", "model": "dummy-model"}
    )

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    async with ContextManager(
        system_prompt="Test agent",
        config=config,
        enable_events=True,
        journal_path=tmp_path / "test.jsonl",
    ) as cm:
        cm._event_bus.subscribe("l1_5.entity_anchored", capture_event)

        # Trigger the update directly on the live state object.
        # The new EventBus.emit_nowait() handles the sync->async bridge natively!
        cm._state.l1_5_entities.upsert({"user_uuid": "1234-abcd"})

    # MAGIC: Exiting the 'async with' block automatically calls cm.stop(),
    # which calls bus.drain(), guaranteeing the background task finishes before we assert!
    assert len(captured_events) == 1
    event = captured_events[0]
    assert isinstance(event, EntityAnchoredEvent)
    assert event.entity_key == "user_uuid"
    assert event.entity_value == "1234-abcd"
    assert event.operation == "insert"


async def test_add_message_triggers_eviction_event(mocker, tmp_path):
    # We mock the worker enqueue so we don't actually try to call LLM APIs
    mocker.patch("sawtooth_memory.worker.CompressionWorker.enqueue")

    config = ContextManagerConfig(
        soft_limit_tokens=10,
        hard_limit_tokens=50,
        chunk_size=1,
        cloud={"api_key": "dummy", "provider": "openai", "model": "dummy-model"},
    )

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    async with ContextManager(
        system_prompt="Agent",
        config=config,
        enable_events=True,
        journal_path=tmp_path / "test2.jsonl",
    ) as cm:
        cm._event_bus.subscribe("l1.eviction", capture_event)

        await cm.add_message(
            "user", "This is a long message that definitely exceeds ten tokens by far."
        )

    # Exiting the block flushes all events
    assert len(captured_events) == 1
    event = captured_events[0]
    assert isinstance(event, L1EvictionEvent)
    assert event.trigger == "soft_limit_exceeded"
    assert event.messages_evicted == 1
