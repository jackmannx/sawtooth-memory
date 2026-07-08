import pytest

from sawtooth_memory.events.bus import EventBus
from sawtooth_memory.events.types import L1EvictionEvent, SawtoothEvent

pytestmark = pytest.mark.asyncio


async def test_event_bus_subscribe_and_emit():
    bus = EventBus()
    received_events = []

    async def mock_handler(event: SawtoothEvent):
        received_events.append(event)

    bus.subscribe("l1.eviction", mock_handler)

    event = L1EvictionEvent(tokens_evicted=500)
    # Await the emit so we can assert immediately
    await bus.emit(event, fire_and_forget=False)

    assert len(received_events) == 1
    assert received_events[0].event_type == "l1.eviction"
    assert received_events[0].tokens_evicted == 500


async def test_event_bus_error_shielding():
    bus = EventBus()
    success_tracker = []

    async def buggy_handler(event: SawtoothEvent):
        raise ValueError("Simulated user logging failure")

    async def good_handler(event: SawtoothEvent):
        success_tracker.append(True)

    # Subscribe both to the same event
    bus.subscribe("base", buggy_handler)
    bus.subscribe("base", good_handler)

    # This should NOT crash, even with fire_and_forget=False
    await bus.emit(SawtoothEvent(), fire_and_forget=False)

    # The good handler should still have executed
    assert len(success_tracker) == 1
