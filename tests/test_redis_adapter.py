"""
tests/test_redis_adapter.py

Validates the RedisStorageAdapter serialization, deserialization, and
state management using asynchronous mocks.
"""

import pytest
from unittest.mock import AsyncMock, patch

from sawtooth_memory.state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    SystemPrompt,
)
from sawtooth_memory.storage.redis_adapter import RedisStorageAdapter


@pytest.mark.asyncio
@patch("redis.asyncio.from_url")
async def test_redis_save_load_delete_lifecycle(mock_from_url):
    """Verify that the MemoryState accurately converts to JSON and back."""
    # 1. Setup the mocked Redis client
    mock_client = AsyncMock()
    mock_from_url.return_value = mock_client

    adapter = RedisStorageAdapter(redis_url="redis://fake:6379", ttl_seconds=3600)
    session_id = "user_alpha_99"
    expected_key = "sawtooth:session:user_alpha_99:l1"

    # 2. Create a fresh memory state
    original_state = MemoryState(
        l0_system=SystemPrompt(content="You are a distributed agent.")
    )
    # --- TEST SAVE ---
    await adapter.save_state(session_id, original_state)

    # Verify Redis SETEX was called with the correct key and TTL
    mock_client.setex.assert_called_once()
    called_args = mock_client.setex.call_args[0]
    assert called_args[0] == expected_key
    assert called_args[1] == 3600

    # Extract the JSON payload sent to Redis
    saved_json_payload = called_args[2]
    assert "You are a distributed agent." in saved_json_payload

    # --- TEST LOAD ---
    # Simulate Redis returning the JSON string when GET is called
    mock_client.get.return_value = saved_json_payload

    loaded_state = await adapter.load_state(session_id)

    # Verify the Pydantic model hydrated perfectly
    assert loaded_state is not None
    assert loaded_state.l0_system.content == "You are a distributed agent."
    assert loaded_state.l1_working.messages == []
    assert loaded_state.l1_5_entities.entities == {}
    assert loaded_state.l2_archival.narrative == ""

    # --- TEST DELETE ---
    await adapter.delete_state(session_id)
    mock_client.delete.assert_called_once_with(expected_key)


@pytest.mark.asyncio
@patch("redis.asyncio.from_url")
async def test_redis_pool_shared_state_roundtrip(mock_from_url):
    mock_client = AsyncMock()
    mock_from_url.return_value = mock_client

    adapter = RedisStorageAdapter(redis_url="redis://fake:6379", ttl_seconds=3600)
    pool_id = "team_pool_alpha"
    expected_key = "sawtooth:pool:team_pool_alpha:shared"

    entities = EntityLedger()
    entities.upsert({"txn_id": "txn_998877_alpha"})
    archive = ArchivalMemory(narrative="Node A summarized previous failures.")

    await adapter.save_pool_state(pool_id, entities, archive)
    mock_client.setex.assert_called()
    save_args = mock_client.setex.call_args[0]
    assert save_args[0] == expected_key
    payload = save_args[2]
    assert "txn_998877_alpha" in payload
    assert "Node A summarized previous failures." in payload

    mock_client.get.return_value = payload
    loaded = await adapter.load_pool_state(pool_id)
    assert loaded is not None
    loaded_entities, loaded_archive = loaded
    assert loaded_entities.get_latest("txn_id") == "txn_998877_alpha"
    assert "Node A summarized previous failures." in loaded_archive.narrative
