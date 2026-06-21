"""
tests/test_postgres_adapter.py

Validates the PostgresStorageAdapter serialization, deserialization, and
state management using asynchronous mocks.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sawtooth_memory.state import MemoryState, Message, SystemPrompt, WorkingMemory
from sawtooth_memory.storage.postgres_adapter import PostgresStorageAdapter


def _make_pool_mock(mock_conn: AsyncMock) -> MagicMock:
    mock_pool = MagicMock()

    @asynccontextmanager
    async def acquire():
        yield mock_conn

    mock_pool.acquire = acquire
    mock_pool.close = AsyncMock()
    return mock_pool


def _wire_adapter(mock_conn: AsyncMock) -> PostgresStorageAdapter:
    adapter = PostgresStorageAdapter(
        dsn="postgresql://user:pass@localhost:5432/sawtooth_db",
        embedding_dimension=1536,
    )
    adapter._pool = _make_pool_mock(mock_conn)
    adapter._schema_ready = True
    return adapter


@pytest.mark.asyncio
async def test_postgres_save_load_delete_lifecycle():
    """Verify that MemoryState accurately converts to JSONB and back."""
    mock_conn = AsyncMock()

    @asynccontextmanager
    async def transaction():
        yield

    mock_conn.transaction = transaction
    adapter = _wire_adapter(mock_conn)

    session_id = "user_alpha_99"
    original_state = MemoryState(
        l0_system=SystemPrompt(content="You are a distributed agent."),
        l1_working=WorkingMemory(
            messages=[Message(role="user", content="Hello from pod B.")]
        ),
    )

    await adapter.save_state(session_id, original_state)

    save_calls = mock_conn.execute.call_args_list
    assert len(save_calls) == 2
    lock_sql = save_calls[0][0][0]
    upsert_sql = save_calls[1][0][0]
    assert "FOR UPDATE" in lock_sql
    assert "INSERT INTO sawtooth_sessions" in upsert_sql
    assert save_calls[1][0][1] == session_id

    saved_json_payload = save_calls[1][0][2]
    assert "You are a distributed agent." in saved_json_payload
    assert "Hello from pod B." in saved_json_payload

    mock_conn.fetchrow.return_value = {"state_payload": saved_json_payload}
    loaded_state = await adapter.load_state(session_id)

    mock_conn.fetchrow.assert_called_once()
    fetch_sql = mock_conn.fetchrow.call_args[0][0]
    assert "SELECT state_payload FROM sawtooth_sessions" in fetch_sql
    assert mock_conn.fetchrow.call_args[0][1] == session_id

    assert loaded_state is not None
    assert loaded_state.l0_system.content == "You are a distributed agent."
    assert len(loaded_state.l1_working.messages) == 1
    assert loaded_state.l1_working.messages[0].content == "Hello from pod B."

    await adapter.delete_state(session_id)
    mock_conn.execute.assert_called_with(
        "DELETE FROM sawtooth_sessions WHERE session_id = $1",
        session_id,
    )


@pytest.mark.asyncio
async def test_postgres_load_returns_none_for_missing_session():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None
    adapter = _wire_adapter(mock_conn)

    loaded_state = await adapter.load_state("missing_session")

    assert loaded_state is None


@pytest.mark.asyncio
async def test_postgres_upsert_vector_chunk():
    mock_conn = AsyncMock()
    adapter = _wire_adapter(mock_conn)

    session_id = "enterprise_session_552"
    text = "User initiated troubleshooting for router."
    embedding = [0.1] * 1536

    await adapter.upsert_vector_chunk(session_id, text, embedding)

    mock_conn.execute.assert_called_once()
    sql = mock_conn.execute.call_args[0][0]
    assert "INSERT INTO sawtooth_semantic_vectors" in sql
    assert mock_conn.execute.call_args[0][1:] == (session_id, text, embedding)


@pytest.mark.asyncio
@patch("asyncpg.create_pool", new_callable=AsyncMock)
async def test_postgres_bootstrap_schema_on_first_use(mock_create_pool):
    """Schema bootstrap runs once when the pool is first acquired."""
    mock_conn = AsyncMock()
    mock_pool = _make_pool_mock(mock_conn)
    mock_create_pool.return_value = mock_pool

    adapter = PostgresStorageAdapter(
        dsn="postgresql://user:pass@localhost:5432/sawtooth_db",
        embedding_dimension=384,
    )

    mock_conn.fetchrow.return_value = None
    await adapter.load_state("bootstrap_probe")

    mock_create_pool.assert_awaited_once()
    bootstrap_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
    assert any(
        "CREATE EXTENSION IF NOT EXISTS vector" in sql for sql in bootstrap_calls
    )
    assert any(
        "CREATE TABLE IF NOT EXISTS sawtooth_sessions" in sql for sql in bootstrap_calls
    )
    assert any("VECTOR(384)" in sql for sql in bootstrap_calls)
    assert any("sawtooth_vector_idx" in sql for sql in bootstrap_calls)
