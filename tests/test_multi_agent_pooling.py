"""
tests/test_multi_agent_pooling.py

Validates multi-agent shared pool synchronization across ContextManager instances.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from sawtooth_memory.config import ContextManagerConfig
from sawtooth_memory.middleware import ContextManager
from sawtooth_memory.state import ArchivalMemory, EntityLedger, MemoryState
from sawtooth_memory.storage.base import BaseStorageAdapter


class MockRedisStorageAdapter(BaseStorageAdapter):
    """In-memory async adapter used to simulate Redis-backed pooling."""

    def __init__(self) -> None:
        self.session_store: dict[str, MemoryState] = {}
        self.pool_store: dict[str, tuple[EntityLedger, ArchivalMemory]] = {}
        self.load_state_mock = AsyncMock()
        self.save_state_mock = AsyncMock()
        self.delete_state_mock = AsyncMock()
        self.load_pool_state_mock = AsyncMock()
        self.save_pool_state_mock = AsyncMock()

    async def load_state(self, session_id: str) -> MemoryState | None:
        await self.load_state_mock(session_id)
        state = self.session_store.get(session_id)
        return None if state is None else state.model_copy(deep=True)

    async def save_state(self, session_id: str, state: MemoryState) -> None:
        await self.save_state_mock(session_id, state)
        self.session_store[session_id] = state.model_copy(deep=True)

    async def delete_state(self, session_id: str) -> None:
        await self.delete_state_mock(session_id)
        self.session_store.pop(session_id, None)

    async def load_pool_state(
        self, pool_id: str
    ) -> tuple[EntityLedger, ArchivalMemory] | None:
        await self.load_pool_state_mock(pool_id)
        state = self.pool_store.get(pool_id)
        if state is None:
            return None
        entities, archive = state
        return entities.model_copy(deep=True), archive.model_copy(deep=True)

    async def save_pool_state(
        self, pool_id: str, entities: EntityLedger, archive: ArchivalMemory
    ) -> None:
        await self.save_pool_state_mock(pool_id, entities, archive)
        self.pool_store[pool_id] = (
            entities.model_copy(deep=True),
            archive.model_copy(deep=True),
        )


@pytest.mark.asyncio
async def test_multi_agent_pool_push_and_pull_sync():
    adapter = MockRedisStorageAdapter()
    pool_id = "engineering_pool"

    fake_compressor = AsyncMock()
    fake_compressor.compress = AsyncMock(
        return_value={
            "narrative_summary": "User provided transaction id txn_998877_alpha.",
            "extracted_entities": {"transaction_id": "txn_998877_alpha"},
        }
    )
    fake_compressor.close = AsyncMock()

    with patch(
        "sawtooth_memory.middleware.OllamaCompressor",
        return_value=fake_compressor,
    ):
        config_a = ContextManagerConfig(
            soft_limit_tokens=1,
            hard_limit_tokens=500,
            chunk_size=1,
            storage_adapter=adapter,
            session_id="node_a_session",
            pool_id=pool_id,
            enable_deterministic_ner=False,
            compression_mode="always_llm",
        )
        async with ContextManager(
            "You are node A.", config_a, enable_events=False
        ) as a:
            await a.add_message(
                "user", "Please remember this transaction: txn_998877_alpha."
            )
            await asyncio.wait_for(a._worker._queue.join(), timeout=2.0)

        # Push-on-write assertion: Node A compression syncs to the shared pool.
        assert adapter.save_pool_state_mock.await_count >= 1
        pooled = adapter.pool_store.get(pool_id)
        assert pooled is not None
        pooled_entities, pooled_archive = pooled
        assert pooled_entities.get_latest("transaction_id") == "txn_998877_alpha"
        assert "[origin:node_a_session]" in pooled_archive.narrative

        config_b = ContextManagerConfig(
            soft_limit_tokens=100,
            hard_limit_tokens=500,
            chunk_size=1,
            storage_adapter=adapter,
            session_id="node_b_session",
            pool_id=pool_id,
            enable_deterministic_ner=False,
        )
        async with ContextManager(
            "You are node B.", config_b, enable_events=False
        ) as b:
            prompt = await b.build_prompt()

            # Pull-on-read assertion: Node B sees Node A's pooled entity data.
            assert (
                b.state.l1_5_entities.get_latest("transaction_id") == "txn_998877_alpha"
            )
            assert "txn_998877_alpha" in prompt[0]["content"]
            assert "[ENTITY_LEDGER_L1_5]" in prompt[0]["content"]


@pytest.mark.asyncio
async def test_multi_agent_pool_synced_on_start():
    """Pool state should hydrate during start(), not only on build_prompt()."""
    adapter = MockRedisStorageAdapter()
    pool_id = "startup_pool"

    shared_entities = EntityLedger()
    shared_entities.upsert({"cluster_token": "secret_pass_123"})
    shared_archive = ArchivalMemory(narrative="[origin:node_a] Shared deployment context.")
    adapter.pool_store[pool_id] = (shared_entities, shared_archive)

    config = ContextManagerConfig(
        storage_adapter=adapter,
        session_id="node_b_session",
        pool_id=pool_id,
    )

    async with ContextManager("You are node B.", config, enable_events=False) as cm:
        assert cm.state.l1_5_entities.get_latest("cluster_token") == "secret_pass_123"
        assert "Shared deployment context." in cm.state.l2_archival.narrative


@pytest.mark.asyncio
async def test_build_prompt_preserves_unsynced_local_entities():
    """Pool sync must merge shared state without clobbering local-only entities."""
    adapter = MockRedisStorageAdapter()
    pool_id = "merge_pool"

    shared_entities = EntityLedger()
    shared_entities.upsert({"shared_key": "from_pool_only"})
    adapter.pool_store[pool_id] = (shared_entities, ArchivalMemory())

    config = ContextManagerConfig(
        storage_adapter=adapter,
        pool_id=pool_id,
        session_id="agent_b",
        soft_limit_tokens=10000,
        hard_limit_tokens=20000,
    )

    async with ContextManager("Sys.", config, enable_events=False) as cm:
        cm.state.l1_5_entities.upsert({"local_only": "agent_b_secret"})
        await cm.build_prompt()

        assert cm.state.l1_5_entities.get_latest("local_only") == "agent_b_secret"
        assert cm.state.l1_5_entities.get_latest("shared_key") == "from_pool_only"


@pytest.mark.asyncio
async def test_crushed_observation_entity_still_reaches_pool():
    """Regression test: an entity that only appears inside a large tool
    observation (and is therefore replaced by a crushed placeholder before
    it's ever stored in L1) must still propagate to the shared multi-agent
    pool when the message holding it is folded.

    Before the fix, create_fold_unit() re-ran NER over the *stored* (already
    crushed) message text to compute entity_keys, and apply_fold_delta_to_pool
    only pushes keys listed there -- so this entity, despite being safely in
    the local ledger via the ingest-time scan, never reached the pool.
    """
    adapter = MockRedisStorageAdapter()
    pool_id = "crush_pool"
    entity_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    # A JSON array large enough that observation_crush samples it down to
    # head/tail, dropping the middle item that carries our entity.
    items = [f"log entry {i}" for i in range(60)]
    items[30] = f"log entry 30 ref={entity_uuid}"
    import json

    tool_content = json.dumps(items)

    config = ContextManagerConfig(
        storage_adapter=adapter,
        pool_id=pool_id,
        session_id="agent_a",
        soft_limit_tokens=1,
        hard_limit_tokens=5000,
        chunk_size=1,
        compression_mode="dte",
        obs_crush_min_tokens=20,
    )

    async with ContextManager("Sys.", config, enable_events=False) as cm:
        await cm.add_message("tool", tool_content)

        # Sanity check the observation actually got crushed -- otherwise this
        # test wouldn't be exercising the bug at all. (soft_limit_tokens=1
        # means the message may already have been evicted into the fold by
        # the time add_message() returns, so check the DTE savings counter
        # rather than l1_working's contents.)
        assert cm.state.dte.observation_tokens_saved > 0

        await asyncio.wait_for(cm._worker._queue.join(), timeout=2.0)

    pooled = adapter.pool_store.get(pool_id)
    assert pooled is not None
    pooled_entities, _ = pooled
    assert pooled_entities.get_latest("uuid") == entity_uuid


class _RacyPoolAdapter(BaseStorageAdapter):
    """Fake adapter with an injected delay inside ``save_pool_state`` so that
    two concurrent ``merge_pool_state()`` callers are forced to both finish
    reading before either finishes writing -- modeling the network
    round-trip a real backend has for each call. With ``atomic=False`` it
    falls back to BaseStorageAdapter's plain load-modify-save default; with
    ``atomic=True`` it holds a lock across the whole sequence, mirroring what
    PostgresStorageAdapter's ``FOR UPDATE`` transaction and
    RedisStorageAdapter's WATCH/MULTI/EXEC now do in production.
    """

    def __init__(self, *, atomic: bool) -> None:
        self._store: tuple[EntityLedger, ArchivalMemory] | None = None
        self._lock = asyncio.Lock() if atomic else None

    async def load_state(self, session_id: str) -> MemoryState | None:
        raise NotImplementedError

    async def save_state(self, session_id: str, state: MemoryState) -> None:
        raise NotImplementedError

    async def delete_state(self, session_id: str) -> None:
        raise NotImplementedError

    async def load_pool_state(
        self, pool_id: str
    ) -> tuple[EntityLedger, ArchivalMemory] | None:
        if self._store is None:
            return None
        entities, archive = self._store
        return entities.model_copy(deep=True), archive.model_copy(deep=True)

    async def save_pool_state(
        self, pool_id: str, entities: EntityLedger, archive: ArchivalMemory
    ) -> None:
        # The delay sits here (between a caller's read and its write) so two
        # concurrent merge_pool_state() callers are forced to both finish
        # reading before either finishes writing -- the exact window Bug #1
        # exploited.
        await asyncio.sleep(0.02)
        self._store = (entities.model_copy(deep=True), archive.model_copy(deep=True))

    async def merge_pool_state(self, pool_id, merge_fn):
        if self._lock is None:
            return await super().merge_pool_state(pool_id, merge_fn)
        async with self._lock:
            return await super().merge_pool_state(pool_id, merge_fn)


@pytest.mark.asyncio
async def test_pool_merge_default_is_racy_but_locked_adapter_is_not():
    """Characterizes the Bug #1 fix at the storage-contract level.

    Two agents calling merge_pool_state() concurrently must not clobber each
    other's deltas. The unlocked default (what the old load_pool_state() +
    save_pool_state() call sites effectively did) loses an update under a
    forced interleave; holding a lock across the full sequence -- as
    PostgresStorageAdapter and RedisStorageAdapter now do -- does not.
    """

    async def _two_concurrent_merges(adapter: BaseStorageAdapter) -> EntityLedger:
        async def add(key: str, value: str) -> None:
            def _merge(entities: EntityLedger, archive: ArchivalMemory) -> None:
                entities.upsert({key: value})

            await adapter.merge_pool_state("pool", _merge)

        await asyncio.gather(add("x", "from_a"), add("y", "from_b"))
        pool_state = await adapter.load_pool_state("pool")
        assert pool_state is not None
        entities, _ = pool_state
        return entities

    racy_result = await _two_concurrent_merges(_RacyPoolAdapter(atomic=False))
    # Lost-update: both callers read before either wrote, so one caller's
    # upsert never survives the second caller's save.
    assert not (
        racy_result.get_latest("x") == "from_a"
        and racy_result.get_latest("y") == "from_b"
    )

    safe_result = await _two_concurrent_merges(_RacyPoolAdapter(atomic=True))
    assert safe_result.get_latest("x") == "from_a"
    assert safe_result.get_latest("y") == "from_b"
