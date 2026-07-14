"""
multi_agent_pool.py — Share L1.5 entities + L2 archive across agents via pool_id.

Uses an in-memory storage adapter so the example runs without Redis.
In production, swap for RedisStorageAdapter or PostgresStorageAdapter.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from sawtooth_memory import ContextManager, ContextManagerConfig, MemoryState
from sawtooth_memory.state import ArchivalMemory, EntityLedger
from sawtooth_memory.storage import BaseStorageAdapter


class InMemoryPoolStorage(BaseStorageAdapter):
    """Minimal adapter that supports session + pool persistence."""

    def __init__(self) -> None:
        self.sessions: dict[str, MemoryState] = {}
        self.pools: dict[str, tuple[EntityLedger, ArchivalMemory]] = {}

    async def load_state(self, session_id: str) -> MemoryState | None:
        state = self.sessions.get(session_id)
        return None if state is None else state.model_copy(deep=True)

    async def save_state(self, session_id: str, state: MemoryState) -> None:
        self.sessions[session_id] = state.model_copy(deep=True)

    async def delete_state(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def load_pool_state(
        self, pool_id: str
    ) -> tuple[EntityLedger, ArchivalMemory] | None:
        state = self.pools.get(pool_id)
        if state is None:
            return None
        entities, archive = state
        return entities.model_copy(deep=True), archive.model_copy(deep=True)

    async def save_pool_state(
        self, pool_id: str, entities: EntityLedger, archive: ArchivalMemory
    ) -> None:
        self.pools[pool_id] = (
            entities.model_copy(deep=True),
            archive.model_copy(deep=True),
        )


async def main() -> None:
    storage = InMemoryPoolStorage()
    pool_id = "support_desk"

    fake = AsyncMock()
    fake.compress = AsyncMock(
        return_value={
            "narrative_summary": "Agent A logged ticket INC-4421 for customer ACME.",
            "extracted_entities": {"ticket_id": "INC-4421", "customer": "ACME"},
        }
    )
    fake.close = AsyncMock()
    fake.ping = AsyncMock()

    def make_config(session_id: str) -> ContextManagerConfig:
        return ContextManagerConfig(
            storage_adapter=storage,
            session_id=session_id,
            pool_id=pool_id,
            soft_limit_tokens=50,
            hard_limit_tokens=200,
            chunk_size=2,
            compression_mode="always_llm",
            enable_deterministic_ner=True,
            enable_salience_extractor=True,
            enable_ingest_entity_scan=True,
            fallback_truncate=True,
        )

    with patch("sawtooth_memory.middleware.OllamaCompressor", return_value=fake):
        async with ContextManager(
            "Agent A — intake", config=make_config("agent_a")
        ) as agent_a:
            await agent_a.add_message(
                "user", "Please open ticket INC-4421 for ACME Corp."
            )
            await agent_a.add_message(
                "assistant", "Opened INC-4421 for ACME and synced the pool."
            )
            # Force a compression cycle so pool L1.5/L2 are written.
            await agent_a.add_message("user", "Also notify billing.")
            await agent_a.add_message("assistant", "Billing notified.")
            await asyncio.sleep(0.3)

            a_entities = dict(agent_a.state.l1_5_entities.entities)
            print("Agent A entities:", a_entities)

        async with ContextManager(
            "Agent B — resolution", config=make_config("agent_b")
        ) as agent_b:
            # Pull shared pool state on start/sync.
            await agent_b.add_message("user", "What ticket are we resolving?")
            prompt = await agent_b.build_prompt()
            b_entities = dict(agent_b.state.l1_5_entities.entities)

            print("Agent B entities (from pool):", b_entities)
            print("Shared ticket present:", "INC-4421" in str(b_entities))
            print("Compiled prompt messages:", len(prompt))


if __name__ == "__main__":
    asyncio.run(main())
