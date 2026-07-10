"""
redis_adapter.py — High-Speed Ephemeral Storage Backend.

Utilizes redis.asyncio to persist MemoryState across distributed clusters
with near-zero latency.
"""

import json
from typing import Any, Optional

try:
    import redis.asyncio as redis
except ImportError as exc:  # pragma: no cover - exercised when redis extra missing
    redis = None  # type: ignore[assignment]
    _REDIS_IMPORT_ERROR = exc
else:
    _REDIS_IMPORT_ERROR = None

from ..state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    SemanticVectorMemory,
    SystemPrompt,
    WorkingMemory,
)
from .base import BaseStorageAdapter


class RedisStorageAdapter(BaseStorageAdapter):
    """
    Async Redis implementation of the Sawtooth storage contract.
    Stores the MemoryState as a serialized JSON string.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "sawtooth:session:",
        ttl_seconds: Optional[int] = 86400,  # Default 24-hour expiration
    ) -> None:
        """
        Initialize the Redis connection pool.

        Args:
            redis_url: Standard Redis connection string.
            key_prefix: Namespace to prevent key collisions in shared clusters.
            ttl_seconds: How long to keep inactive sessions alive (Time-To-Live).
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

        if redis is None:
            raise ImportError(
                "Redis support requires the optional redis dependency. "
                "Install with: pip install sawtooth-memory[redis]"
            ) from _REDIS_IMPORT_ERROR

        # Initialize the async connection pool
        self._client = redis.from_url(self.redis_url, decode_responses=True)

    def _get_key(self, session_id: str) -> str:
        """Formats the namespace key for a given session's private L1 state."""
        return f"{self.key_prefix}{session_id}:l1"

    @staticmethod
    def _get_pool_key(pool_id: str) -> str:
        """Formats the namespace key for shared multi-agent pool state."""
        return f"sawtooth:pool:{pool_id}:shared"

    async def load_state(self, session_id: str) -> Optional[MemoryState]:
        """Fetch the private session payload and hydrate L0/L1/L3 metadata."""
        key = self._get_key(session_id)
        raw_data = await self._client.get(key)

        if not raw_data:
            return None

        payload: dict[str, Any] = json.loads(raw_data)
        system_payload = payload.get("l0_system", {})
        l1_payload = payload.get("l1_working", {})
        l3_payload = payload.get("l3_semantic", {})

        return MemoryState(
            l0_system=SystemPrompt.model_validate(system_payload),
            l1_working=WorkingMemory.model_validate(l1_payload),
            l1_5_entities=EntityLedger(),
            l2_archival=ArchivalMemory(),
            l3_semantic=SemanticVectorMemory.model_validate(l3_payload),
        )

    async def save_state(self, session_id: str, state: MemoryState) -> None:
        """Serialize and write private per-session L0/L1/L3 metadata with optional TTL."""
        key = self._get_key(session_id)
        payload = {
            "l0_system": state.l0_system.model_dump(mode="json"),
            "l1_working": state.l1_working.model_dump(mode="json"),
            "l3_semantic": state.l3_semantic.model_dump(mode="json"),
        }
        json_payload = json.dumps(payload)

        if self.ttl_seconds:
            await self._client.setex(key, self.ttl_seconds, json_payload)
        else:
            await self._client.set(key, json_payload)

    async def delete_state(self, session_id: str) -> None:
        """Purge the session from the cache."""
        key = self._get_key(session_id)
        await self._client.delete(key)

    async def load_pool_state(
        self, pool_id: str
    ) -> Optional[tuple[EntityLedger, ArchivalMemory]]:
        """Fetch shared L1.5 + L2 state for all agents in a pool."""
        key = self._get_pool_key(pool_id)
        raw_data = await self._client.get(key)
        if not raw_data:
            return None

        payload: dict[str, Any] = json.loads(raw_data)
        entities = EntityLedger.model_validate(payload.get("l1_5_entities", {}))
        archive = ArchivalMemory.model_validate(payload.get("l2_archival", {}))
        return entities, archive

    async def save_pool_state(
        self, pool_id: str, entities: EntityLedger, archive: ArchivalMemory
    ) -> None:
        """Persist shared L1.5 + L2 state for all agents in a pool."""
        key = self._get_pool_key(pool_id)
        payload = {
            "l1_5_entities": entities.model_dump(mode="json"),
            "l2_archival": archive.model_dump(mode="json"),
        }
        json_payload = json.dumps(payload)
        if self.ttl_seconds:
            await self._client.setex(key, self.ttl_seconds, json_payload)
        else:
            await self._client.set(key, json_payload)

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        await self._client.aclose()
