"""
redis_adapter.py — High-Speed Ephemeral Storage Backend.

Utilizes redis.asyncio to persist MemoryState across distributed clusters
with near-zero latency.
"""

from typing import Optional
import redis.asyncio as redis

from ..state import MemoryState
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

        # Initialize the async connection pool
        self._client = redis.from_url(self.redis_url, decode_responses=True)

    def _get_key(self, session_id: str) -> str:
        """Formats the namespace key for a given session."""
        return f"{self.key_prefix}{session_id}"

    async def load_state(self, session_id: str) -> Optional[MemoryState]:
        """Fetch the JSON payload and hydrate the Pydantic state machine."""
        key = self._get_key(session_id)
        raw_data = await self._client.get(key)

        if not raw_data:
            return None

        # Pydantic v2 flawlessly reconstructs the L0, L1, L1.5, and L2 tiers from JSON
        return MemoryState.model_validate_json(raw_data)

    async def save_state(self, session_id: str, state: MemoryState) -> None:
        """Serialize the state and write to Redis with an optional TTL."""
        key = self._get_key(session_id)

        # Serialize entire state machine to a string
        json_payload = state.model_dump_json()

        if self.ttl_seconds:
            await self._client.setex(key, self.ttl_seconds, json_payload)
        else:
            await self._client.set(key, json_payload)

    async def delete_state(self, session_id: str) -> None:
        """Purge the session from the cache."""
        key = self._get_key(session_id)
        await self._client.delete(key)

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        await self._client.aclose()
