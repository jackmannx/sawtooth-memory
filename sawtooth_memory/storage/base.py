"""
base.py — Distributed Storage Architecture Contract.

Defines the Abstract Base Class for all remote state persistence.
Ensures ContextManager and CompressionWorker remain database-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..state import ArchivalMemory, EntityLedger, MemoryState


class BaseStorageAdapter(ABC):
    """
    Abstract interface for Sawtooth-Memory distributed storage backends.
    Requires asynchronous implementation to match the non-blocking engine.
    """

    @abstractmethod
    async def load_state(self, session_id: str) -> Optional[MemoryState]:
        """
        Fetch and hydrate the MemoryState for a given session.
        Should return None if the session does not exist.
        """
        pass

    @abstractmethod
    async def save_state(self, session_id: str, state: MemoryState) -> None:
        """
        Serialize and persist the complete MemoryState to the distributed backend.
        """
        pass

    @abstractmethod
    async def delete_state(self, session_id: str) -> None:
        """
        Wipes the session from the remote database (used for clear/reset ops).
        """
        pass

    @abstractmethod
    async def load_pool_state(
        self, pool_id: str
    ) -> Optional[tuple[EntityLedger, ArchivalMemory]]:
        """
        Fetch and hydrate shared (L1.5, L2) state for a multi-agent pool.
        Should return None if the pool does not exist.
        """
        pass

    @abstractmethod
    async def save_pool_state(
        self, pool_id: str, entities: EntityLedger, archive: ArchivalMemory
    ) -> None:
        """
        Persist shared (L1.5, L2) pool state to the distributed backend.
        """
        pass

    async def merge_pool_state(
        self,
        pool_id: str,
        merge_fn: Callable[[EntityLedger, ArchivalMemory], None],
    ) -> tuple[EntityLedger, ArchivalMemory]:
        """
        Atomically load pool state, apply ``merge_fn`` in place, and persist
        the result -- holding a lock across the full read-modify-write
        sequence so two agents syncing concurrently can't clobber each
        other's deltas. If no pool state exists yet, ``merge_fn`` runs
        against freshly created ``EntityLedger``/``ArchivalMemory`` instances.

        The default implementation below is a plain load-modify-save with no
        such lock; it exists only so adapters that implement nothing beyond
        ``load_pool_state``/``save_pool_state`` keep working. Backends that
        can hold a lock across the sequence (``PostgresStorageAdapter``,
        ``RedisStorageAdapter``) override this method to actually do so --
        any adapter meant to support true multi-agent concurrency should too.
        """
        pool_state = await self.load_pool_state(pool_id)
        if pool_state is None:
            entities, archive = EntityLedger(), ArchivalMemory()
        else:
            entities, archive = pool_state

        merge_fn(entities, archive)

        await self.save_pool_state(pool_id, entities, archive)
        return entities, archive
