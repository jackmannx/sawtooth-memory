"""
base.py — Distributed Storage Architecture Contract.

Defines the Abstract Base Class for all remote state persistence.
Ensures ContextManager and CompressionWorker remain database-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Optional

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
