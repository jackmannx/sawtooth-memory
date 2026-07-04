"""
semantic.py — L3 semantic vector storage contract.

Optional mixin for storage adapters that persist pgvector-backed
archival chunks. Retrieval is exposed via adapter methods but is
not wired into ``build_prompt()`` until a future release.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class SemanticChunkResult:
    """A text chunk returned from vector similarity search."""

    text: str
    similarity: float


class SemanticStorageAdapter(ABC):
    """
    Optional interface for L3 semantic vector persistence.

    Implementations must support efficient batch writes; single-chunk
    upserts may delegate to the batch path.
    """

    @abstractmethod
    async def upsert_vector_chunks_batch(
        self,
        session_id: str,
        chunks: Sequence[Tuple[str, List[float]]],
    ) -> int:
        """
        Insert multiple text chunks and embeddings in one round-trip.

        Returns:
            Number of rows inserted.
        """
        ...

    @abstractmethod
    async def search_similar(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[SemanticChunkResult]:
        """Return the *top_k* most similar chunks for *session_id*."""
        ...

    @abstractmethod
    async def count_vector_chunks(self, session_id: str) -> int:
        """Return the number of indexed vector chunks for *session_id*."""
        ...

    async def upsert_vector_chunk(
        self, session_id: str, text: str, embedding: List[float]
    ) -> None:
        """Convenience wrapper around :meth:`upsert_vector_chunks_batch`."""
        await self.upsert_vector_chunks_batch(session_id, [(text, embedding)])


def supports_semantic_storage(adapter: object) -> bool:
    """Return True when *adapter* implements the L3 semantic contract."""
    return isinstance(adapter, SemanticStorageAdapter)
