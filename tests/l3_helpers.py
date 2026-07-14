"""Shared helpers for L3 semantic storage tests."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from sawtooth_memory.config import ContextManagerConfig, OllamaConfig
from sawtooth_memory.state import ArchivalMemory, EntityLedger, MemoryState
from sawtooth_memory.storage.base import BaseStorageAdapter
from sawtooth_memory.storage.semantic import SemanticChunkResult, SemanticStorageAdapter


class InMemorySemanticStorage(BaseStorageAdapter, SemanticStorageAdapter):
    """Minimal in-memory adapter for L3 integration tests."""

    def __init__(self, embedding_dimension: int = 64) -> None:
        self.embedding_dimension = embedding_dimension
        self.sessions: dict[str, MemoryState] = {}
        self.vectors: list[tuple[str, str, list[float]]] = []

    async def load_state(self, session_id: str) -> Optional[MemoryState]:
        return self.sessions.get(session_id)

    async def save_state(self, session_id: str, state: MemoryState) -> None:
        self.sessions[session_id] = state.model_copy(deep=True)

    async def delete_state(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.vectors = [v for v in self.vectors if v[0] != session_id]

    async def load_pool_state(
        self, pool_id: str
    ) -> Optional[tuple[EntityLedger, ArchivalMemory]]:
        return None

    async def save_pool_state(
        self, pool_id: str, entities: EntityLedger, archive: ArchivalMemory
    ) -> None:
        pass

    async def upsert_vector_chunks_batch(
        self,
        session_id: str,
        chunks: Sequence[Tuple[str, list[float]]],
    ) -> int:
        for text, embedding in chunks:
            self.vectors.append((session_id, text, embedding))
        return len(chunks)

    async def search_similar(
        self,
        session_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> List[SemanticChunkResult]:
        session_vectors = [v for v in self.vectors if v[0] == session_id]
        return [
            SemanticChunkResult(text=text, similarity=0.9)
            for _, text, _ in session_vectors[:top_k]
        ]

    async def count_vector_chunks(self, session_id: str) -> int:
        return sum(1 for sid, _, _ in self.vectors if sid == session_id)


def make_l3_config(storage: InMemorySemanticStorage, **overrides) -> ContextManagerConfig:
    defaults = dict(
        soft_limit_tokens=50,
        hard_limit_tokens=500,
        chunk_size=2,
        compression_mode="always_llm",
        storage_adapter=storage,
        session_id="test-session",
        enable_l3_semantic_storage=True,
        embedding_backend="hash",
        embedding_dimension=storage.embedding_dimension,
        l3_chunk_max_chars=500,
        ollama=OllamaConfig(base_url="http://localhost:11434", model="phi4"),
    )
    defaults.update(overrides)
    return ContextManagerConfig(**defaults)
