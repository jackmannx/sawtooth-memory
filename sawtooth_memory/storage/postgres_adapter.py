"""
postgres_adapter.py — Durable PostgreSQL Storage Backend with pgvector.

Utilizes asyncpg to persist MemoryState as JSONB with row-level locking
for safe concurrent access across distributed stateless containers.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ..state import ArchivalMemory, EntityLedger, MemoryState
from .base import BaseStorageAdapter
from .semantic import SemanticChunkResult, SemanticStorageAdapter


class PostgresStorageAdapter(BaseStorageAdapter, SemanticStorageAdapter):
    """
    Async PostgreSQL implementation of the Sawtooth storage contract.

    Stores the full MemoryState tree in a JSONB column and maintains a
    separate pgvector table for semantic similarity queries.
    """

    def __init__(
        self,
        dsn: str,
        embedding_dimension: int = 1536,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        """
        Initialize the PostgreSQL adapter configuration.

        Args:
            dsn: PostgreSQL connection string (e.g. postgresql://user:pass@host/db).
            embedding_dimension: Vector column width matching the embeddings provider.
            min_pool_size: Minimum asyncpg connection pool size.
            max_pool_size: Maximum asyncpg connection pool size.
        """
        self.dsn = dsn
        self.embedding_dimension = embedding_dimension
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self._pool: Any = None
        self._schema_ready = False

    async def _ensure_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
            )
        if not self._schema_ready:
            await self._bootstrap_schema()
            self._schema_ready = True
        return self._pool

    async def _bootstrap_schema(self) -> None:
        """Ensure pgvector extension, tables, and HNSW index exist."""
        pool = self._pool
        assert pool is not None

        dim = self.embedding_dimension
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sawtooth_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    state_payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS sawtooth_semantic_vectors (
                    id BIGSERIAL PRIMARY KEY,
                    session_id VARCHAR(255) REFERENCES sawtooth_sessions(session_id)
                        ON DELETE CASCADE,
                    text_chunk TEXT NOT NULL,
                    embedding VECTOR({dim}),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS sawtooth_vector_idx
                ON sawtooth_semantic_vectors
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS sawtooth_vector_session_idx
                ON sawtooth_semantic_vectors (session_id)
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sawtooth_pools (
                    pool_id VARCHAR(255) PRIMARY KEY,
                    entities_payload JSONB NOT NULL,
                    archive_payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )

    @staticmethod
    def _hydrate_state(raw_payload: Any) -> MemoryState:
        if isinstance(raw_payload, str):
            return MemoryState.model_validate_json(raw_payload)
        if isinstance(raw_payload, (bytes, bytearray)):
            return MemoryState.model_validate_json(raw_payload)
        return MemoryState.model_validate(raw_payload)

    async def load_state(self, session_id: str) -> MemoryState | None:
        """Fetch the JSONB payload and hydrate the Pydantic state machine."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state_payload FROM sawtooth_sessions WHERE session_id = $1",
                session_id,
            )

        if row is None:
            return None

        return self._hydrate_state(row["state_payload"])

    async def save_state(self, session_id: str, state: MemoryState) -> None:
        """
        Persist state using row-level locking to prevent concurrent clobbering.

        Acquires a ``SELECT ... FOR UPDATE`` lock inside a transaction before
        performing an atomic upsert of the JSONB payload.
        """
        pool = await self._ensure_pool()
        json_payload = state.model_dump_json()

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    SELECT session_id FROM sawtooth_sessions
                    WHERE session_id = $1 FOR UPDATE
                    """,
                    session_id,
                )
                await conn.execute(
                    """
                    INSERT INTO sawtooth_sessions (session_id, state_payload, updated_at)
                    VALUES ($1, $2::jsonb, NOW())
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                        state_payload = EXCLUDED.state_payload,
                        updated_at = NOW()
                    """,
                    session_id,
                    json_payload,
                )

    async def delete_state(self, session_id: str) -> None:
        """Purge the session and cascade-delete associated vector chunks."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM sawtooth_sessions WHERE session_id = $1",
                session_id,
            )

    async def load_pool_state(
        self, pool_id: str
    ) -> tuple[EntityLedger, ArchivalMemory] | None:
        """Fetch shared L1.5 + L2 state for all agents in a pool."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT entities_payload, archive_payload
                FROM sawtooth_pools
                WHERE pool_id = $1
                """,
                pool_id,
            )
        if row is None:
            return None

        entities = EntityLedger.model_validate(row["entities_payload"])
        archive = ArchivalMemory.model_validate(row["archive_payload"])
        return entities, archive

    async def save_pool_state(
        self, pool_id: str, entities: EntityLedger, archive: ArchivalMemory
    ) -> None:
        """Persist shared L1.5 + L2 state for all agents in a pool."""
        pool = await self._ensure_pool()
        entities_payload = entities.model_dump(mode="json")
        archive_payload = archive.model_dump(mode="json")

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    SELECT pool_id FROM sawtooth_pools
                    WHERE pool_id = $1 FOR UPDATE
                    """,
                    pool_id,
                )
                await conn.execute(
                    """
                    INSERT INTO sawtooth_pools (
                        pool_id, entities_payload, archive_payload, updated_at
                    )
                    VALUES ($1, $2::jsonb, $3::jsonb, NOW())
                    ON CONFLICT (pool_id)
                    DO UPDATE SET
                        entities_payload = EXCLUDED.entities_payload,
                        archive_payload = EXCLUDED.archive_payload,
                        updated_at = NOW()
                    """,
                    pool_id,
                    entities_payload,
                    archive_payload,
                )

    async def upsert_vector_chunks_batch(
        self,
        session_id: str,
        chunks: Sequence[Tuple[str, List[float]]],
    ) -> int:
        """
        Batch-insert text chunks and embeddings in a single round-trip.

        Ensures the parent session row exists so the foreign-key constraint
        on ``sawtooth_semantic_vectors`` is satisfied.
        """
        if not chunks:
            return 0

        pool = await self._ensure_pool()
        texts = [text for text, _ in chunks]
        embeddings = [embedding for _, embedding in chunks]

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO sawtooth_sessions (session_id, state_payload, updated_at)
                    VALUES ($1, '{}'::jsonb, NOW())
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    session_id,
                )
                await conn.executemany(
                    """
                    INSERT INTO sawtooth_semantic_vectors
                        (session_id, text_chunk, embedding)
                    VALUES ($1, $2, $3)
                    """,
                    [(session_id, text, embedding) for text, embedding in zip(texts, embeddings)],
                )

        return len(chunks)

    async def upsert_vector_chunk(
        self, session_id: str, text: str, embedding: List[float]
    ) -> None:
        """Insert a single text chunk and its embedding into the semantic vector layer."""
        await self.upsert_vector_chunks_batch(session_id, [(text, embedding)])

    async def search_similar(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[SemanticChunkResult]:
        """Return the *top_k* chunks most similar to *query_embedding*."""
        if top_k < 1:
            return []

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    text_chunk,
                    1 - (embedding <=> $2) AS similarity
                FROM sawtooth_semantic_vectors
                WHERE session_id = $1
                ORDER BY embedding <=> $2
                LIMIT $3
                """,
                session_id,
                query_embedding,
                top_k,
            )

        return [
            SemanticChunkResult(text=row["text_chunk"], similarity=float(row["similarity"]))
            for row in rows
        ]

    async def count_vector_chunks(self, session_id: str) -> int:
        """Return the number of indexed vector chunks for *session_id*."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM sawtooth_semantic_vectors
                WHERE session_id = $1
                """,
                session_id,
            )
        return int(count or 0)

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._schema_ready = False
