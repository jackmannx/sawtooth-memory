"""
l3_indexer.py — L3 semantic vector archival indexing service.

Chunks evicted L1 text, batch-embeds via an :class:`EmbeddingProvider`,
and persists vectors through a :class:`SemanticStorageAdapter`.

Retrieval is exposed via :meth:`SemanticIndexer.search` but is **not**
wired into ``ContextManager.build_prompt()`` in this release.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Sequence

from .embeddings.base import EmbeddingProvider
from .state import MemoryState, SemanticVectorMemory
from .storage.semantic import SemanticChunkResult, SemanticStorageAdapter

logger = logging.getLogger(__name__)

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split *text* into retrieval-sized chunks without breaking mid-word
    when possible.

    Strategy:
      1. Split on paragraph boundaries (``\\n\\n``).
      2. Merge paragraphs until ``max_chars`` is reached.
      3. Hard-split oversized paragraphs on whitespace boundaries.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(stripped) if p.strip()]
    if not paragraphs:
        paragraphs = [stripped]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(paragraph, max_chars))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split *text* into segments of at most *max_chars* on word boundaries."""
    parts: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        segment = text[start:end].strip()
        if segment:
            parts.append(segment)
        start = end if end > start else end + 1

    return parts


class SemanticIndexer:
    """
    Indexes compressed L1 text into L3 semantic vector storage.

    Designed for batch efficiency: one embed call and one DB round-trip
    per compression cycle.
    """

    def __init__(
        self,
        storage: SemanticStorageAdapter,
        embedder: EmbeddingProvider,
        *,
        chunk_max_chars: int = 2000,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._chunk_max_chars = chunk_max_chars

    async def index(
        self,
        session_id: str,
        text: str,
        state: MemoryState,
    ) -> int:
        """
        Chunk, embed, and persist *text* for *session_id*.

        Updates ``state.l3_semantic`` metadata in place.

        Returns:
            Number of new chunks indexed (0 when *text* is empty).
        """
        chunks = chunk_text(text, self._chunk_max_chars)
        if not chunks:
            return 0

        embeddings = await self._embedder.embed(chunks)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding provider returned {len(embeddings)} vectors "
                f"for {len(chunks)} chunks"
            )

        payload = list(zip(chunks, embeddings))
        inserted = await self._storage.upsert_vector_chunks_batch(session_id, payload)

        state.l3_semantic.chunk_count += inserted
        state.l3_semantic.last_indexed_at = datetime.now(timezone.utc)

        logger.debug(
            "SemanticIndexer: indexed %d chunk(s) for session %s.",
            inserted,
            session_id,
        )
        return inserted

    async def search(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[SemanticChunkResult]:
        """Embed *query* and return the most similar stored chunks."""
        query_embedding = await self._embedder.embed_one(query)
        return await self._storage.search_similar(session_id, query_embedding, top_k)

    async def count(self, session_id: str) -> int:
        """Return the number of indexed chunks for *session_id*."""
        return await self._storage.count_vector_chunks(session_id)
