"""
embeddings/base.py — Embedding provider contract for L3 semantic storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    """Generates dense vectors for semantic archival indexing."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width produced by this provider."""
        ...

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed one or more texts in a single provider call.

        Returns:
            One embedding vector per input text, in the same order.
        """
        ...

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string."""
        vectors = await self.embed([text])
        return vectors[0]
