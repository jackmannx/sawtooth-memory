"""
embeddings/hash.py — Deterministic local embedding provider.

Produces fixed-width unit-normalised vectors from text hashes.
No network calls — ideal for tests and offline development.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from .base import EmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Fast, deterministic embeddings derived from SHA-256 digests.

    Each dimension is seeded independently so similar texts are not
    artificially correlated; this provider is for storage-layer testing,
    not production semantic quality.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_to_vector(text, self._dimension) for text in texts]


def _hash_to_vector(text: str, dimension: int) -> list[float]:
    """Map *text* to a unit-normalised vector of length *dimension*."""
    values = [0.0] * dimension
    digest_size = 32
    blocks_needed = (dimension + digest_size - 1) // digest_size

    for block_idx in range(blocks_needed):
        digest = hashlib.sha256(f"{text}:{block_idx}".encode()).digest()
        for byte_idx, raw_byte in enumerate(digest):
            dim_idx = block_idx * digest_size + byte_idx
            if dim_idx >= dimension:
                break
            # Map byte to [-1, 1] for a centred distribution.
            values[dim_idx] = (raw_byte / 127.5) - 1.0

    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return values
    return [v / norm for v in values]
