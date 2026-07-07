"""
embeddings/factory.py — Construct embedding providers from configuration.
"""

from __future__ import annotations

from typing import Literal, Optional

from ..config import Provider, resolve_cloud_api_key
from .base import EmbeddingProvider
from .hash import HashEmbeddingProvider
from .openai import OpenAIEmbeddingProvider

EmbeddingBackend = Literal["hash", "openai"]


def create_embedding_provider(
    backend: EmbeddingBackend = "hash",
    *,
    model: str = "text-embedding-3-small",
    dimension: int = 1536,
    api_key: Optional[str] = None,
) -> EmbeddingProvider:
    """
    Instantiate an :class:`EmbeddingProvider` for L3 semantic storage.

    Args:
        backend: ``"hash"`` for deterministic local vectors (tests/dev),
            ``"openai"`` for production-quality embeddings.
        model: OpenAI embedding model name when ``backend="openai"``.
        dimension: Vector width (used by hash backend; validated for OpenAI).
        api_key: Optional explicit API key; falls back to ``OPENAI_API_KEY``.
    """
    if backend == "hash":
        return HashEmbeddingProvider(dimension=dimension)

    if backend == "openai":
        resolved_key = api_key or resolve_cloud_api_key(Provider.OPENAI)
        if not resolved_key:
            raise ValueError(
                "OpenAI embedding backend requires OPENAI_API_KEY or an explicit api_key."
            )
        return OpenAIEmbeddingProvider(model=model, api_key=resolved_key, dimension=dimension)

    raise ValueError(f"Unsupported embedding backend: {backend!r}")
