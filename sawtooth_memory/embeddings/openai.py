"""
embeddings/openai.py — OpenAI embeddings API provider.
"""

from __future__ import annotations

from typing import Sequence

import httpx

from .base import EmbeddingProvider

_OPENAI_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Batch embedding via the OpenAI ``/v1/embeddings`` endpoint.

    Accepts a list of texts per request to minimise HTTP round-trips.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 60,
        *,
        dimension: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIEmbeddingProvider requires an api_key")
        self._model = model
        self._api_key = api_key
        self._endpoint = base_url.rstrip("/") + "/embeddings"
        self._timeout = timeout_seconds
        self._dimension = dimension or _OPENAI_EMBEDDING_DIMS.get(model, 1536)
        self._client: httpx.AsyncClient | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        client = await self._get_client()
        response = await client.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": list(texts)},
        )
        response.raise_for_status()
        payload = response.json()

        # OpenAI returns embeddings out of order via the ``index`` field.
        ordered: list[list[float] | None] = [None] * len(texts)
        for item in payload.get("data", []):
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(texts):
                ordered[idx] = item.get("embedding", [])
        return [vec if vec is not None else [] for vec in ordered]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
