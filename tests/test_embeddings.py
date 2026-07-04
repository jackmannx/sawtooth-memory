"""Tests for embedding providers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sawtooth_memory.embeddings.factory import create_embedding_provider
from sawtooth_memory.embeddings.hash import HashEmbeddingProvider


class TestHashEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_deterministic_output(self):
        provider = HashEmbeddingProvider(dimension=128)
        a = await provider.embed(["hello"])
        b = await provider.embed(["hello"])
        assert a == b

    @pytest.mark.asyncio
    async def test_different_texts_differ(self):
        provider = HashEmbeddingProvider(dimension=128)
        vectors = await provider.embed(["alpha", "beta"])
        assert vectors[0] != vectors[1]

    @pytest.mark.asyncio
    async def test_unit_normalised(self):
        provider = HashEmbeddingProvider(dimension=64)
        vector = (await provider.embed(["test"]))[0]
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_batch_embed(self):
        provider = HashEmbeddingProvider(dimension=32)
        vectors = await provider.embed(["a", "b", "c"])
        assert len(vectors) == 3
        assert all(len(v) == 32 for v in vectors)

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            HashEmbeddingProvider(dimension=0)


class TestEmbeddingFactory:
    def test_hash_backend(self):
        provider = create_embedding_provider("hash", dimension=256)
        assert isinstance(provider, HashEmbeddingProvider)
        assert provider.dimension == 256

    def test_openai_backend_requires_key(self):
        with pytest.raises(ValueError):
            create_embedding_provider("openai")


@pytest.mark.asyncio
async def test_openai_embedding_provider_batch():
    from sawtooth_memory.embeddings.openai import OpenAIEmbeddingProvider

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]
    }

    provider = OpenAIEmbeddingProvider(
        model="text-embedding-3-small", api_key="sk-test", dimension=2
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    provider._client = mock_client

    vectors = await provider.embed(["one", "two"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    mock_client.post.assert_awaited_once()
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["input"] == ["one", "two"]
