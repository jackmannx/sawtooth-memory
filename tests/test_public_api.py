"""Public export surface and sync wrapper API parity."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import sawtooth_memory as sm
from sawtooth_memory import (
    CloudConfig,
    ContextManager,
    ContextManagerConfig,
    PostgresStorageAdapter,
    Provider,
    RedisStorageAdapter,
    SawtoothSyncWrapper,
    SyncContextManager,
    create_embedding_provider,
    get_event_bus,
)


def test_package_exports_core_surface():
    for name in (
        "ContextManager",
        "SyncContextManager",
        "SawtoothSyncWrapper",
        "ContextManagerConfig",
        "OllamaConfig",
        "CloudConfig",
        "Provider",
        "MemoryState",
        "SemanticChunkResult",
        "BaseStorageAdapter",
        "RedisStorageAdapter",
        "PostgresStorageAdapter",
        "EventBus",
        "get_event_bus",
        "L3VectorIndexedEvent",
        "DTEFoldCreatedEvent",
        "EmbeddingProvider",
        "create_embedding_provider",
        "SawtoothError",
    ):
        assert name in sm.__all__
        assert hasattr(sm, name)


def test_storage_and_embeddings_construct_without_live_backends():
    redis = RedisStorageAdapter(redis_url="redis://localhost:6379/15")
    assert redis.key_prefix.startswith("sawtooth:")

    postgres = PostgresStorageAdapter(
        dsn="postgresql://user:pass@localhost/sawtooth",
        embedding_dimension=64,
    )
    assert postgres.embedding_dimension == 64

    embedder = create_embedding_provider("hash", dimension=32)
    assert embedder.dimension == 32

    cfg = CloudConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        api_key="sk-test",
    )
    assert cfg.provider is Provider.OPENAI
    assert get_event_bus() is not None


def test_sync_wrapper_pin_entity_and_state(tmp_path: Path):
    config = ContextManagerConfig(soft_limit_tokens=1000)
    journal = tmp_path / "wrapper.jsonl"

    with patch(
        "sawtooth_memory.middleware.OllamaCompressor.ping",
        new_callable=AsyncMock,
    ):
        with SawtoothSyncWrapper(
            "Parity test",
            config=config,
            enable_events=True,
            journal_path=journal,
        ) as memory:
            memory.add_message("user", "Track ALPHA-991 please.")
            memory.pin_entity("tracking_code", "ALPHA-991")

            assert memory.state.l1_5_entities.get_latest("tracking_code") == "ALPHA-991"
            assert memory.retrieve_observation("missing") is None

            prompt = memory.build_prompt()
            assert prompt[0]["role"] == "system"
            assert any("ALPHA-991" in msg["content"] for msg in prompt)


def test_managers_are_distinct_types():
    assert ContextManager is not SyncContextManager
    assert SyncContextManager is not SawtoothSyncWrapper
