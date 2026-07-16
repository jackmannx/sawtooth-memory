"""Public export surface, version alignment, and sync wrapper API parity."""

import inspect
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SYNC_PUBLIC_METHODS = (
    "add_message",
    "retrieve_observation",
    "pin_entity",
    "build_prompt",
    "explain_prompt",
    "search_semantic_archive",
    "l3_chunk_count",
    "get_stats",
    "health_check",
)

_SYNC_PUBLIC_PROPERTIES = ("state",)


def _pyproject_version() -> str:
    data = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_version_matches_pyproject():
    assert sm.__version__ == _pyproject_version()


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


def test_all_exports_are_importable():
    for name in sm.__all__:
        obj = getattr(sm, name)
        assert obj is not None, f"sawtooth_memory.{name} resolved to None"


def test_sync_manager_public_api_parity():
    for name in _SYNC_PUBLIC_METHODS:
        assert hasattr(SyncContextManager, name), f"SyncContextManager missing {name}"
        assert hasattr(SawtoothSyncWrapper, name), f"SawtoothSyncWrapper missing {name}"
        assert hasattr(ContextManager, name), f"ContextManager missing {name}"

    for name in _SYNC_PUBLIC_PROPERTIES:
        assert isinstance(
            inspect.getattr_static(SyncContextManager, name), property
        )
        assert isinstance(
            inspect.getattr_static(SawtoothSyncWrapper, name), property
        )
        assert isinstance(inspect.getattr_static(ContextManager, name), property)


def test_config_default_compression_mode_is_dte():
    config = ContextManagerConfig()
    assert config.compression_mode == "dte"


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


@pytest.fixture
def _ping_patch():
    with patch(
        "sawtooth_memory.middleware.OllamaCompressor.ping",
        new_callable=AsyncMock,
    ):
        yield


def test_sync_wrapper_pin_entity_and_state(tmp_path: Path, _ping_patch):
    config = ContextManagerConfig(soft_limit_tokens=1000)
    journal = tmp_path / "wrapper.jsonl"

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


def test_sync_manager_pin_entity_and_state(tmp_path: Path):
    config = ContextManagerConfig.for_sync_script(soft_limit_tokens=1000)

    with SyncContextManager("Parity test", config=config) as memory:
        memory.add_message("user", "Track INC-4421 please.")
        memory.pin_entity("ticket_id", "INC-4421")

        assert memory.state.l1_5_entities.get_latest("ticket_id") == "INC-4421"
        prompt = memory.build_prompt()
        assert any("INC-4421" in msg["content"] for msg in prompt)


def test_managers_are_distinct_types():
    assert ContextManager is not SyncContextManager
    assert SyncContextManager is not SawtoothSyncWrapper
