"""conftest.py — Shared pytest configuration."""


from sawtooth_memory.config import ContextManagerConfig, OllamaConfig
from sawtooth_memory.events.bus import reset_event_bus
from tests.l3_helpers import InMemorySemanticStorage, make_l3_config


@pytest.fixture
def ollama_config() -> OllamaConfig:
    return OllamaConfig(base_url="http://localhost:11434", model="phi4")


@pytest.fixture
def config(ollama_config: OllamaConfig) -> ContextManagerConfig:
    """Default ContextManagerConfig for integration tests."""
    return ContextManagerConfig(
        soft_limit_tokens=50,
        hard_limit_tokens=200,
        chunk_size=3,
        ollama=ollama_config,
    )


@pytest.fixture
def in_memory_semantic_storage() -> InMemorySemanticStorage:
    return InMemorySemanticStorage(embedding_dimension=64)


@pytest.fixture
def l3_config(in_memory_semantic_storage: InMemorySemanticStorage) -> ContextManagerConfig:
    return make_l3_config(in_memory_semantic_storage)


@pytest.fixture(autouse=True)
def isolated_event_bus():
    """Reset the global EventBus singleton between tests."""
    reset_event_bus()
    yield
    reset_event_bus()
