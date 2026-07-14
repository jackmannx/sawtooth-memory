"""
Sawtooth-Memory — Hierarchical context middleware for LLM agents.

Solves the "Lost in the Middle" problem by dynamically compressing
context windows via local or cloud models, with an exact entity ledger.

Public API (stable surface):

Managers
    SyncContextManager    — Sync-native API; inline blocking compression
    SawtoothSyncWrapper   — Sync façade over async ContextManager (AnyIO portal)
    ContextManager        — Async middleware with background compression worker

Config
    ContextManagerConfig  — Token limits, DTE, backends, storage, Entity Guard
    OllamaConfig          — Local Ollama compression settings
    CloudConfig           — Cloud compression settings
    Provider              — OpenAI / Anthropic / Gemini enum

State & results
    MemoryState           — L0–L3 state tree (read access via ``.state``)
    SemanticChunkResult   — L3 similarity search hit

Storage
    BaseStorageAdapter      — Persistence contract
    RedisStorageAdapter     — Redis session/pool store
    PostgresStorageAdapter  — Postgres + pgvector L3 store

Events
    get_event_bus, EventBus, and typed event dataclasses

Embeddings
    create_embedding_provider, EmbeddingProvider, Hash/OpenAI providers

Exceptions
    SawtoothError and typed subclasses

Integrations (optional extras — import from submodules):
    sawtooth_memory.integrations.langgraph
    sawtooth_memory.integrations.langchain_adapter

Example (sync script):
    from sawtooth_memory import SyncContextManager, ContextManagerConfig

    config = ContextManagerConfig.for_sync_script(soft_limit_tokens=1500)

    with SyncContextManager("You are a helpful agent.", config=config) as memory:
        memory.add_message("user", "What is 2 + 2?")
        messages = memory.build_prompt()

Example (async app):
    from sawtooth_memory import ContextManager, ContextManagerConfig

    config = ContextManagerConfig(soft_limit_tokens=3000)

    async with ContextManager("You are a helpful agent.", config) as cm:
        await cm.add_message("user", "What is 2 + 2?")
        messages = await cm.build_prompt()
"""

from .config import CloudConfig, ContextManagerConfig, OllamaConfig, Provider
from .embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from .events import (
    CompressionCycleCompleteEvent,
    CompressionCycleFailedEvent,
    CompressionCycleStartEvent,
    DTEFoldCreatedEvent,
    EntityAnchoredEvent,
    EventBus,
    HardLimitReachedEvent,
    L1EvictionEvent,
    L2SummaryGeneratedEvent,
    L3VectorIndexedEvent,
    SawtoothEvent,
    SoftLimitReachedEvent,
    get_event_bus,
    make_journal_handler,
    reset_event_bus,
)
from .exceptions import (
    CompressionError,
    MalformedOutputError,
    OllamaConnectionError,
    SawtoothError,
    TokenLimitExceededError,
)
from .middleware import ContextManager
from .state import MemoryState
from .storage import (
    BaseStorageAdapter,
    PostgresStorageAdapter,
    RedisStorageAdapter,
    SemanticChunkResult,
)
from .sync_manager import SyncContextManager
from .sync_wrapper import SawtoothSyncWrapper

__all__ = [
    # Managers
    "ContextManager",
    "SyncContextManager",
    "SawtoothSyncWrapper",
    # Config
    "ContextManagerConfig",
    "OllamaConfig",
    "CloudConfig",
    "Provider",
    # State
    "MemoryState",
    "SemanticChunkResult",
    # Storage
    "BaseStorageAdapter",
    "RedisStorageAdapter",
    "PostgresStorageAdapter",
    # Events
    "EventBus",
    "get_event_bus",
    "reset_event_bus",
    "make_journal_handler",
    "SawtoothEvent",
    "L1EvictionEvent",
    "EntityAnchoredEvent",
    "L2SummaryGeneratedEvent",
    "CompressionCycleStartEvent",
    "CompressionCycleCompleteEvent",
    "CompressionCycleFailedEvent",
    "SoftLimitReachedEvent",
    "HardLimitReachedEvent",
    "DTEFoldCreatedEvent",
    "L3VectorIndexedEvent",
    # Embeddings
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "create_embedding_provider",
    # Exceptions
    "SawtoothError",
    "CompressionError",
    "OllamaConnectionError",
    "MalformedOutputError",
    "TokenLimitExceededError",
]

__version__ = "0.2.2"
