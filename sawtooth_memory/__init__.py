"""
Sawtooth-Memory — Context manager middleware for LLM agents.

Solves the "Lost in the Middle" problem by dynamically compressing
context windows via local or cloud models.

Public API:
    SyncContextManager   — Sync-native API for scripts and WSGI apps
    ContextManager       — Async middleware with background compression worker
    ContextManagerConfig — Configuration (token limits, backend settings)
    OllamaConfig         — Ollama-specific connection settings
    MemoryState          — The five-tier state object (L0–L3; read access)
    SawtoothError        — Base exception

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

from .config import ContextManagerConfig, OllamaConfig
from .exceptions import (
    CompressionError,
    MalformedOutputError,
    OllamaConnectionError,
    SawtoothError,
    TokenLimitExceededError,
)
from .middleware import ContextManager
from .state import MemoryState
from .sync_manager import SyncContextManager

__all__ = [
    "ContextManager",
    "SyncContextManager",
    "ContextManagerConfig",
    "OllamaConfig",
    "MemoryState",
    "SawtoothError",
    "CompressionError",
    "OllamaConnectionError",
    "MalformedOutputError",
    "TokenLimitExceededError",
]

__version__ = "0.2.2"
