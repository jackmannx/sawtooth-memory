"""
Sawtooth-Memory — Async context manager middleware for LLM agents.

Solves the "Lost in the Middle" problem by dynamically compressing
context windows via local Ollama models on a background asyncio thread.

Public API:
    ContextManager       — Main middleware class
    ContextManagerConfig — Configuration (token limits, Ollama settings)
    OllamaConfig         — Ollama-specific connection settings
    MemoryState          — The four-tier state object (read access)
    SawtoothError        — Base exception

Example:
    from sawtooth_memory import ContextManager, ContextManagerConfig

    config = ContextManagerConfig(soft_limit_tokens=3000)

    async with ContextManager("You are a helpful agent.", config) as cm:
        await cm.add_message("user", "What is 2 + 2?")
        messages = cm.build_prompt()
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

__all__ = [
    "ContextManager",
    "ContextManagerConfig",
    "OllamaConfig",
    "MemoryState",
    "SawtoothError",
    "CompressionError",
    "OllamaConnectionError",
    "MalformedOutputError",
    "TokenLimitExceededError",
]

__version__ = "0.2.0"
