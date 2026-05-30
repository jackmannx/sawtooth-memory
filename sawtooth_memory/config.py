"""
config.py — Configuration models for Sawtooth-Memory.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, SecretStr


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


class Provider(str, Enum):
    """Supported cloud LLM providers for the compression backend."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


# ---------------------------------------------------------------------------
# Backend configs
# ---------------------------------------------------------------------------


class OllamaConfig(BaseModel):
    """Connection settings for the local Ollama compression backend."""

    base_url: str = "http://localhost:11434"
    model: str = "phi4"
    timeout_seconds: int = 90


class CloudConfig(BaseModel):
    """
    Connection settings for a cloud LLM compression backend.

    Supports OpenAI, Anthropic, and Gemini via their respective APIs.
    Use ``base_url`` to route traffic through proxies like Helicone,
    LiteLLM, or Azure OpenAI without changing provider-specific payload
    construction.

    Example::

        from sawtooth_memory.config import CloudConfig, Provider

        cfg = CloudConfig(
            provider=Provider.ANTHROPIC,
            model="claude-3-5-haiku-latest",
            api_key="sk-ant-...",
        )
    """

    provider: Provider
    model: str
    api_key: SecretStr
    base_url: Optional[str] = None
    timeout_seconds: int = 60


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ContextManagerConfig(BaseModel):
    """Top-level configuration for the ContextManager middleware."""

    soft_limit_tokens: int = 3000
    hard_limit_tokens: int = 6000
    chunk_size: int = 10
    tokenizer_model: str = "gpt-4o"
    fallback_truncate: bool = True

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    cloud: Optional[CloudConfig] = None
