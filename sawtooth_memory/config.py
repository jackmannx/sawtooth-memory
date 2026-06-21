"""
config.py — Configuration models for Sawtooth-Memory.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, SecretStr, model_validator

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
    model: str = "phi4-mini:latest"
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
    soft_limit_tokens: int = Field(
        default=1000,
        description="Trigger compression when L1 tokens exceed this soft limit.",
    )
    hard_limit_tokens: int = Field(
        default=2500,
        description="Failsafe limit to hard-truncate older L1 messages if compression is too slow.",
    )
    chunk_size: int = Field(
        default=4,
        description="Number of oldest L1 messages to summarize and evict per compression cycle.",
    )
    fallback_truncate: bool = Field(
        default=True,
        description="Whether to aggressively discard messages if hard_limit is reached.",
    )
    max_unsummarized_turns: Optional[int] = Field(
        default=None,
        description="Trigger compression if unsummarized L1 messages reach this count.",
    )

    # NEW: Highly descriptive architectural configuration
    background_model: Optional[str] = Field(
        default=None,
        description="Define a cheaper, faster model (e.g., 'gpt-4o-mini', 'gemini-1.5-flash') purely for background tasks.",
    )

    tokenizer_model: str = Field(
        default="gpt-4o",
        description="Tokenizer encoding to use for precise context monitoring.",
    )
    journal_path: str = Field(
        default=".sawtooth_journal.jsonl",
        description="Path to the JSONL auditing journal.",
    )
    enable_deterministic_ner: bool = Field(
        default=True,
        description="Enable the fast regex/deterministic NER extraction wave inside the compression worker.",
    )
    custom_ner_patterns: dict[str, str] = Field(
        default_factory=dict,
        description="User-defined key-to-regex-string mappings that extend or override default tracking.",
    )
    storage_adapter: Optional[Any] = Field(
        default=None,
        description=(
            "Pass an instance of a BaseStorageAdapter "
            "(e.g. RedisStorageAdapter or PostgresStorageAdapter) for distributed state."
        ),
    )
    session_id: str = Field(
        default="local_default",
        description="Unique identifier for the user session when using distributed storage.",
    )
    pool_id: Optional[str] = Field(
        default=None,
        description="Namespace mapping for multi-agent pool synchronization.",
    )

    ollama: Optional[OllamaConfig] = None
    cloud: Optional[CloudConfig] = None

    @model_validator(mode="after")
    def __v2_normalize_dual_model_architecture__(self) -> "ContextManagerConfig":
        """
        Architecture Enforcement: Automatically resolves the background model
        hierarchical override seamlessly during object validation.
        """
        if not self.background_model:
            return self

        # If a cloud provider setup is present, cascade the override down
        if self.cloud:
            self.cloud.model = self.background_model

        # If an Ollama configuration block is active, cascade down
        if self.ollama:
            self.ollama.model = self.background_model
        # If neither is defined, initialize a default Ollama config with the target background model
        elif not self.cloud:
            self.ollama = OllamaConfig(model=self.background_model)

        return self
