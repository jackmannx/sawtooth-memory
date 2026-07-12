"""
config.py — Configuration models for Sawtooth-Memory.
"""

from __future__ import annotations

import os
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


# Cloud model name prefixes used by ``background_model`` auto-routing.
_CLOUD_MODEL_PREFIXES: dict[Provider, tuple[str, ...]] = {
    Provider.OPENAI: ("gpt-", "o1", "o3", "chatgpt-"),
    Provider.ANTHROPIC: ("claude-",),
    Provider.GEMINI: ("gemini-",),
}

_PROVIDER_ENV_KEYS: dict[Provider, tuple[str, ...]] = {
    Provider.OPENAI: ("OPENAI_API_KEY",),
    Provider.ANTHROPIC: ("ANTHROPIC_API_KEY",),
    Provider.GEMINI: ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def infer_cloud_provider(model: str) -> Provider | None:
    """Return the cloud provider for *model*, or ``None`` for local Ollama models."""
    normalized = model.lower()
    for provider, prefixes in _CLOUD_MODEL_PREFIXES.items():
        if any(normalized.startswith(prefix) for prefix in prefixes):
            return provider
    return None


def resolve_cloud_api_key(provider: Provider) -> str | None:
    """Read the API key for *provider* from standard environment variables."""
    for env_var in _PROVIDER_ENV_KEYS[provider]:
        value = os.environ.get(env_var)
        if value:
            return value
    return None


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
    enable_salience_extractor: bool = Field(
        default=True,
        description="Enable the local salience heuristic extractor for unstructured identifiers.",
    )
    salience_threshold: float = Field(
        default=0.5,
        description="Minimum salience score (0–1) for heuristic entity promotion to L1.5.",
    )
    salience_max_entities: int = Field(
        default=20,
        description="Maximum heuristic entities extracted per compression or ingest scan.",
    )
    enable_ingest_entity_scan: bool = Field(
        default=True,
        description="Scan incoming L1 messages for critical entities at ingest time.",
    )
    enable_entity_verifier: bool = Field(
        default=True,
        description="Re-inject protected entities dropped by the compression LLM.",
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

    # L3 semantic vector archival (storage layer; retrieval not in build_prompt)
    enable_l3_semantic_storage: bool = Field(
        default=False,
        description=(
            "Index evicted L1 text into pgvector-backed L3 semantic storage "
            "during background compression. Requires a SemanticStorageAdapter "
            "(e.g. PostgresStorageAdapter)."
        ),
    )
    enable_l3_prompt_retrieval: bool = Field(
        default=True,
        description="Automatically retrieve and inject L3 chunks into build_prompt(). Ignored if L3 storage is disabled.",
    )
    l3_retrieval_top_k: int = Field(
        default=3,
        description="Maximum number of L3 semantic chunks to retrieve during build_prompt().",
    )
    l3_retrieval_max_tokens: int = Field(
        default=500,
        description="Token budget for the L3 retrieval block in build_prompt().",
    )
    embedding_backend: str = Field(
        default="hash",
        description='Embedding provider for L3 indexing: "hash" (local/tests) or "openai".',
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model when embedding_backend='openai'.",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Vector width; must match PostgresStorageAdapter.embedding_dimension.",
    )
    l3_chunk_max_chars: int = Field(
        default=2000,
        description="Maximum characters per L3 semantic chunk before splitting.",
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
        elif not self.cloud:
            provider = infer_cloud_provider(self.background_model)
            if provider is not None:
                api_key = resolve_cloud_api_key(provider)
                if not api_key:
                    env_vars = " or ".join(_PROVIDER_ENV_KEYS[provider])
                    raise ValueError(
                        f"background_model={self.background_model!r} requires a "
                        f"{provider.value} API key. Set {env_vars} in your environment, "
                        "or pass an explicit cloud=CloudConfig(...)."
                    )
                self.cloud = CloudConfig(
                    provider=provider,
                    model=self.background_model,
                    api_key=api_key,
                )
            else:
                self.ollama = OllamaConfig(model=self.background_model)

        return self

    @model_validator(mode="after")
    def __validate_l3_semantic_storage__(self) -> "ContextManagerConfig":
        """Ensure L3 is only enabled with a compatible semantic storage backend."""
        if not self.enable_l3_semantic_storage:
            return self

        if self.storage_adapter is None:
            raise ValueError(
                "enable_l3_semantic_storage=True requires storage_adapter "
                "(PostgresStorageAdapter with pgvector)."
            )

        from .storage.semantic import supports_semantic_storage

        if not supports_semantic_storage(self.storage_adapter):
            raise ValueError(
                "enable_l3_semantic_storage=True requires a SemanticStorageAdapter "
                "(e.g. PostgresStorageAdapter). RedisStorageAdapter does not support L3."
            )

        adapter_dim = getattr(self.storage_adapter, "embedding_dimension", None)
        if adapter_dim is not None and adapter_dim != self.embedding_dimension:
            raise ValueError(
                f"embedding_dimension ({self.embedding_dimension}) must match "
                f"storage_adapter.embedding_dimension ({adapter_dim})."
            )

        if self.embedding_backend not in ("hash", "openai"):
            raise ValueError(
                f"embedding_backend must be 'hash' or 'openai', got {self.embedding_backend!r}."
            )

        if self.l3_chunk_max_chars < 1:
            raise ValueError("l3_chunk_max_chars must be positive.")

        if self.enable_l3_prompt_retrieval and not self.enable_l3_semantic_storage:
            raise ValueError(
                "enable_l3_prompt_retrieval=True requires enable_l3_semantic_storage=True."
            )

        if self.l3_retrieval_top_k < 1:
            raise ValueError("l3_retrieval_top_k must be positive.")

        if self.l3_retrieval_max_tokens < 1:
            raise ValueError("l3_retrieval_max_tokens must be positive.")

        if not 0.0 <= self.salience_threshold <= 1.0:
            raise ValueError("salience_threshold must be between 0.0 and 1.0.")

        if self.salience_max_entities < 1:
            raise ValueError("salience_max_entities must be positive.")

        return self
