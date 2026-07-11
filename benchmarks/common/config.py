"""Benchmark-specific ContextManager configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from sawtooth_memory.config import ContextManagerConfig, OllamaConfig
from sawtooth_memory.middleware import ContextManager

from .mock_compressor import MockCompressor


def benchmark_config(
    *,
    soft_limit_tokens: int = 250,
    hard_limit_tokens: int = 600,
    chunk_size: int = 4,
    max_unsummarized_turns: int | None = None,
    enable_deterministic_ner: bool = True,
    enable_events: bool = False,
    journal_path: str | Path | None = None,
    storage_adapter: Any = None,
    **overrides: Any,
) -> ContextManagerConfig:
    """Return a ContextManagerConfig tuned for benchmark scenarios."""
    kwargs: dict[str, Any] = {
        "soft_limit_tokens": soft_limit_tokens,
        "hard_limit_tokens": hard_limit_tokens,
        "chunk_size": chunk_size,
        "max_unsummarized_turns": max_unsummarized_turns,
        "enable_deterministic_ner": enable_deterministic_ner,
        "fallback_truncate": True,
        "tokenizer_model": "gpt-4o",
        "ollama": OllamaConfig(base_url="http://localhost:11434", model="phi4-mini"),
    }
    if journal_path is not None:
        kwargs["journal_path"] = str(journal_path)
    if storage_adapter is not None:
        kwargs["storage_adapter"] = storage_adapter
    kwargs.update(overrides)
    return ContextManagerConfig(**kwargs)


class MockedContextManager:
    """Context manager that patches OllamaCompressor with MockCompressor."""

    def __init__(
        self,
        system_prompt: str,
        config: ContextManagerConfig,
        *,
        mock_delay_ms: float = 1.0,
        enable_events: bool = False,
    ) -> None:
        self.system_prompt = system_prompt
        self.config = config
        self.mock_delay_ms = mock_delay_ms
        self.enable_events = enable_events
        self.mock = MockCompressor(delay_ms=mock_delay_ms)
        self._patch = patch(
            "sawtooth_memory.middleware.OllamaCompressor",
            return_value=self.mock,
        )
        self._cm: ContextManager | None = None

    async def __aenter__(self) -> ContextManager:
        self._patch.start()
        self._cm = ContextManager(
            self.system_prompt,
            self.config,
            enable_events=self.enable_events,
        )
        await self._cm.start()
        return self._cm

    async def __aexit__(self, *args: object) -> None:
        if self._cm is not None:
            await self._cm.stop()
        self._patch.stop()
