"""
middleware.py — ContextManager: four-tier memory with async compression.

Usage:
    from sawtooth_memory import ContextManager, ContextManagerConfig

    config = ContextManagerConfig(soft_limit_tokens=3000)

    async with ContextManager("You are a data analysis agent.", config) as cm:
        await cm.add_message("user", "Analyse Q3 revenue.")
        await cm.add_message("assistant", "Connecting to the database...")
        messages = cm.build_prompt()
"""

from __future__ import annotations

import logging

from .config import ContextManagerConfig
from .compressor import OllamaCompressor
from .monitor import TokenMonitor
from .state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    Message,
    MessageRole,
    SystemPrompt,
    WorkingMemory,
)
from .worker import CompressionTask, CompressionWorker

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Middleware that sits between your agent loop and the LLM API.

    Maintains four-tier memory (L0/L1/L1.5/L2). When L1 exceeds the soft
    limit, the oldest chunk is sliced off and queued for async compression
    via a background Ollama worker without blocking the main thread.
    """

    def __init__(
        self,
        system_prompt: str,
        config: ContextManagerConfig | None = None,
    ) -> None:
        self._config = config or ContextManagerConfig()

        self._monitor = TokenMonitor(
            model=self._config.tokenizer_model,
            soft_limit=self._config.soft_limit_tokens,
            hard_limit=self._config.hard_limit_tokens,
        )

        sp_tokens = self._monitor.count_text(system_prompt)
        self._state = MemoryState(
            l0_system=SystemPrompt(content=system_prompt, token_count=sp_tokens),
            l1_working=WorkingMemory(),
            l1_5_entities=EntityLedger(),
            l2_archival=ArchivalMemory(),
        )

        self._compressor = OllamaCompressor(self._config.ollama)
        self._worker = CompressionWorker(
            compressor=self._compressor,
            fallback_truncate=self._config.fallback_truncate,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._worker.start()

    async def stop(self) -> None:
        await self._worker.stop()

    async def __aenter__(self) -> "ContextManager":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def add_message(self, role: MessageRole, content: str) -> None:
        """
        Add a message to Working Memory (L1).

        If the soft token limit is crossed after adding this message,
        the oldest chunk_size messages are sliced off and enqueued for
        background compression without blocking this call.
        """
        msg = Message(role=role, content=content)
        msg.token_count = self._monitor.count_message(msg)
        self._state.l1_working.append(msg)

        if self._monitor.exceeds_soft_limit(self._state):
            self._trigger_compression()

    def build_prompt(self) -> list[dict[str, str]]:
        """
        Compile all memory tiers into an OpenAI-compatible messages list.

        The system message is structured as:
            [SYSTEM_L0]
            <system prompt>

            [ARCHIVE_L2]          (omitted if empty)
            <compressed history narrative>

            [ENTITY_LEDGER_L1_5]  (omitted if empty)
            <json key-value pairs>

        Followed by raw Working Memory (L1) messages.
        """
        state = self._state
        system_parts: list[str] = []

        system_parts.append(f"[SYSTEM_L0]\n{state.l0_system.content}")

        if state.l2_archival.narrative.strip():
            system_parts.append(
                f"[ARCHIVE_L2]\n{state.l2_archival.narrative.strip()}"
            )

        if state.l1_5_entities.entities:
            system_parts.append(
                f"[ENTITY_LEDGER_L1_5]\n{state.l1_5_entities.to_json_str()}"
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n\n".join(system_parts)}
        ]

        for msg in state.l1_working.messages:
            messages.append(msg.to_openai_dict())

        return messages

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trigger_compression(self) -> None:
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        if not chunk:
            return
        task = CompressionTask(messages=chunk, state=self._state)
        self._worker.enqueue(task)
        logger.info(
            f"Compression triggered: offloaded {len(chunk)} messages to worker."
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def state(self) -> MemoryState:
        return self._state

    def get_stats(self) -> dict:
        return {
            "l0_tokens": self._state.l0_system.token_count,
            "l1_tokens": self._state.l1_working.token_count,
            "l1_message_count": len(self._state.l1_working.messages),
            "l1_5_entity_count": len(self._state.l1_5_entities.entities),
            "l2_tokens": self._monitor.count_text(self._state.l2_archival.narrative),
            "worker": self._worker.stats,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<ContextManager "
            f"l1={stats['l1_tokens']}/{self._config.soft_limit_tokens} tokens, "
            f"l1_msgs={stats['l1_message_count']}, "
            f"l2_tokens={stats['l2_tokens']}, "
            f"entities={stats['l1_5_entity_count']}, "
            f"queue={stats['worker']['queue_depth']}>"
        )
