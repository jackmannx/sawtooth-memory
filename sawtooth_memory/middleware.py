"""
middleware.py — ContextManager: the primary public API for Sawtooth-Memory.

Drop this between your agent loop and your LLM API call.

Quick start:
    from sawtooth_memory import ContextManager, ContextManagerConfig

    config = ContextManagerConfig(soft_limit_tokens=3000)

    async with ContextManager("You are a data analysis agent.", config) as cm:
        await cm.add_message("user", "Analyse Q3 revenue.")
        await cm.add_message("assistant", "Connecting to the database...")

        messages = cm.build_prompt()
        # response = await openai_client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=messages,
        # )
"""

from __future__ import annotations

import logging

from .config import ContextManagerConfig
from .compressor import CloudCompressor, OllamaCompressor
from .exceptions import TokenLimitExceededError
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

    Responsibilities:
    - Maintains the four-tier memory state (L0 / L1 / L1.5 / L2).
    - Counts tokens locally via tiktoken (no API call needed).
    - When L1 exceeds the soft limit, slices the oldest chunk and queues
      it for async compression via a background Ollama or Cloud worker.
    - When L1 exceeds the hard limit, falls back to synchronous truncation
      or raises TokenLimitExceededError (configurable).
    - Compiles all tiers into a structured prompt via build_prompt().

    Thread safety:
        asyncio is single-threaded cooperative. The background worker only
        mutates shared state between `await` checkpoints, so no explicit
        locking is required.

    Usage:
        async with ContextManager(system_prompt, config) as cm:
            await cm.add_message("user", "...")
            messages = cm.build_prompt()
    """

    def __init__(
        self,
        system_prompt: str,
        config: ContextManagerConfig | None = None,
    ) -> None:
        # 1. Assign configuration first to prevent AttributeError
        self._config = config or ContextManagerConfig()

        self._monitor = TokenMonitor(
            model=self._config.tokenizer_model,
            soft_limit=self._config.soft_limit_tokens,
            hard_limit=self._config.hard_limit_tokens,
        )

        sp_tokens = self._monitor.count_text(system_prompt)
        # 2. Initialize memory state
        self._state = MemoryState(
            l0_system=SystemPrompt(content=system_prompt, token_count=sp_tokens),
            l1_working=WorkingMemory(),
            l1_5_entities=EntityLedger(),
            l2_archival=ArchivalMemory(),
        )

        # 3. Dynamic Compressor Routing
        if self._config.cloud:
            self._compressor = CloudCompressor(self._config.cloud)
        else:
            self._compressor = OllamaCompressor(self._config.ollama)

        self._worker = CompressionWorker(
            compressor=self._compressor,
            fallback_truncate=self._config.fallback_truncate,
        )

        logger.debug(
            f"ContextManager initialised. "
            f"soft_limit={self._config.soft_limit_tokens}, "
            f"hard_limit={self._config.hard_limit_tokens}, "
            f"chunk_size={self._config.chunk_size}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background compression worker. Must be called before use."""
        await self._worker.start()

    async def stop(self) -> None:
        """Drain the compression queue and shut down the worker."""
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
        background compression — without blocking this call.

        If the hard limit is crossed and fallback_truncate is False,
        raises TokenLimitExceededError.
        """
        msg = Message(role=role, content=content)
        msg.token_count = self._monitor.count_message(msg)
        self._state.l1_working.append(msg)

        logger.debug(
            f"add_message: role={role}, tokens={msg.token_count}, "
            f"l1_total={self._state.l1_working.token_count}"
        )

        if self._monitor.exceeds_hard_limit(self._state):
            if not self._config.fallback_truncate:
                raise TokenLimitExceededError(
                    f"Working Memory exceeded hard limit of "
                    f"{self._config.hard_limit_tokens} tokens and "
                    f"fallback_truncate is disabled."
                )
            logger.warning(
                "Hard token limit reached before compression finished. "
                "Forcing immediate truncation of oldest messages."
            )
            self._force_truncate()

        elif self._monitor.exceeds_soft_limit(self._state):
            self._trigger_compression()

    def build_prompt(self) -> list[dict[str, str]]:
        """
        Compile all memory tiers into an OpenAI-compatible messages list.

        Returns a list of {"role": "...", "content": "..."} dicts, ready
        to pass directly to openai.chat.completions.create() or equivalent.

        Structure of the injected system message:
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
    # Internal compression triggers
    # ------------------------------------------------------------------

    def _trigger_compression(self) -> None:
        """
        Non-blocking: slice the oldest chunk and hand it off to the worker.
        The main thread continues running immediately.
        """
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        if not chunk:
            return

        task = CompressionTask(messages=chunk, state=self._state)
        self._worker.enqueue(task)

        logger.info(
            f"Compression triggered: offloaded {len(chunk)} messages to worker. "
            f"L1 remaining: {self._state.l1_working.token_count} tokens"
        )

    def _force_truncate(self) -> None:
        """
        Hard-limit fallback: discard the oldest messages immediately on
        the main thread without waiting for Ollama/Cloud.
        """
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        note = (
            f"[HARD TRUNCATION: {len(chunk)} messages dropped because the "
            f"compression worker has not yet caught up.]"
        )
        self._state.l2_archival.append_narrative(note)
        logger.warning(f"Hard truncation: dropped {len(chunk)} messages from L1.")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def state(self) -> MemoryState:
        """Read-only access to the current MemoryState."""
        return self._state

    def get_stats(self) -> dict:
        """
        Return a snapshot of current token usage and worker health.

        Returns:
            {
                "l0_tokens": int,
                "l1_tokens": int,
                "l1_message_count": int,
                "l1_5_entity_count": int,
                "l2_tokens": int,
                "worker": {"processed": int, "failed": int, "queue_depth": int, ...}
            }
        """
        return {
            "l0_tokens": self._state.l0_system.token_count,
            "l1_tokens": self._state.l1_working.token_count,
            "l1_message_count": len(self._state.l1_working.messages),
            "l1_5_entity_count": len(self._state.l1_5_entities.entities),
            "l2_tokens": self._monitor.count_text(
                self._state.l2_archival.narrative
            ),
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