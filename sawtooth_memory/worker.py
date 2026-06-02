"""
worker.py — Async background compression worker.

Runs as an asyncio Task. Pulls CompressionTasks from a queue, calls the
compressor (Ollama or cloud), then merges results into the MemoryState —
all without blocking the main agent thread.

Graceful degradation: if compression fails, appends a truncation note to L2
instead of crashing (if fallback_truncate is True).

- Emits structured telemetry events (L2 summary generated, cycle complete/failed).
- The CompressionCycleCompleteEvent triggers the journal writer (async JSONL).
- All emissions are fire‑and‑forget; the main compression path never blocks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Union, Optional, Any, Dict

from .compressor import OllamaCompressor, CloudCompressor
from .exceptions import CompressionError, OllamaConnectionError
from .state import MemoryState, Message

# Phase 2 event imports (optional – may be None if events disabled)
from .events.bus import EventBus
from .events.types import (
    L2SummaryGeneratedEvent,
    CompressionCycleCompleteEvent,
    CompressionCycleFailedEvent,
)
from .journal import AsyncCompressionJournal

logger = logging.getLogger(__name__)

_SENTINEL = None


@dataclass
class CompressionTask:
    """A chunk of messages queued for background compression."""

    messages: list[Message]
    state: MemoryState
    cycle_id: str = ""  # Unique ID for this compression cycle (for event correlation)


def _messages_to_text(messages: list[Message]) -> str:
    """Flatten a message list to a readable string for the compressor."""
    parts = []
    for msg in messages:
        parts.append(f"{msg.role.upper()}: {msg.content}")
    return "\n\n".join(parts)


class CompressionWorker:
    """
    Background asyncio worker that processes compression tasks off the
    critical path.

    Lifecycle:
        worker = CompressionWorker(compressor, fallback_truncate=True,
                                   event_bus=bus, journal=journal)
        await worker.start()
        worker.enqueue(task)          # non-blocking
        await worker.stop()           # drains queue then exits
    """

    def __init__(
        self,
        compressor: Union[OllamaCompressor, CloudCompressor],
        fallback_truncate: bool = True,
        event_bus: Optional[EventBus] = None,
        journal: Optional[AsyncCompressionJournal] = None,
    ) -> None:
        self._compressor = compressor
        self._fallback_truncate = fallback_truncate
        self._event_bus = event_bus  # None if events disabled
        self._journal = (
            journal  # None if journal disabled (but event handler will use it)
        )

        self._queue: asyncio.Queue[CompressionTask | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._processed: int = 0
        self._failed: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._loop(), name="sawtooth-compression-worker"
        )
        logger.info("CompressionWorker: started.")

    async def stop(self) -> None:
        """
        Signal the worker to stop after draining the queue.
        Waits for in-flight compression to finish before returning.
        """
        if not self._running:
            return
        self._running = False
        await self._queue.put(_SENTINEL)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=120)
            except asyncio.TimeoutError:
                logger.warning(
                    "CompressionWorker: shutdown timed out, cancelling task."
                )
                self._task.cancel()
        await self._compressor.close()
        logger.info(
            f"CompressionWorker: stopped. "
            f"Processed={self._processed}, Failed={self._failed}"
        )

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, task: CompressionTask) -> None:
        """Put a task on the queue. Returns immediately; does not block."""
        self._queue.put_nowait(task)
        logger.debug(
            f"CompressionWorker: enqueued {len(task.messages)} messages (cycle={task.cycle_id}). "
            f"Queue depth: {self._queue.qsize()}"
        )

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            task = await self._queue.get()
            if task is _SENTINEL:
                self._queue.task_done()
                break
            try:
                await self._process(task)
                self._processed += 1
            except Exception as exc:
                self._failed += 1
                logger.error(
                    f"CompressionWorker: unhandled error processing task: {exc}",
                    exc_info=True,
                )
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Task processing (Phase 2 events integrated)
    # ------------------------------------------------------------------

    async def _process(self, task: CompressionTask) -> None:
        state = task.state
        messages_text = _messages_to_text(task.messages)
        cycle_id = task.cycle_id
        start_time = asyncio.get_event_loop().time()

        try:
            # Perform compression (calls external LLM)
            result = await self._compressor.compress(messages_text)
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Extract data from result
            narrative = result.get("narrative_summary", "").strip()
            entities = result.get("extracted_entities", {})
            original_tokens = self._estimate_tokens(messages_text)
            compressed_tokens = self._estimate_tokens(narrative)

            # 1. Merge results into memory state (upsert will trigger entity events via callback)
            self._merge(state, result)

            # 2. Emit L2 summary generated event (if bus enabled)
            if self._event_bus:
                asyncio.create_task(
                    self._event_bus.emit(
                        L2SummaryGeneratedEvent(
                            summary_text=narrative,
                            compressed_message_count=len(task.messages),
                            original_tokens=original_tokens,
                            compressed_tokens=compressed_tokens,
                            compression_ratio=original_tokens
                            / max(compressed_tokens, 1),
                            provider=self._get_provider_name(),
                            model=self._get_model_name(),
                            compression_duration_ms=duration_ms,
                            fallback_used=False,
                            cycle_id=cycle_id,
                        )
                    )
                )

            # 3. Emit compression cycle complete event (triggers journal write via handler)
            if self._event_bus:
                # Calculate tokens evicted (approximate: sum of message tokens)
                tokens_evicted = sum(m.token_count for m in task.messages)
                asyncio.create_task(
                    self._event_bus.emit(
                        CompressionCycleCompleteEvent(
                            l1_tokens_evicted=tokens_evicted,
                            l1_5_entities_retained=entities,  # exact entities preserved
                            l2_summary_generated=narrative,
                            messages_compressed=len(task.messages),
                            final_l1_tokens=state.l1_working.token_count,
                            total_duration_ms=duration_ms,
                            cycle_id=cycle_id,
                        )
                    )
                )

            logger.info(
                f"CompressionWorker: compressed {len(task.messages)} messages → "
                f"narrative ({len(narrative)} chars), "
                f"{len(entities)} entities extracted (cycle {cycle_id})."
            )

        except (OllamaConnectionError, CompressionError) as exc:
            logger.warning(
                f"CompressionWorker: compression failed ({exc}). "
                f"fallback_truncate={self._fallback_truncate}, cycle={cycle_id}"
            )

            # Emit failure event (if bus enabled)
            if self._event_bus:
                asyncio.create_task(
                    self._event_bus.emit(
                        CompressionCycleFailedEvent(
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            fallback_triggered=self._fallback_truncate,
                            cycle_id=cycle_id,
                        )
                    )
                )

            if self._fallback_truncate:
                self._fallback_merge(state, task.messages)
            else:
                raise

    # ------------------------------------------------------------------
    # State merging
    # ------------------------------------------------------------------

    def _merge(self, state: MemoryState, result: Dict[str, Any]) -> None:
        narrative = result.get("narrative_summary", "").strip()
        entities = result.get("extracted_entities", {})

        if narrative:
            state.l2_archival.append_narrative(narrative)
            logger.debug("CompressionWorker: appended narrative to L2.")

        if entities:
            # This upsert will call the entity event callback if set on the ledger
            state.l1_5_entities.upsert(entities)
            logger.debug(
                f"CompressionWorker: upserted {len(entities)} entities into L1.5."
            )

    def _fallback_merge(self, state: MemoryState, messages: list[Message]) -> None:
        note = (
            f"[COMPRESSION UNAVAILABLE: {len(messages)} messages were truncated. "
            f"First message role: {messages[0].role if messages else 'unknown'}]"
        )
        state.l2_archival.append_narrative(note)
        logger.warning("CompressionWorker: fallback truncation note written to L2.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (uses tiktoken if available, else rough)."""
        # We could inject a tokenizer, but for event metrics an approximation is fine.
        # For accurate counts, the monitor uses tiktoken. Here we keep it simple.
        return len(text) // 4  # Very rough: 4 chars per token

    def _get_provider_name(self) -> str:
        """Return a human-readable provider name."""
        if hasattr(self._compressor, "provider"):
            return self._compressor.provider  # type: ignore
        if isinstance(self._compressor, OllamaCompressor):
            return "ollama"
        if isinstance(self._compressor, CloudCompressor):
            return "cloud"
        return "unknown"

    def _get_model_name(self) -> str:
        """Return the model name from the compressor."""
        if hasattr(self._compressor, "model"):
            return self._compressor.model  # type: ignore
        return "unknown"

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "processed": self._processed,
            "failed": self._failed,
            "queue_depth": self.queue_depth,
            "running": self._running,
        }
