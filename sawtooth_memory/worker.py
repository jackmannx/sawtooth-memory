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
from typing import Any, Dict, Optional, Union

from .ner import NERPipeline, active_strategy_context
from .compressor import OllamaCompressor, CloudCompressor
from .exceptions import CompressionError, OllamaConnectionError
from .state import ArchivalMemory, EntityLedger, MemoryState, Message

from .events.bus import EventBus
from .events.types import (
    L2SummaryGeneratedEvent,
    L3VectorIndexedEvent,
    CompressionCycleCompleteEvent,
    CompressionCycleFailedEvent,
)
from .journal import AsyncCompressionJournal
from .l3_indexer import SemanticIndexer

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
        enable_deterministic_ner: bool = True,  # NEW
        custom_ner_patterns: Optional[dict] = None,  # NEW
        storage_adapter: Optional[Any] = None,
        pool_id: Optional[str] = None,
        session_id: Optional[str] = None,
        l3_indexer: Optional[SemanticIndexer] = None,
        embedding_backend: str = "hash",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._compressor = compressor
        self._fallback_truncate = fallback_truncate
        self._event_bus = event_bus
        self._journal = journal
        self._storage_adapter = storage_adapter
        self._pool_id = pool_id
        self._session_id = session_id or "unknown_agent"
        self._l3_indexer = l3_indexer
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
        self._queue: asyncio.Queue[CompressionTask | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._processed: int = 0
        self._failed: int = 0

        # Initialize the NER Pipeline
        self._enable_ner = enable_deterministic_ner
        self._ner_pipeline = NERPipeline.from_config(
            enable=self._enable_ner,
            custom_patterns=custom_ner_patterns,
        )

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
            # 1. Run local deterministic regex extraction
            deterministic_entities: dict[str, str] = {}
            if self._enable_ner:
                deterministic_entities = self._ner_pipeline.extract(messages_text)

            # 2. Execute background LLM compression wave as normal
            result = await self._compressor.compress(messages_text)
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            narrative = result.get("narrative_summary", "").strip()
            llm_entities = result.get("extracted_entities", {})
            original_tokens = self._estimate_tokens(messages_text)
            compressed_tokens = self._estimate_tokens(narrative)

            # 3. Secure Merge: Deterministic regex matches override LLM hallucinations
            combined_entities = {**llm_entities, **deterministic_entities}
            result["extracted_entities"] = combined_entities

            # 4. Generate Strategy Mapping Context for Event Consumers
            strategy_map = {k: "deterministic" for k in deterministic_entities}
            for k in llm_entities:
                if k not in strategy_map:
                    strategy_map[k] = "llm_synthesis"

            # 5. Bind context token and merge into state securely
            token = active_strategy_context.set(strategy_map)
            try:
                self._merge(state, result)
            finally:
                active_strategy_context.reset(token)

            try:
                await self._sync_pool_state(combined_entities, narrative)
            except Exception as exc:
                logger.warning(
                    "CompressionWorker: failed to sync pool state (%s).",
                    exc,
                    exc_info=True,
                )

            chunks_indexed = await self._index_l3_semantic(state, messages_text, cycle_id)

            # 6. Emit existing L2SummaryGeneratedEvent and CompressionCycleCompleteEvent
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

            if self._event_bus:
                tokens_evicted = sum(m.token_count for m in task.messages)
                asyncio.create_task(
                    self._event_bus.emit(
                        CompressionCycleCompleteEvent(
                            l1_tokens_evicted=tokens_evicted,
                            l1_5_entities_retained=combined_entities,
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
                f"{len(combined_entities)} entities extracted, "
                f"{chunks_indexed} L3 chunk(s) indexed (cycle {cycle_id})."
            )

        except (OllamaConnectionError, CompressionError) as exc:
            logger.warning(
                f"CompressionWorker: compression failed ({exc}). "
                f"fallback_truncate={self._fallback_truncate}, cycle={cycle_id}"
            )
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
        messages_text = _messages_to_text(messages)
        deterministic_entities: dict[str, str] = {}

        # Even during a provider outage, deterministic values are salvaged
        if self._enable_ner:
            deterministic_entities = self._ner_pipeline.extract(messages_text)

        if deterministic_entities:
            strategy_map = {k: "deterministic" for k in deterministic_entities}
            token = active_strategy_context.set(strategy_map)
            try:
                state.l1_5_entities.upsert(deterministic_entities)
            finally:
                active_strategy_context.reset(token)

        note = (
            f"[COMPRESSION UNAVAILABLE: {len(messages)} messages were truncated. "
            f"First message role: {messages[0].role if messages else 'unknown'}]"
        )
        state.l2_archival.append_narrative(note)
        logger.warning("CompressionWorker: fallback truncation note written to L2.")

    async def _sync_pool_state(
        self, entities_delta: dict[str, str], narrative_delta: str
    ) -> None:
        """
        Persist shared L1.5/L2 state for multi-agent pools after compression.
        """
        adapter = self._storage_adapter
        pool_id = self._pool_id
        if not adapter or not pool_id:
            return

        pool_state = await adapter.load_pool_state(pool_id)
        if pool_state is None:
            shared_entities = EntityLedger()
            shared_archive = ArchivalMemory()
        else:
            shared_entities, shared_archive = pool_state

        if entities_delta:
            shared_entities.upsert(entities_delta)

        if narrative_delta.strip():
            shared_archive.append_narrative(
                f"[origin:{self._session_id}] {narrative_delta.strip()}"
            )

        await adapter.save_pool_state(pool_id, shared_entities, shared_archive)

    async def _index_l3_semantic(
        self,
        state: MemoryState,
        messages_text: str,
        cycle_id: str,
    ) -> int:
        """Index evicted L1 text into L3 semantic vector storage."""
        indexer = self._l3_indexer
        if not indexer or not messages_text.strip():
            return 0

        try:
            chunks_indexed = await indexer.index(
                self._session_id, messages_text, state
            )
        except Exception as exc:
            logger.warning(
                "CompressionWorker: L3 semantic indexing failed (%s).",
                exc,
                exc_info=True,
            )
            return 0

        if chunks_indexed and self._storage_adapter:
            try:
                await self._storage_adapter.save_state(self._session_id, state)
            except Exception as exc:
                logger.warning(
                    "CompressionWorker: failed to persist L3 metadata (%s).",
                    exc,
                    exc_info=True,
                )

        if chunks_indexed and self._event_bus:
            asyncio.create_task(
                self._event_bus.emit(
                    L3VectorIndexedEvent(
                        session_id=self._session_id,
                        cycle_id=cycle_id,
                        chunks_indexed=chunks_indexed,
                        total_chunks=state.l3_semantic.chunk_count,
                        source_chars=len(messages_text),
                        embedding_backend=self._embedding_backend,
                        embedding_model=self._embedding_model,
                    )
                )
            )

        return chunks_indexed

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
