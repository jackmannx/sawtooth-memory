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
from typing import Any, Literal, Optional, Union

from .compression_core import (
    CompressionCycleInput,
    CompressionEngineConfig,
    messages_to_text,
    run_compression_cycle_async,
)
from .compressor import CloudCompressor, OllamaCompressor
from .dte_runtime import apply_fold_delta_to_pool
from .events.bus import EventBus
from .events.types import (
    CompressionCycleCompleteEvent,
    CompressionCycleFailedEvent,
    DTEFoldCreatedEvent,
    L2SummaryGeneratedEvent,
    L3VectorIndexedEvent,
)
from .fold_unit import remove_fold_lines
from .l3_indexer import SemanticIndexer
from .ner import NERPipeline
from .state import ArchivalMemory, EntityLedger, MemoryState, Message

logger = logging.getLogger(__name__)

_SENTINEL = None


@dataclass
class CompressionTask:
    """A chunk of messages queued for background compression."""

    messages: list[Message]
    state: MemoryState
    cycle_id: str = ""  # Unique ID for this compression cycle (for event correlation)
    task_kind: Literal["compress", "consolidate", "fold_finalize"] = "compress"
    fold_stub: str = ""
    entity_keys: tuple[str, ...] = ()
    tokens_evicted: int = 0


def _messages_to_text(messages: list[Message]) -> str:
    """Backward-compatible alias for middleware imports."""
    return messages_to_text(messages)


class CompressionWorker:
    """
    Background asyncio worker that processes compression tasks off the
    critical path.

    Lifecycle:
        worker = CompressionWorker(compressor, fallback_truncate=True,
                                   event_bus=bus)
        await worker.start()
        worker.enqueue(task)          # non-blocking
        await worker.stop()           # drains queue then exits
    """

    def __init__(
        self,
        compressor: Union[OllamaCompressor, CloudCompressor],
        fallback_truncate: bool = True,
        event_bus: Optional[EventBus] = None,
        enable_deterministic_ner: bool = True,  # NEW
        custom_ner_patterns: Optional[dict] = None,  # NEW
        enable_salience_extractor: bool = True,
        salience_threshold: float = 0.5,
        salience_max_entities: int = 20,
        enable_entity_verifier: bool = True,
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

        self._enable_ner = enable_deterministic_ner
        self._enable_entity_verifier = enable_entity_verifier
        self._ner_pipeline = NERPipeline.from_config(
            enable=self._enable_ner,
            custom_patterns=custom_ner_patterns,
            enable_salience=enable_salience_extractor,
            salience_threshold=salience_threshold,
            salience_max_entities=salience_max_entities,
        )
        self._engine = CompressionEngineConfig(
            ner_pipeline=self._ner_pipeline,
            enable_ner=self._enable_ner,
            fallback_truncate=self._fallback_truncate,
            enable_entity_verifier=self._enable_entity_verifier,
            storage_adapter=self._storage_adapter,
            pool_id=self._pool_id,
            session_id=self._session_id,
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
        if task.task_kind == "fold_finalize":
            await self._finalize_fold(task)
            return

        cycle_input = CompressionCycleInput(
            messages=task.messages,
            state=task.state,
            cycle_id=task.cycle_id,
        )

        async def on_success(outcome) -> None:
            task.state.dte.background_llm_input_tokens += sum(
                message.token_count for message in task.messages
            )
            if task.task_kind == "consolidate":
                task.state.l2_archival.narrative = remove_fold_lines(
                    task.state.l2_archival.narrative
                )
                if self._storage_adapter and self._pool_id:
                    try:
                        pool_state = await self._storage_adapter.load_pool_state(
                            self._pool_id
                        )
                        if pool_state is not None:
                            shared_entities, shared_archive = pool_state
                            shared_archive.narrative = remove_fold_lines(
                                shared_archive.narrative
                            )
                            await self._storage_adapter.save_pool_state(
                                self._pool_id, shared_entities, shared_archive
                            )
                    except Exception as exc:
                        logger.warning(
                            "CompressionWorker: failed to compact shared fold "
                            "records (%s).",
                            exc,
                            exc_info=True,
                        )
                task.state.dte.narrative_debt_tokens = 0
                task.state.dte.folds_since_narrative = 0
                task.state.dte.consolidation_cycles += 1
                task.state.dte.consolidation_queued = False
            if not self._event_bus:
                return
            asyncio.create_task(
                self._event_bus.emit(
                    L2SummaryGeneratedEvent(
                        summary_text=outcome.narrative,
                        compressed_message_count=len(task.messages),
                        original_tokens=outcome.original_tokens,
                        compressed_tokens=outcome.compressed_tokens,
                        compression_ratio=outcome.original_tokens
                        / max(outcome.compressed_tokens, 1),
                        provider=self._get_provider_name(),
                        model=self._get_model_name(),
                        compression_duration_ms=outcome.duration_ms,
                        fallback_used=False,
                        cycle_id=task.cycle_id,
                    )
                )
            )
            tokens_evicted = sum(m.token_count for m in task.messages)
            asyncio.create_task(
                self._event_bus.emit(
                    CompressionCycleCompleteEvent(
                        l1_tokens_evicted=tokens_evicted,
                        l1_5_entities_retained=outcome.combined_entities,
                        l2_summary_generated=outcome.narrative,
                        messages_compressed=len(task.messages),
                        final_l1_tokens=task.state.l1_working.token_count,
                        total_duration_ms=outcome.duration_ms,
                        cycle_id=task.cycle_id,
                    )
                )
            )
            if outcome.chunks_indexed and self._event_bus:
                asyncio.create_task(
                    self._event_bus.emit(
                        L3VectorIndexedEvent(
                            session_id=self._session_id,
                            cycle_id=task.cycle_id,
                            chunks_indexed=outcome.chunks_indexed,
                            total_chunks=task.state.l3_semantic.chunk_count,
                            source_chars=len(messages_to_text(task.messages)),
                            embedding_backend=self._embedding_backend,
                            embedding_model=self._embedding_model,
                        )
                    )
                )

        async def on_failure(outcome) -> None:
            task.state.dte.background_llm_input_tokens += sum(
                message.token_count for message in task.messages
            )
            if task.task_kind == "consolidate":
                task.state.l2_archival.narrative = "\n".join(
                    line
                    for line in task.state.l2_archival.narrative.splitlines()
                    if not line.startswith("[COMPRESSION UNAVAILABLE:")
                ).strip()
                task.state.dte.consolidation_queued = False
            if not self._event_bus:
                return
            asyncio.create_task(
                self._event_bus.emit(
                    CompressionCycleFailedEvent(
                        error_type=outcome.error_type or "Unknown",
                        error_message=outcome.error_message or "",
                        fallback_triggered=outcome.fallback_used,
                        cycle_id=task.cycle_id,
                    )
                )
            )

        try:
            await run_compression_cycle_async(
                cycle_input,
                self._compressor,
                self._engine,
                index_l3=(
                    self._index_l3_semantic if task.task_kind == "compress" else None
                ),
                on_success=on_success,
                on_failure=on_failure,
            )
        finally:
            if task.task_kind == "consolidate":
                task.state.dte.consolidation_queued = False

    async def _finalize_fold(self, task: CompressionTask) -> None:
        """Index L3 and sync pool for a DTE fold without blocking ingest."""
        messages_text = messages_to_text(task.messages)
        l3_chunks = await self._index_l3_semantic(
            task.state, messages_text, task.cycle_id
        )

        if self._storage_adapter and self._pool_id and task.fold_stub:
            try:
                pool_state = await self._storage_adapter.load_pool_state(self._pool_id)
                if pool_state is None:
                    shared_entities = EntityLedger()
                    shared_archive = ArchivalMemory()
                else:
                    shared_entities, shared_archive = pool_state
                apply_fold_delta_to_pool(
                    session_id=self._session_id,
                    fold_stub=task.fold_stub,
                    entity_keys=task.entity_keys,
                    local_entities=task.state.l1_5_entities,
                    shared_entities=shared_entities,
                    shared_archive=shared_archive,
                )
                await self._storage_adapter.save_pool_state(
                    self._pool_id, shared_entities, shared_archive
                )
            except Exception as exc:
                logger.warning(
                    "CompressionWorker: fold pool sync failed (%s).",
                    exc,
                    exc_info=True,
                )

        if self._event_bus:
            asyncio.create_task(
                self._event_bus.emit(
                    DTEFoldCreatedEvent(
                        cycle_id=task.cycle_id,
                        messages_folded=len(task.messages),
                        tokens_evicted=task.tokens_evicted,
                        entity_keys=list(task.entity_keys),
                        l3_chunks_indexed=l3_chunks,
                        recoverable=l3_chunks > 0,
                    )
                )
            )
            if l3_chunks:
                asyncio.create_task(
                    self._event_bus.emit(
                        L3VectorIndexedEvent(
                            session_id=self._session_id,
                            cycle_id=task.cycle_id,
                            chunks_indexed=l3_chunks,
                            total_chunks=task.state.l3_semantic.chunk_count,
                            source_chars=len(messages_text),
                            embedding_backend=self._embedding_backend,
                            embedding_model=self._embedding_model,
                        )
                    )
                )

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

        return chunks_indexed

    async def index_l3_semantic(
        self,
        state: MemoryState,
        messages_text: str,
        cycle_id: str,
    ) -> int:
        """Public entry point for L3 indexing (used by ContextManager hard-truncate)."""
        return await self._index_l3_semantic(state, messages_text, cycle_id)

    @property
    def ner_pipeline(self) -> NERPipeline:
        """Expose the NER pipeline for ingest-time entity scanning."""
        return self._ner_pipeline

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
