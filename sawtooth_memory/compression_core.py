"""
compression_core.py — Shared compression cycle logic for async and sync runtimes.

Orchestrates NER extraction, LLM compression, entity guard, state merge,
optional pool sync, and optional L3 indexing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from .entity_guard import apply_entity_guard, build_protected_entities
from .exceptions import CompressionError, OllamaConnectionError
from .ner import NERPipeline, active_strategy_context
from .state import ArchivalMemory, EntityLedger, MemoryState, Message

logger = logging.getLogger(__name__)


def messages_to_text(messages: list[Message]) -> str:
    """Flatten a message list to a readable string for the compressor."""
    parts = []
    for msg in messages:
        parts.append(f"{msg.role.upper()}: {msg.content}")
    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimation for event metrics."""
    return len(text) // 4


def merge_compression_into_state(state: MemoryState, result: dict[str, Any]) -> None:
    """Append compression narrative and entities into L2 / L1.5."""
    narrative = result.get("narrative_summary", "").strip()
    entities = result.get("extracted_entities", {})

    if narrative:
        state.l2_archival.append_narrative(narrative)
        logger.debug("compression_core: appended narrative to L2.")

    if entities:
        state.l1_5_entities.upsert(entities)
        logger.debug(
            "compression_core: upserted %d entities into L1.5.",
            len(entities),
        )


def fallback_merge_into_state(
    state: MemoryState,
    messages: list[Message],
    ner_pipeline: NERPipeline,
    *,
    enable_ner: bool,
) -> None:
    """Write a truncation note to L2 and preserve locally extracted entities."""
    messages_text = messages_to_text(messages)

    if enable_ner:
        extraction = ner_pipeline.extract_with_metadata(messages_text)
        if extraction.entities:
            token = active_strategy_context.set(extraction.strategies)
            try:
                state.l1_5_entities.upsert(extraction.entities)
            finally:
                active_strategy_context.reset(token)

    note = (
        f"[COMPRESSION UNAVAILABLE: {len(messages)} messages were truncated. "
        f"First message role: {messages[0].role if messages else 'unknown'}]"
    )
    state.l2_archival.append_narrative(note)
    logger.warning("compression_core: fallback truncation note written to L2.")


class SyncCompressor(Protocol):
    """Minimal synchronous compressor interface."""

    def compress(
        self,
        messages_text: str,
        *,
        protected_entities: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...

    def ping(self) -> None: ...

    def close(self) -> None: ...


class AsyncCompressor(Protocol):
    """Minimal asynchronous compressor interface."""

    async def compress(
        self,
        messages_text: str,
        *,
        protected_entities: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


@dataclass
class CompressionCycleInput:
    """Inputs for a single compression cycle."""

    messages: list[Message]
    state: MemoryState
    cycle_id: str = ""
    pinned_entities: dict[str, str] | None = None


@dataclass
class CompressionCycleOutcome:
    """Result of a compression cycle."""

    success: bool
    narrative: str = ""
    combined_entities: dict[str, str] = field(default_factory=dict)
    duration_ms: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    chunks_indexed: int = 0
    fallback_used: bool = False
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class CompressionEngineConfig:
    """Shared compression engine settings."""

    ner_pipeline: NERPipeline
    enable_ner: bool = True
    fallback_truncate: bool = True
    enable_entity_verifier: bool = True
    storage_adapter: Any | None = None
    pool_id: str | None = None
    session_id: str = "unknown_agent"


def _apply_llm_result(
    state: MemoryState,
    messages_text: str,
    llm_result: dict[str, Any],
    engine: CompressionEngineConfig,
    *,
    pinned_entities: dict[str, str] | None,
) -> tuple[str, dict[str, str]]:
    extraction = engine.ner_pipeline.extract_with_metadata(messages_text)
    narrative = llm_result.get("narrative_summary", "").strip()
    llm_entities = llm_result.get("extracted_entities", {})

    combined_entities, strategy_map = apply_entity_guard(
        extraction,
        llm_entities,
        narrative,
        pinned_entities=pinned_entities,
        enable_verifier=engine.enable_entity_verifier,
    )
    llm_result["extracted_entities"] = combined_entities

    token = active_strategy_context.set(strategy_map)
    try:
        merge_compression_into_state(state, llm_result)
    finally:
        active_strategy_context.reset(token)

    return narrative, combined_entities


def _sync_pool_state_sync(
    engine: CompressionEngineConfig,
    entities_delta: dict[str, str],
    narrative_delta: str,
    run_async: Callable[[Awaitable[Any]], Any],
) -> None:
    adapter = engine.storage_adapter
    pool_id = engine.pool_id
    if not adapter or not pool_id:
        return

    async def _load_and_save() -> None:
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
                f"[origin:{engine.session_id}] {narrative_delta.strip()}"
            )

        await adapter.save_pool_state(pool_id, shared_entities, shared_archive)

    run_async(_load_and_save())


def run_compression_cycle_sync(
    task: CompressionCycleInput,
    compressor: SyncCompressor,
    engine: CompressionEngineConfig,
    *,
    index_l3: Callable[[MemoryState, str, str], int] | None = None,
    run_async: Callable[[Awaitable[Any]], Any] | None = None,
) -> CompressionCycleOutcome:
    """
    Run a full compression cycle on the calling thread (blocking).

    *index_l3* receives (state, messages_text, cycle_id) and returns chunk count.
    *run_async* bridges async storage when pool sync or L3 persistence is needed.
    """
    state = task.state
    messages_text = messages_to_text(task.messages)
    cycle_id = task.cycle_id
    start = time.perf_counter()

    try:
        extraction = engine.ner_pipeline.extract_with_metadata(messages_text)
        protected_entities = build_protected_entities(
            extraction, task.pinned_entities
        )

        result = compressor.compress(
            messages_text,
            protected_entities=protected_entities or None,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)

        narrative, combined_entities = _apply_llm_result(
            state,
            messages_text,
            result,
            engine,
            pinned_entities=task.pinned_entities,
        )
        original_tokens = estimate_tokens(messages_text)
        compressed_tokens = estimate_tokens(narrative)

        if run_async is not None:
            try:
                _sync_pool_state_sync(engine, combined_entities, narrative, run_async)
            except Exception as exc:
                logger.warning(
                    "compression_core: failed to sync pool state (%s).",
                    exc,
                    exc_info=True,
                )

        chunks_indexed = 0
        if index_l3 is not None:
            chunks_indexed = index_l3(state, messages_text, cycle_id)

        logger.info(
            "compression_core: compressed %d messages → narrative (%d chars), "
            "%d entities, %d L3 chunk(s) (cycle %s).",
            len(task.messages),
            len(narrative),
            len(combined_entities),
            chunks_indexed,
            cycle_id,
        )

        return CompressionCycleOutcome(
            success=True,
            narrative=narrative,
            combined_entities=combined_entities,
            duration_ms=duration_ms,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            chunks_indexed=chunks_indexed,
        )

    except (OllamaConnectionError, CompressionError) as exc:
        logger.warning(
            "compression_core: compression failed (%s). fallback_truncate=%s cycle=%s",
            exc,
            engine.fallback_truncate,
            cycle_id,
        )
        if engine.fallback_truncate:
            fallback_merge_into_state(
                state,
                task.messages,
                engine.ner_pipeline,
                enable_ner=engine.enable_ner,
            )
            chunks_indexed = 0
            if index_l3 is not None:
                chunks_indexed = index_l3(state, messages_text, cycle_id)
            duration_ms = int((time.perf_counter() - start) * 1000)
            return CompressionCycleOutcome(
                success=False,
                duration_ms=duration_ms,
                original_tokens=estimate_tokens(messages_text),
                chunks_indexed=chunks_indexed,
                fallback_used=True,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        raise


async def run_compression_cycle_async(
    task: CompressionCycleInput,
    compressor: AsyncCompressor,
    engine: CompressionEngineConfig,
    *,
    index_l3: Callable[[MemoryState, str, str], Awaitable[int]] | None = None,
    on_success: Callable[[CompressionCycleOutcome], Awaitable[None]] | None = None,
    on_failure: Callable[[CompressionCycleOutcome], Awaitable[None]] | None = None,
) -> CompressionCycleOutcome:
    """Run a full compression cycle on the active asyncio event loop."""
    import asyncio

    state = task.state
    messages_text = messages_to_text(task.messages)
    cycle_id = task.cycle_id
    loop = asyncio.get_running_loop()
    start = loop.time()

    try:
        extraction = engine.ner_pipeline.extract_with_metadata(messages_text)
        protected_entities = build_protected_entities(
            extraction, task.pinned_entities
        )

        result = await compressor.compress(
            messages_text,
            protected_entities=protected_entities or None,
        )
        duration_ms = int((loop.time() - start) * 1000)

        narrative, combined_entities = _apply_llm_result(
            state,
            messages_text,
            result,
            engine,
            pinned_entities=task.pinned_entities,
        )
        original_tokens = estimate_tokens(messages_text)
        compressed_tokens = estimate_tokens(narrative)

        if engine.storage_adapter and engine.pool_id:
            try:
                pool_state = await engine.storage_adapter.load_pool_state(
                    engine.pool_id
                )
                if pool_state is None:
                    shared_entities = EntityLedger()
                    shared_archive = ArchivalMemory()
                else:
                    shared_entities, shared_archive = pool_state

                if combined_entities:
                    shared_entities.upsert(combined_entities)

                if narrative.strip():
                    shared_archive.append_narrative(
                        f"[origin:{engine.session_id}] {narrative.strip()}"
                    )

                await engine.storage_adapter.save_pool_state(
                    engine.pool_id, shared_entities, shared_archive
                )
            except Exception as exc:
                logger.warning(
                    "compression_core: failed to sync pool state (%s).",
                    exc,
                    exc_info=True,
                )

        chunks_indexed = 0
        if index_l3 is not None:
            chunks_indexed = await index_l3(state, messages_text, cycle_id)

        outcome = CompressionCycleOutcome(
            success=True,
            narrative=narrative,
            combined_entities=combined_entities,
            duration_ms=duration_ms,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            chunks_indexed=chunks_indexed,
        )

        if on_success is not None:
            await on_success(outcome)

        logger.info(
            "compression_core: compressed %d messages → narrative (%d chars), "
            "%d entities, %d L3 chunk(s) (cycle %s).",
            len(task.messages),
            len(narrative),
            len(combined_entities),
            chunks_indexed,
            cycle_id,
        )
        return outcome

    except (OllamaConnectionError, CompressionError) as exc:
        logger.warning(
            "compression_core: compression failed (%s). fallback_truncate=%s cycle=%s",
            exc,
            engine.fallback_truncate,
            cycle_id,
        )
        duration_ms = int((loop.time() - start) * 1000)
        if engine.fallback_truncate:
            fallback_merge_into_state(
                state,
                task.messages,
                engine.ner_pipeline,
                enable_ner=engine.enable_ner,
            )
            chunks_indexed = 0
            if index_l3 is not None:
                chunks_indexed = await index_l3(state, messages_text, cycle_id)

            outcome = CompressionCycleOutcome(
                success=False,
                duration_ms=duration_ms,
                original_tokens=estimate_tokens(messages_text),
                chunks_indexed=chunks_indexed,
                fallback_used=True,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            if on_failure is not None:
                await on_failure(outcome)
            return outcome
        raise
