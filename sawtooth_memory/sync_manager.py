"""
sync_manager.py — Sync-native ContextManager for scripts and WSGI applications.

Runs compression inline on the calling thread. No asyncio event loop, background
worker, or AnyIO blocking portal is required.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Union

from .async_bridge import run_coro_once
from .compression_core import (
    CompressionCycleInput,
    CompressionEngineConfig,
    messages_to_text,
    run_compression_cycle_sync,
)
from .config import ContextManagerConfig, OllamaConfig
from .embeddings.factory import create_embedding_provider
from .exceptions import TokenLimitExceededError
from .fold_unit import create_fold_unit, fold_lines, remove_fold_lines
from .intent_planner import PromptIntentPlan, plan_prompt
from .l3_indexer import SemanticIndexer
from .middleware import _extract_entity_event
from .monitor import TokenMonitor
from .ner import NERPipeline, active_strategy_context
from .novelty import residualize
from .observation_crush import crush_observation
from .prompt_compiler import (
    compile_prompt,
    format_l3_retrieval_block,
    resolve_l3_retrieval_query,
)
from .state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    Message,
    MessageRole,
    SystemPrompt,
    WorkingMemory,
)
from .storage.semantic import SemanticChunkResult
from .sync_compressor import SyncCloudCompressor, SyncOllamaCompressor

logger = logging.getLogger(__name__)


class SyncContextManager:
    """
    Synchronous memory manager with full L0–L2 tier support and optional L3.

    Compression runs inline when token limits are reached (blocking). For
    non-blocking compression in sync hosts, use SawtoothSyncWrapper instead.
    """

    def __init__(
        self,
        system_prompt: str,
        config: ContextManagerConfig | None = None,
    ) -> None:
        self._config = config or ContextManagerConfig.for_sync_script()
        self._monitor = TokenMonitor(
            model=self._config.tokenizer_model,
            soft_limit=self._config.soft_limit_tokens,
            hard_limit=self._config.hard_limit_tokens,
            max_unsummarized_turns=self._config.max_unsummarized_turns,
            event_bus=None,
        )

        sp_tokens = self._monitor.count_text(system_prompt)
        self._state = MemoryState(
            l0_system=SystemPrompt(content=system_prompt, token_count=sp_tokens),
            l1_working=WorkingMemory(),
            l1_5_entities=EntityLedger(),
            l2_archival=ArchivalMemory(),
        )

        self._compressor: Union[SyncCloudCompressor, SyncOllamaCompressor]
        if self._config.cloud:
            self._compressor = SyncCloudCompressor(self._config.cloud)
        else:
            ollama_cfg = (
                self._config.ollama
                if self._config.ollama is not None
                else OllamaConfig()
            )
            self._compressor = SyncOllamaCompressor(ollama_cfg)

        self._ner_pipeline = NERPipeline.from_config(
            enable=self._config.enable_deterministic_ner,
            custom_patterns=self._config.custom_ner_patterns,
            enable_salience=self._config.enable_salience_extractor,
            salience_threshold=self._config.salience_threshold,
            salience_max_entities=self._config.salience_max_entities,
        )
        self._engine = CompressionEngineConfig(
            ner_pipeline=self._ner_pipeline,
            enable_ner=self._config.enable_deterministic_ner,
            fallback_truncate=self._config.fallback_truncate,
            enable_entity_verifier=self._config.enable_entity_verifier,
            storage_adapter=self._config.storage_adapter,
            pool_id=self._config.pool_id,
            session_id=self._config.session_id,
        )

        self._l3_indexer: SemanticIndexer | None = None
        if self._config.enable_l3_semantic_storage:
            embedder = create_embedding_provider(
                self._config.embedding_backend,  # type: ignore[arg-type]
                model=self._config.embedding_model,
                dimension=self._config.embedding_dimension,
            )
            self._l3_indexer = SemanticIndexer(
                storage=self._config.storage_adapter,
                embedder=embedder,
                chunk_max_chars=self._config.l3_chunk_max_chars,
            )

        self._last_l3_retrieval: list[dict[str, Any]] = []
        self._observation_cache: OrderedDict[str, str] = OrderedDict()
        self._compression_cycles: int = 0
        self._compression_failures: int = 0
        self._started = False

    def __enter__(self) -> "SyncContextManager":
        self._load_state()
        self._sync_pool_state_from_storage()
        self._started = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._compressor.close()
        self._started = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to L1; compress inline when limits are exceeded."""
        if not self._started:
            raise RuntimeError(
                "SyncContextManager must be used within a 'with' context."
            )

        stored_content = content
        if self._config.enable_observation_crush and role == "tool":
            crushed = crush_observation(
                content,
                count_text=self._monitor.count_text,
                min_tokens=self._config.obs_crush_min_tokens,
            )
            stored_content = crushed.content
            if crushed.crushed and crushed.cache_id:
                self._observation_cache[crushed.cache_id] = content
                self._observation_cache.move_to_end(crushed.cache_id)
                while (
                    len(self._observation_cache)
                    > self._config.obs_cache_max_entries
                ):
                    self._observation_cache.popitem(last=False)
                self._state.dte.observation_tokens_saved += crushed.tokens_saved

        msg = Message(role=role, content=stored_content)
        msg.token_count = self._monitor.count_message(msg)
        self._state.l1_working.append(msg)

        if (
            self._config.enable_ingest_entity_scan
            and self._config.enable_deterministic_ner
        ):
            self._scan_message_entities(content)

        if self._monitor.exceeds_hard_limit(self._state):
            if not self._config.fallback_truncate:
                raise TokenLimitExceededError(
                    f"Working Memory exceeded hard limit of "
                    f"{self._config.hard_limit_tokens} tokens and "
                    f"fallback_truncate is disabled."
                )
            logger.warning(
                "Hard token limit reached. Forcing immediate truncation."
            )
            self._force_truncate()
        elif self._monitor.should_trigger_compression(self._state):
            self._trigger_compression()

        if self._config.storage_adapter:
            run_coro_once(
                self._config.storage_adapter.save_state(
                    self._config.session_id, self._state
                )
            )

    def retrieve_observation(self, cache_id: str) -> str | None:
        """Return a raw tool observation retained by Observation Crush."""
        value = self._observation_cache.get(cache_id)
        if value is not None:
            self._observation_cache.move_to_end(cache_id)
        return value

    def pin_entity(self, key: str, value: str) -> None:
        """Pin a critical entity into the L1.5 ledger."""
        token = active_strategy_context.set({key: "pinned"})
        try:
            self._state.l1_5_entities.upsert({key: value})
        finally:
            active_strategy_context.reset(token)

        if self._config.storage_adapter:
            run_coro_once(
                self._config.storage_adapter.save_state(
                    self._config.session_id, self._state
                )
            )

    def build_prompt(self, *, retrieval_query: str | None = None) -> list[dict[str, str]]:
        """Compile all memory tiers into an OpenAI-compatible messages list."""
        if not self._started:
            raise RuntimeError(
                "SyncContextManager must be used within a 'with' context."
            )

        self._sync_pool_state_from_storage()
        l3_block = ""
        l3_retrieval: list[dict[str, Any]] = []
        query = resolve_l3_retrieval_query(self._state, retrieval_query)
        if self._config.enable_intent_prompt_planner:
            intent_plan = plan_prompt(
                query, self._state.l1_5_entities, self._config
            )
        else:
            intent_plan = PromptIntentPlan(
                intent="general",
                include_l2=True,
                ledger_covers_query=False,
                l3_top_k=self._config.l3_retrieval_top_k,
                l3_token_budget=self._config.l3_retrieval_max_tokens,
            )

        if self._config.enable_l3_prompt_retrieval and self._l3_indexer:
            if query:
                results = run_coro_once(
                    self._l3_indexer.search(
                        self._config.session_id,
                        query,
                        top_k=intent_plan.l3_top_k,
                    )
                )
                l3_block, l3_retrieval = format_l3_retrieval_block(
                    results,
                    token_budget=intent_plan.l3_token_budget,
                    count_text=self._monitor.count_text,
                )

        result = compile_prompt(
            self._state,
            self._config,
            l3_block=l3_block,
            l3_retrieval=l3_retrieval,
            include_l2=intent_plan.include_l2,
        )
        self._last_l3_retrieval = result.l3_retrieval
        self._state.dte.main_prompt_tokens += sum(
            self._monitor.count_text(message["content"]) for message in result.messages
        )
        self._maybe_consolidate(intent_plan)
        return result.messages

    def explain_prompt(self) -> dict[str, Any]:
        """Return explainability traces for the active prompt."""
        trace: dict[str, Any] = {
            "l0_system": {
                "content": self._state.l0_system.content,
                "origin": "Hardcoded System Initialization",
            },
            "l2_archival": {
                "content": self._state.l2_archival.narrative,
                "origin": "Inline compression (L1 -> L2)",
            },
            "l1_5_entities": [],
            "l1_working_messages": len(self._state.l1_working.messages),
            "l3_semantic": {
                "chunk_count": self._state.l3_semantic.chunk_count,
                "last_indexed_at": (
                    self._state.l3_semantic.last_indexed_at.isoformat()
                    if self._state.l3_semantic.last_indexed_at
                    else None
                ),
                "origin": "L3 vector indexing (L1 evictions)",
                "in_prompt": len(self._last_l3_retrieval) > 0,
                "retrieved_chunks": self._last_l3_retrieval,
            },
        }

        journal_path = Path(self._config.journal_path)
        journal_history: dict[str, dict[str, str]] = {}
        if journal_path.exists():
            with open(journal_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        key, operation, timestamp = _extract_entity_event(record)
                        if key:
                            payload = (
                                record.get("payload") or record.get("data") or record
                            )
                            strategy = (
                                payload.get("strategy", "deterministic")
                                if isinstance(payload, dict)
                                else "deterministic"
                            )
                            journal_history[key] = {
                                "operation": operation,
                                "timestamp": timestamp,
                                "strategy": strategy,
                            }
                    except json.JSONDecodeError:
                        continue

        confidence_map = {
            "deterministic": "100% (Deterministic)",
            "salience_heuristic": "90% (Salience Heuristic)",
            "pinned": "100% (Pinned)",
            "llm_synthesis": "85% (LLM Synthesized)",
        }

        for key, value in self._state.l1_5_entities.entities.items():
            history = journal_history.get(
                key,
                {
                    "operation": "unknown",
                    "timestamp": "unknown",
                    "strategy": "deterministic",
                },
            )
            strategy = history.get("strategy", "deterministic")
            trace["l1_5_entities"].append(
                {
                    "prompt_component": "[ENTITY_LEDGER_L1_5]",
                    "entity_key": key,
                    "entity_value": value,
                    "origin": (
                        f"Anchored via explicit tracking engine "
                        f"(Operation: {history['operation']}) [Strategy: {strategy}]"
                    ),
                    "timestamp": history["timestamp"],
                    "confidence": confidence_map.get(
                        strategy, "100% (Deterministic)"
                    ),
                }
            )

        return trace

    def search_semantic_archive(
        self, query: str, top_k: int = 5
    ) -> list[SemanticChunkResult]:
        if not self._l3_indexer:
            return []
        return run_coro_once(
            self._l3_indexer.search(self._config.session_id, query, top_k)
        )

    def l3_chunk_count(self) -> int:
        if not self._l3_indexer:
            return self._state.l3_semantic.chunk_count
        return run_coro_once(
            self._l3_indexer.count(self._config.session_id)
        )

    @property
    def state(self) -> MemoryState:
        return self._state

    def get_stats(self) -> dict[str, Any]:
        return {
            "l0_tokens": self._state.l0_system.token_count,
            "l1_tokens": self._state.l1_working.token_count,
            "l1_message_count": len(self._state.l1_working.messages),
            "l1_5_entity_count": len(self._state.l1_5_entities.entities),
            "l2_tokens": self._monitor.count_text(self._state.l2_archival.narrative),
            "l3_chunk_count": self._state.l3_semantic.chunk_count,
            "l3_enabled": self._l3_indexer is not None,
            "l3_retrieved_chunk_count": len(self._last_l3_retrieval),
            "dte": {
                **self._state.dte.model_dump(),
                "mode": self._config.compression_mode,
                "background_spend_ratio_actual": (
                    self._state.dte.background_llm_input_tokens
                    / max(self._state.dte.main_prompt_tokens, 1)
                ),
                "observation_cache_entries": len(self._observation_cache),
            },
            "compression": {
                "cycles": self._compression_cycles,
                "failures": self._compression_failures,
                "mode": "inline_sync",
            },
        }

    def health_check(self) -> dict[str, Any]:
        report: dict[str, Any] = {"status": "healthy", "checks": {}}

        if self._config.soft_limit_tokens >= self._config.hard_limit_tokens:
            report["status"] = "unhealthy"
            raise ValueError(
                f"Configuration Error: soft_limit_tokens ({self._config.soft_limit_tokens}) "
                f"must be strictly less than hard_limit_tokens ({self._config.hard_limit_tokens})."
            )
        report["checks"]["configuration"] = "OK"
        report["checks"]["runtime"] = "sync_inline"
        report["checks"]["compression_mode"] = "blocking"

        if isinstance(self._compressor, SyncOllamaCompressor):
            ollama_cfg = self._config.ollama
            report["checks"]["backend"] = "ollama"
            report["checks"]["model"] = ollama_cfg.model if ollama_cfg else "unknown"
            try:
                self._compressor.ping()
                report["checks"]["backend_reachable"] = "OK"
            except Exception as exc:
                report["status"] = "degraded"
                report["checks"]["backend_reachable"] = f"UNREACHABLE: {exc}"
        elif isinstance(self._compressor, SyncCloudCompressor):
            cloud_cfg = self._config.cloud
            report["checks"]["backend"] = "cloud"
            report["checks"]["provider"] = (
                cloud_cfg.provider.value if cloud_cfg else "unknown"
            )
            report["checks"]["model"] = cloud_cfg.model if cloud_cfg else "unknown"
            try:
                self._compressor.ping()
                report["checks"]["backend_reachable"] = "CONFIGURED"
            except Exception as exc:
                report["status"] = "unhealthy"
                report["checks"]["backend_reachable"] = f"MISSING: {exc}"

        if self._config.enable_l3_semantic_storage:
            report["checks"]["l3_semantic_storage"] = "ENABLED"
            report["checks"]["l3_embedding_backend"] = self._config.embedding_backend
            report["checks"]["l3_chunk_count"] = self._state.l3_semantic.chunk_count
            report["checks"]["l3_prompt_retrieval"] = (
                "ENABLED" if self._config.enable_l3_prompt_retrieval else "DISABLED"
            )
        else:
            report["checks"]["l3_semantic_storage"] = "DISABLED"
            report["checks"]["l3_prompt_retrieval"] = "DISABLED"

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        if not self._config.storage_adapter:
            return
        loaded = run_coro_once(
            self._config.storage_adapter.load_state(self._config.session_id)
        )
        if loaded is None:
            return
        self._state.l0_system = loaded.l0_system
        self._state.l1_working = loaded.l1_working
        self._state.l3_semantic = loaded.l3_semantic
        self._state.dte = loaded.dte
        self._state.dte.consolidation_queued = False
        if not self._config.pool_id:
            self._state.l1_5_entities = loaded.l1_5_entities
            self._state.l2_archival = loaded.l2_archival

    def _sync_pool_state_from_storage(self) -> None:
        adapter = self._config.storage_adapter
        pool_id = self._config.pool_id
        if not adapter or not pool_id:
            return

        pool_state = run_coro_once(adapter.load_pool_state(pool_id))
        if pool_state is None:
            return

        pool_entities, pool_archive = pool_state
        for key, history in pool_entities.entities.items():
            for value in history:
                self._state.l1_5_entities.upsert({key: value})

        pool_narrative = pool_archive.narrative.strip()
        if pool_narrative:
            local_narrative = self._state.l2_archival.narrative.strip()
            if not local_narrative:
                self._state.l2_archival.narrative = pool_narrative
            elif pool_narrative not in local_narrative:
                self._state.l2_archival.append_narrative(pool_narrative)

    def _sync_fold_to_pool(
        self, narrative: str, entity_keys: tuple[str, ...]
    ) -> None:
        adapter = self._config.storage_adapter
        pool_id = self._config.pool_id
        if not adapter or not pool_id:
            return

        pool_state = run_coro_once(adapter.load_pool_state(pool_id))
        if pool_state is None:
            shared_entities = EntityLedger()
            shared_archive = ArchivalMemory()
        else:
            shared_entities, shared_archive = pool_state

        for key in entity_keys:
            value = self._state.l1_5_entities.get_latest(key)
            if value is not None:
                shared_entities.upsert({key: value})
        shared_archive.append_narrative(
            f"[origin:{self._config.session_id}] {narrative}"
        )
        run_coro_once(
            adapter.save_pool_state(pool_id, shared_entities, shared_archive)
        )

    def _compact_pool_folds(self) -> None:
        adapter = self._config.storage_adapter
        pool_id = self._config.pool_id
        if not adapter or not pool_id:
            return
        try:
            pool_state = run_coro_once(adapter.load_pool_state(pool_id))
            if pool_state is None:
                return
            shared_entities, shared_archive = pool_state
            shared_archive.narrative = remove_fold_lines(shared_archive.narrative)
            run_coro_once(
                adapter.save_pool_state(pool_id, shared_entities, shared_archive)
            )
        except Exception as exc:
            logger.warning(
                "SyncContextManager: failed to compact shared fold records (%s).",
                exc,
                exc_info=True,
            )

    def _scan_message_entities(self, content: str) -> None:
        extraction = self._ner_pipeline.extract_with_metadata(content)
        if not extraction.entities:
            return
        token = active_strategy_context.set(extraction.strategies)
        try:
            self._state.l1_5_entities.upsert(extraction.entities)
        finally:
            active_strategy_context.reset(token)

    def _index_l3(self, state: MemoryState, messages_text: str, cycle_id: str) -> int:
        if not self._l3_indexer or not messages_text.strip():
            return 0
        try:
            chunks = run_coro_once(
                self._l3_indexer.index(
                    self._config.session_id, messages_text, state
                )
            )
        except Exception as exc:
            logger.warning(
                "SyncContextManager: L3 indexing failed (%s).", exc, exc_info=True
            )
            return 0

        if chunks and self._config.storage_adapter:
            try:
                run_coro_once(
                    self._config.storage_adapter.save_state(
                        self._config.session_id, state
                    )
                )
            except Exception as exc:
                logger.warning(
                    "SyncContextManager: failed to persist L3 metadata (%s).",
                    exc,
                    exc_info=True,
                )
        return chunks

    def _maybe_consolidate(self, intent_plan: PromptIntentPlan) -> None:
        if (
            self._config.compression_mode != "dte"
            or not self._config.consolidation_on_idle
            or not self._config.enable_sync_consolidation
            or self._state.dte.consolidation_queued
        ):
            return

        debt = self._state.dte.narrative_debt_tokens
        if debt <= 0 or (
            debt < self._config.narrative_debt_trigger_tokens
            and not intent_plan.prefers_narrative
        ):
            return

        folds = fold_lines(self._state.l2_archival.narrative)
        if not folds:
            self._state.dte.narrative_debt_tokens = 0
            return

        source = "\n".join(folds)
        residual = source
        if self._config.enable_novelty_filter:
            novelty = residualize(
                source,
                self._state.l1_5_entities,
                self._state.l2_archival.narrative,
                count_text=self._monitor.count_text,
            )
            residual = novelty.residual
            if (
                not residual
                or novelty.residual_ratio < self._config.novelty_min_residual
            ):
                self._state.l2_archival.narrative = remove_fold_lines(
                    self._state.l2_archival.narrative
                )
                self._state.dte.narrative_debt_tokens = 0
                self._state.dte.folds_since_narrative = 0
                self._state.dte.novelty_skips += 1
                return

        guideline = self._config.compression_guideline or (
            "Consolidate these structured fold outcomes into one dense, "
            "causal narrative. Do not repeat exact identifiers already protected "
            "by the entity ledger."
        )
        content = f"{guideline}\n\n{residual}"
        message = Message(role="system", content=content)
        message.token_count = self._monitor.count_message(message)

        allowance = int(
            self._state.dte.main_prompt_tokens
            * self._config.background_spend_ratio
        )
        projected = (
            self._state.dte.background_llm_input_tokens + message.token_count
        )
        if projected > allowance:
            return

        self._state.dte.consolidation_queued = True
        try:
            outcome = run_compression_cycle_sync(
                CompressionCycleInput(
                    messages=[message],
                    state=self._state,
                    cycle_id=f"consolidate-{uuid.uuid4()}",
                ),
                self._compressor,
                self._engine,
                run_async=run_coro_once,
            )
            self._state.dte.background_llm_input_tokens += message.token_count
            if outcome.success:
                self._state.l2_archival.narrative = remove_fold_lines(
                    self._state.l2_archival.narrative
                )
                self._compact_pool_folds()
                self._state.dte.narrative_debt_tokens = 0
                self._state.dte.folds_since_narrative = 0
                self._state.dte.consolidation_cycles += 1
                self._compression_cycles += 1
            else:
                self._state.l2_archival.narrative = "\n".join(
                    line
                    for line in self._state.l2_archival.narrative.splitlines()
                    if not line.startswith("[COMPRESSION UNAVAILABLE:")
                ).strip()
                self._compression_failures += 1
        finally:
            self._state.dte.consolidation_queued = False

    def _trigger_compression(self) -> None:
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        if not chunk:
            self._monitor.release_compression_lock()
            return

        cycle_id = str(uuid.uuid4())
        if self._config.compression_mode == "dte":
            messages_text = messages_to_text(chunk)
            l3_chunks = self._index_l3(self._state, messages_text, cycle_id)
            fold = create_fold_unit(
                chunk,
                self._state,
                self._ner_pipeline,
                cycle_id=cycle_id,
                l3_chunks=l3_chunks,
                enable_ner=self._config.enable_deterministic_ner,
            )
            self._sync_fold_to_pool(fold.stub, fold.entity_keys)
            self._monitor.release_compression_lock()
            logger.info(
                "SyncContextManager: DTE-folded %d messages (cycle=%s).",
                len(chunk),
                cycle_id,
            )
            return

        try:
            outcome = run_compression_cycle_sync(
                CompressionCycleInput(
                    messages=chunk,
                    state=self._state,
                    cycle_id=cycle_id,
                ),
                self._compressor,
                self._engine,
                index_l3=self._index_l3,
                run_async=run_coro_once,
            )
            self._compression_cycles += 1
            if not outcome.success:
                self._compression_failures += 1

            logger.info(
                "SyncContextManager: inline compression completed for %d messages "
                "(cycle=%s, success=%s).",
                len(chunk),
                cycle_id,
                outcome.success,
            )
        finally:
            self._monitor.release_compression_lock()

    def _force_truncate(self) -> None:
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        if not chunk:
            return

        if self._l3_indexer:
            messages_text = messages_to_text(chunk)
            cycle_id = f"hard-truncate-{uuid.uuid4()}"
            self._index_l3(self._state, messages_text, cycle_id)

        note = (
            f"[HARD TRUNCATION: {len(chunk)} messages dropped because the "
            f"inline compression path has not yet caught up.]"
        )
        self._state.l2_archival.append_narrative(note)
        logger.warning("Hard truncation: dropped %d messages from L1.", len(chunk))
        self._monitor.release_compression_lock()
