"""
middleware.py — ContextManager: the primary public API for Sawtooth-Memory.

Drop this between your agent loop and your LLM API call.

Quick start:
    from sawtooth_memory import ContextManager, ContextManagerConfig

    config = ContextManagerConfig(soft_limit_tokens=3000)

    async with ContextManager("You are a data analysis agent.", config) as cm:
        await cm.add_message("user", "Analyse Q3 revenue.")
        await cm.add_message("assistant", "Connecting to the database...")

        messages = await cm.build_prompt()
        # response = await openai_client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=messages,
        # )
"""

import logging
from pathlib import Path
from typing import Any, Literal, Optional

from .compressor import CloudCompressor, OllamaCompressor
from .config import ContextManagerConfig
from .embeddings.factory import create_embedding_provider
from .events.bus import EventBus, get_event_bus
from .events.types import (
    CompressionCycleStartEvent,
    EntityAnchoredEvent,
    L1EvictionEvent,
)
from .exceptions import TokenLimitExceededError
from .journal import AsyncCompressionJournal
from .l3_indexer import SemanticIndexer
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
from .storage.semantic import SemanticChunkResult
from .worker import CompressionTask, CompressionWorker, _messages_to_text

logger = logging.getLogger(__name__)


def _extract_entity_event(record: dict) -> tuple:
    """Extract (entity_key, operation, timestamp) safely from nested OOP JSON records."""
    entity_channel = EntityAnchoredEvent.event_type

    # v2 - fields nested under "payload" FIRST
    payload = record.get("payload")
    if isinstance(payload, dict):
        event_type = str(record.get("event_type", "")).lower()
        if "entity_key" in payload or "entity_anchor" in event_type:
            return (
                payload.get("entity_key"),
                payload.get("operation", "unknown"),
                payload.get("timestamp", record.get("timestamp", "unknown")),
            )

    # v3 - fields nested under "data" SECOND
    data = record.get("data")
    if isinstance(data, dict):
        if record.get("channel") == entity_channel or "entity_key" in data:
            return (
                data.get("entity_key"),
                data.get("operation", "unknown"),
                data.get("timestamp", record.get("timestamp", "unknown")),
            )

    # v1 - fields live at the record root LAST
    if record.get("channel") == entity_channel or "entity_key" in record:
        return (
            record.get("entity_key"),
            record.get("operation", "unknown"),
            record.get("timestamp", "unknown"),
        )

    return (None, "unknown", "unknown")


class ContextManager:
    def __init__(
        self,
        system_prompt: str,
        config: ContextManagerConfig | None = None,
        *,
        enable_events: bool = True,
        journal_path: Optional[Path] = None,
    ) -> None:
        self._config = config or ContextManagerConfig()
        self._enable_events = enable_events

        # 1. Initialize Event Bus and Journal FIRST
        self._event_bus: Optional[EventBus] = None
        self._journal: Optional[AsyncCompressionJournal] = None

        if self._enable_events:
            self._event_bus = get_event_bus()

            # Localize the journal instance to this ContextManager
            j_path = journal_path or Path(self._config.journal_path)
            self._journal_path = j_path
            self._journal = AsyncCompressionJournal(j_path)

            from .events.handlers import make_journal_handler

            # 1. Bind the strict Compression Journal handler
            handler = make_journal_handler(self._journal)
            self._event_bus.subscribe("compression.cycle_complete", handler)  # type: ignore[arg-type]

            # 2. Bind a dedicated, lightweight handler for Entity Anchoring events
            # This safely writes the OOP schema to the JSONL file without crashing on strict dataclass fields.
            import json

            import aiofiles

            async def entity_journal_handler(event: EntityAnchoredEvent) -> None:
                record = {
                    "event_type": "entity_anchored",
                    "payload": {
                        "entity_key": event.entity_key,
                        "entity_value": event.entity_value,
                        "operation": getattr(event, "operation", "unknown"),
                        "strategy": getattr(event, "strategy", "deterministic"),
                    },
                    "timestamp": event.timestamp.isoformat(),
                }
                try:
                    async with aiofiles.open(j_path, "a", encoding="utf-8") as f:
                        await f.write(json.dumps(record) + "\n")
                except Exception as e:
                    logger.error(f"Failed to write entity event to journal: {e}")

            self._event_bus.subscribe(EntityAnchoredEvent.event_type, entity_journal_handler)  # type: ignore[arg-type]

        # 2. Token monitor now receives the initialized event_bus
        self._monitor = TokenMonitor(
            model=self._config.tokenizer_model,
            soft_limit=self._config.soft_limit_tokens,
            hard_limit=self._config.hard_limit_tokens,
            # pass the batching threshold if it exists in the config
            max_unsummarized_turns=self._config.max_unsummarized_turns,
            event_bus=self._event_bus,
        )

        sp_tokens = self._monitor.count_text(system_prompt)
        self._state = MemoryState(
            l0_system=SystemPrompt(content=system_prompt, token_count=sp_tokens),
            l1_working=WorkingMemory(),
            l1_5_entities=EntityLedger(),
            l2_archival=ArchivalMemory(),
        )

        # 3. Bind the Entity Ledger telemetry callback
        if self._enable_events and self._event_bus:

            def handle_ledger_mutation(
                key: str, value: str, op: Literal["insert", "update", "delete"]
            ):
                if self._event_bus:
                    from .ner import active_strategy_context

                    strategies = active_strategy_context.get()

                    # If the key isn't explicitly mapped by the worker, it's a manual injection
                    # from the main thread, which is inherently deterministic.
                    strategy = strategies.get(key, "deterministic")

                    event = EntityAnchoredEvent(
                        entity_key=key, entity_value=value, operation=op
                    )
                    # Dynamically attach strategy without altering strict Pydantic core models
                    setattr(event, "strategy", strategy)
                    self._event_bus.emit_nowait(event)

            self._state.l1_5_entities.set_event_callback(handle_ledger_mutation)

        # 4. Compression backend & Worker
        self._compressor: CloudCompressor | OllamaCompressor
        if self._config.cloud:
            # No custom logic needed here! The model config is already perfectly updated.
            self._compressor = CloudCompressor(self._config.cloud)
        else:
            from .config import OllamaConfig

            # This cleanly handles both pre-configured or default fallback states
            ollama_cfg = (
                self._config.ollama
                if self._config.ollama is not None
                else OllamaConfig()
            )
            self._compressor = OllamaCompressor(ollama_cfg)

        self._l3_indexer: SemanticIndexer | None = None
        self._embedder: Any = None
        if self._config.enable_l3_semantic_storage:
            # Config validator guarantees semantic storage when L3 is enabled.
            embedder = create_embedding_provider(
                self._config.embedding_backend,  # type: ignore[arg-type]
                model=self._config.embedding_model,
                dimension=self._config.embedding_dimension,
            )
            self._embedder = embedder
            self._l3_indexer = SemanticIndexer(
                storage=self._config.storage_adapter,
                embedder=embedder,
                chunk_max_chars=self._config.l3_chunk_max_chars,
            )

        self._worker = CompressionWorker(
            compressor=self._compressor,
            fallback_truncate=self._config.fallback_truncate,
            event_bus=self._event_bus,
            enable_deterministic_ner=self._config.enable_deterministic_ner,
            custom_ner_patterns=self._config.custom_ner_patterns,
            enable_salience_extractor=self._config.enable_salience_extractor,
            salience_threshold=self._config.salience_threshold,
            salience_max_entities=self._config.salience_max_entities,
            enable_entity_verifier=self._config.enable_entity_verifier,
            storage_adapter=self._config.storage_adapter,
            pool_id=self._config.pool_id,
            session_id=self._config.session_id,
            l3_indexer=self._l3_indexer,
            embedding_backend=self._config.embedding_backend,
            embedding_model=self._config.embedding_model,
        )

        self._last_l3_retrieval: list[dict[str, Any]] = []

        logger.debug(
            f"ContextManager initialised. "
            f"soft_limit={self._config.soft_limit_tokens}, "
            f"hard_limit={self._config.hard_limit_tokens}, "
            f"chunk_size={self._config.chunk_size}, "
            f"events_enabled={self._enable_events}"
        )

    async def start(self) -> None:
        if self._config.storage_adapter:
            loaded = await self._config.storage_adapter.load_state(
                self._config.session_id
            )
            if loaded is not None:
                self._state.l0_system = loaded.l0_system
                self._state.l1_working = loaded.l1_working
                self._state.l3_semantic = loaded.l3_semantic
                if not self._config.pool_id:
                    self._state.l1_5_entities = loaded.l1_5_entities
                    self._state.l2_archival = loaded.l2_archival

        if self._config.pool_id:
            await self._sync_pool_state_from_storage()

        if self._enable_events and self._journal:
            await self._journal.start()
        await self._worker.start()

    async def stop(self) -> None:
        await self._worker.stop()
        if self._embedder is not None and hasattr(self._embedder, "close"):
            await self._embedder.close()
        if self._enable_events and self._journal:
            await self._journal.stop()  # Stops just this agent's journal instance
        if self._enable_events and self._event_bus:
            await (
                self._event_bus.drain()
            )  # Flushes any pending background telemetry tasks

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

        When events are enabled, this method emits L1 eviction events
        and compression start events.
        """
        msg = Message(role=role, content=content)
        msg.token_count = self._monitor.count_message(msg)
        self._state.l1_working.append(msg)

        if (
            self._config.enable_ingest_entity_scan
            and self._config.enable_deterministic_ner
        ):
            await self._scan_message_entities(content)

        logger.debug(
            f"add_message: role={role}, tokens={msg.token_count}, "
            f"l1_total={self._state.l1_working.token_count}"
        )

        # Hard limit check (immediate action, no background)
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
            await self._force_truncate()

        # Soft limit & Turn count batching check (Debounced) → trigger async compression
        elif self._monitor.should_trigger_compression(self._state):
            await self._trigger_compression()

        if self._config.storage_adapter:
            await self._config.storage_adapter.save_state(
                self._config.session_id, self._state
            )

    async def pin_entity(self, key: str, value: str) -> None:
        """
        Explicitly pin a critical entity into the L1.5 ledger.

        Pinned entities are protected through compression with highest priority
        and tagged with strategy ``pinned`` in telemetry.
        """
        from .ner import active_strategy_context

        token = active_strategy_context.set({key: "pinned"})
        try:
            self._state.l1_5_entities.upsert({key: value})
        finally:
            active_strategy_context.reset(token)

        if self._config.storage_adapter:
            await self._config.storage_adapter.save_state(
                self._config.session_id, self._state
            )

    async def _scan_message_entities(self, content: str) -> None:
        """Lightweight ingest-time entity scan for the live L1 window."""
        from .ner import active_strategy_context

        extraction = self._worker.ner_pipeline.extract_with_metadata(content)
        if not extraction.entities:
            return

        token = active_strategy_context.set(extraction.strategies)
        try:
            self._state.l1_5_entities.upsert(extraction.entities)
        finally:
            active_strategy_context.reset(token)

    async def _sync_pool_state_from_storage(self) -> None:
        adapter = self._config.storage_adapter
        pool_id = self._config.pool_id
        if not adapter or not pool_id:
            return

        pool_state = await adapter.load_pool_state(pool_id)
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

    async def _retrieve_l3_chunks(self, query: str) -> str:
        """
        Retrieve chunks from L3 semantic storage and format them into a text block,
        respecting the configured token budget.
        """
        self._last_l3_retrieval = []
        if not self._l3_indexer:
            return ""

        results = await self._l3_indexer.search(
            self._config.session_id, query, top_k=self._config.l3_retrieval_top_k
        )
        if not results:
            return ""

        block_lines = []
        current_tokens = 0
        budget = self._config.l3_retrieval_max_tokens

        for i, res in enumerate(results, 1):
            line = f"{i}. {res.text}"
            line_tokens = self._monitor.count_text(line)
            if current_tokens + line_tokens > budget and current_tokens > 0:
                break

            block_lines.append(line)
            current_tokens += line_tokens
            self._last_l3_retrieval.append({
                "text": res.text,
                "similarity": res.similarity,
                "origin": "L3 Semantic Retrieval",
            })

        if not block_lines:
            return ""

        return "\n".join(block_lines)

    async def build_prompt(self, *, retrieval_query: str | None = None) -> list[dict[str, str]]:
        """
        Compile all memory tiers into an OpenAI-compatible messages list.

        Returns a list of {"role": "...", "content": "..."} dicts, ready
        to pass directly to openai.chat.completions.create() or equivalent.

        Structure of the injected system message:
            [SYSTEM_L0]
            <system prompt>

            [ARCHIVE_L2]          (omitted if empty)
            <compressed history narrative>

            [ARCHIVE_L3]          (omitted if empty or disabled)
            <semantic retrieval hits>

            [ENTITY_LEDGER_L1_5]  (omitted if empty)
            <json key-value pairs>

        Followed by raw Working Memory (L1) messages.
        """
        await self._sync_pool_state_from_storage()
        state = self._state
        system_parts: list[str] = []

        system_parts.append(f"[SYSTEM_L0]\n{state.l0_system.content}")

        if state.l2_archival.narrative.strip():
            system_parts.append(f"[ARCHIVE_L2]\n{state.l2_archival.narrative.strip()}")

        # L3 Semantic Retrieval
        self._last_l3_retrieval = []
        if self._config.enable_l3_prompt_retrieval and self._l3_indexer:
            query = retrieval_query
            if not query:
                # Scan L1 messages newest-first for the last user message
                for msg in reversed(state.l1_working.messages):
                    if msg.role == "user":
                        query = msg.content
                        break

            if query:
                l3_block = await self._retrieve_l3_chunks(query)
                if l3_block:
                    system_parts.append(f"[ARCHIVE_L3]\n{l3_block}")

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

    async def search_semantic_archive(
        self, query: str, top_k: int = 5
    ) -> list[SemanticChunkResult]:
        """
        Retrieve L3 semantic chunks similar to *query*.

        This is the storage-layer retrieval API. Results are **not**
        injected into :meth:`build_prompt` until a future release wires
        RAG retrieval into the prompt compiler.
        """
        if not self._l3_indexer:
            return []
        return await self._l3_indexer.search(self._config.session_id, query, top_k)

    async def l3_chunk_count(self) -> int:
        """Return the number of indexed L3 semantic chunks for this session."""
        if not self._l3_indexer:
            return self._state.l3_semantic.chunk_count
        return await self._l3_indexer.count(self._config.session_id)

    def explain_prompt(self) -> dict:
        """
        Deliverable 2.3: Recall Explainability Traces
        Returns a structured developer audit trail explaining exactly why
        specific elements (like L1.5 entities) are present in the active prompt.
        """
        import json

        trace: dict = {
            "l0_system": {
                "content": self._state.l0_system.content,
                "origin": "Hardcoded System Initialization",
            },
            "l2_archival": {
                "content": self._state.l2_archival.narrative,
                "origin": "Background Ollama Compression (L1 -> L2)",
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
                "origin": "Background L3 vector indexing (L1 evictions → pgvector)",
                "in_prompt": len(self._last_l3_retrieval) > 0,
                "retrieved_chunks": self._last_l3_retrieval,
            },
        }

        # 1. Rebuild the historical lineage from the JSONL journal safely
        journal_history: dict = {}

        # Use getattr to safely resolve the path without triggering linter warnings
        journal_path = getattr(self, "_journal_path", None)
        if journal_path is None and getattr(self, "_journal", None) is not None:
            journal_path = getattr(
                self._journal, "path", getattr(self._journal, "_path", None)
            )

        if journal_path and Path(journal_path).exists():
            with open(Path(journal_path), "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        key, operation, timestamp = _extract_entity_event(record)
                        if key:
                            # Safely extract strategy - fallback to deterministic if field is missing
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

        # 2. Map the active L1.5 ledger against the historical lineage
        for key, value in self._state.l1_5_entities.entities.items():
            # If the entry is missing from the log file (e.g., manual developer injection),
            # it is inherently 100% deterministic.
            history = journal_history.get(
                key,
                {
                    "operation": "unknown",
                    "timestamp": "unknown",
                    "strategy": "deterministic",
                },
            )
            strategy = history.get("strategy", "deterministic")
            confidence_map = {
                "deterministic": "100% (Deterministic)",
                "salience_heuristic": "90% (Salience Heuristic)",
                "pinned": "100% (Pinned)",
                "llm_synthesis": "85% (LLM Synthesized)",
            }
            trace["l1_5_entities"].append(
                {
                    "prompt_component": "[ENTITY_LEDGER_L1_5]",
                    "entity_key": key,
                    "entity_value": value,
                    # FIX: Inject the operation back into the origin string so the test catches it
                    "origin": f"Anchored via explicit tracking engine (Operation: {history['operation']}) [Strategy: {strategy}]",
                    "timestamp": history["timestamp"],
                    "confidence": confidence_map.get(
                        strategy, "100% (Deterministic)"
                    ),
                }
            )

        return trace

    # ------------------------------------------------------------------
    # Internal compression triggers
    # ------------------------------------------------------------------

    async def _trigger_compression(self) -> None:
        """
        Non-blocking: slice the oldest chunk and hand it off to the worker.
        The main thread continues running immediately.

        If events are enabled, emits a CompressionCycleStartEvent and
        (later in the worker) the completion/failure events.
        """
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        if not chunk:
            return

        # Generate a unique cycle ID for this compression run
        import uuid

        cycle_id = str(uuid.uuid4())

        # Emit start event if bus exists
        if self._event_bus:
            await self._event_bus.emit(
                CompressionCycleStartEvent(
                    cycle_id=cycle_id,
                    current_l1_tokens=self._state.l1_working.token_count,
                    chunk_size=self._config.chunk_size,
                )
            )

        # Emit L1 eviction event (this compression will evict these messages)
        if self._event_bus:
            evicted_tokens = sum(m.token_count for m in chunk)
            await self._event_bus.emit(
                L1EvictionEvent(
                    tokens_evicted=evicted_tokens,
                    messages_evicted=len(chunk),
                    tokens_remaining_l1=self._state.l1_working.token_count,
                    evicted_message_ids=[m.id for m in chunk],
                    trigger="soft_limit_exceeded",
                    cycle_id=cycle_id,  # link to compression cycle
                )
            )

        # Create task with cycle_id so worker can correlate events
        task = CompressionTask(
            messages=chunk,
            state=self._state,
            cycle_id=cycle_id,
        )
        self._worker.enqueue(task)

        logger.info(
            f"Compression triggered: offloaded {len(chunk)} messages to worker. "
            f"L1 remaining: {self._state.l1_working.token_count} tokens, "
            f"cycle_id={cycle_id}"
        )

    async def _force_truncate(self) -> None:
        """
        Hard-limit fallback: discard the oldest messages immediately on
        the main thread without waiting for Ollama/Cloud.

        When L3 is enabled, evicted text is still indexed into semantic
        storage so retrieval remains possible after truncation.

        Note: This does NOT emit compression cycle events because it's a
        fallback path. The journal remains unaffected.
        """
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        if not chunk:
            return

        if self._l3_indexer:
            import uuid

            messages_text = _messages_to_text(chunk)
            cycle_id = f"hard-truncate-{uuid.uuid4()}"
            await self._worker.index_l3_semantic(
                self._state, messages_text, cycle_id
            )

        note = (
            f"[HARD TRUNCATION: {len(chunk)} messages dropped because the "
            f"compression worker has not yet caught up.]"
        )
        self._state.l2_archival.append_narrative(note)
        logger.warning(f"Hard truncation: dropped {len(chunk)} messages from L1.")

        # The soft-limit path may have queued compression before we got here.
        # Release the debounce lock so L1 can trigger a fresh cycle once truncated.
        self._monitor.release_compression_lock()

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
            "l2_tokens": self._monitor.count_text(self._state.l2_archival.narrative),
            "l3_chunk_count": self._state.l3_semantic.chunk_count,
            "l3_enabled": self._l3_indexer is not None,
            "l3_retrieved_chunk_count": len(self._last_l3_retrieval),
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

    async def health_check(self) -> dict[str, Any]:
        """
        Verifies runtime configurations and basic initialization readiness.
        Returns a diagnostic report dictionary. Raises ValueError on broken configurations.
        """
        report: dict[str, Any] = {"status": "healthy", "checks": {}}

        # 1. Validate Token Configurations
        if self._config.soft_limit_tokens >= self._config.hard_limit_tokens:
            report["status"] = "unhealthy"
            raise ValueError(
                f"Configuration Error: soft_limit_tokens ({self._config.soft_limit_tokens}) "
                f"must be strictly less than hard_limit_tokens ({self._config.hard_limit_tokens})."
            )
        report["checks"]["configuration"] = "OK"

        # 2. Verify Background Worker State
        if getattr(self, "_worker", None) and self._worker._running:
            report["checks"]["worker_status"] = "RUNNING"
        else:
            report["checks"]["worker_status"] = "STOPPED"

        # 3. (Optional) Check event bus and journal health
        if self._enable_events:
            report["checks"]["events"] = "ENABLED"
            if self._journal:
                report["checks"]["journal_path"] = str(self._journal.path)
            else:
                report["checks"]["journal"] = "NOT_INITIALIZED"
        else:
            report["checks"]["events"] = "DISABLED"

        # 4. Verify compression backend routing and reachability
        from .compressor import CloudCompressor, OllamaCompressor

        if isinstance(self._compressor, OllamaCompressor):
            ollama_cfg = self._config.ollama
            report["checks"]["backend"] = "ollama"
            report["checks"]["model"] = ollama_cfg.model if ollama_cfg else "unknown"
            try:
                await self._compressor.ping()
                report["checks"]["backend_reachable"] = "OK"
            except Exception as exc:
                report["status"] = "degraded"
                report["checks"]["backend_reachable"] = f"UNREACHABLE: {exc}"
        elif isinstance(self._compressor, CloudCompressor):
            cloud_cfg = self._config.cloud
            report["checks"]["backend"] = "cloud"
            report["checks"]["provider"] = (
                cloud_cfg.provider.value if cloud_cfg else "unknown"
            )
            report["checks"]["model"] = cloud_cfg.model if cloud_cfg else "unknown"
            try:
                await self._compressor.ping()
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
