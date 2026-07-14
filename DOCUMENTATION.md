# Sawtooth Memory: Official Documentation

Sawtooth Memory is a high-performance, asynchronous hierarchical memory framework designed for Large Language Model (LLM) agents. It solves the primary bottlenecks of standard conversational memory: main-thread latency and context degradation (the "Lost in the Middle" hallucination effect).

This documentation provides a deep dive into the architecture, configuration, and advanced usage of the framework.

---

## Table of Contents
1. [Core Architecture](#1-core-architecture)
2. [The Hierarchical Stack](#2-the-hierarchical-stack)
3. [Component Deep Dive](#3-component-deep-dive)
4. [Configuration Reference](#4-configuration-reference)
5. [API & Usage Guide](#5-api--usage-guide)
6. [Observability & Telemetry](#6-observability--telemetry)
7. [LangGraph Integration](#7-langgraph-integration)
8. [Persistence & Journaling](#8-persistence--journaling)
9. [Distributed Storage & L3 Semantic Archival](#9-distributed-storage--l3-semantic-archival)
10. [API Reference](#10-api-reference)
11. [Examples](#11-examples)

---

## 1. Core Architecture

The fundamental innovation of Sawtooth Memory is the decoupling of message ingestion from context compression.

In traditional architectures, summarizing conversational history is a synchronous operation. The application must wait for the LLM to process the history before it can respond to the user. Sawtooth utilizes an asynchronous event loop and a dedicated background worker to eliminate this bottleneck.

```text
  Main Application Thread                       Background Async Event Loop
  ───────────────────────                       ───────────────────────────

  1. User Input Received
            │
            ▼
  2. cm.add_message() ───────────────────────┐  (Task Dispatched to Queue)
            │                                │
            ├─ Ingest Entity Scan (optional)   │  Regex + salience → L1.5
            │                                │
  (Returns in <0.01s)                        ▼
            │                         3. Worker awakens
            ▼                                │
  4. cm.build_prompt()                       ▼
            │                         4. Check Token Limits (Soft/Hard)
  (Retrieves active payload)                 │
            │                                ▼
            ▼                         5. Local Entity Guard (regex + salience)
  5. Send to Main Agent                      │
                                             ▼
                                      6. Call Compression LLM (with protection manifest)
                                             │
                                             ▼
                                      7. Secure merge + post-merge verifier
                                             │
                                             ▼
                                      8. Update L1.5 Entities & L2 Summary
                                             │
                                             ▼
                                      9. Commit to Disk Journal

```

---

## 2. The Hierarchical Stack

When `build_prompt()` is called, Sawtooth constructs the prompt using a strict hierarchical order. This guarantees that critical system instructions and explicit factual data are placed closest to the generation vectors, mitigating attention degradation.

```text
┌─────────────────────────────────────────────────────────────┐
│ Final Context Payload (Sent to Agent)                       │
├─────────────────────────────────────────────────────────────┤
│ L0: System Instructions                                     │
│   "You are an AI assistant. Format output as JSON."         │
├─────────────────────────────────────────────────────────────┤
│ L2: Archival Summary                                        │
│   "User and AI discussed Python optimization techniques."   │
├─────────────────────────────────────────────────────────────┤
│ L3: Semantic Retrieval Hits                                 │
│   [ARCHIVE_L3]                                              │
│   1. "You can use a B-Tree index..."                        │
├─────────────────────────────────────────────────────────────┤
│ L1.5: Entity Ledger                                         │
│   [user_name]: "Alice"                                      │
│   [target_db]: "postgres://localhost:5432"                  │
├─────────────────────────────────────────────────────────────┤
│ L1: Working Memory (Recent Messages)                        │
│   User: "How do I index the users table?"                   │
└─────────────────────────────────────────────────────────────┘

```

* **L0 (System):** Immutable persona definitions, tool schemas, and core constraints.
* **L2 (Archive):** Highly compressed, token-efficient narrative of older conversation turns.
* **L3 (Semantic Archive):** Vector-indexed chunks of evicted L1 text stored in pgvector. Automatically retrieved and injected into `build_prompt()` based on the latest user query.
* **L1.5 (Ledger):** Exact string matching for UUIDs, transaction IDs, file paths, and other critical values. Populated by the Entity Guard pipeline (regex, salience heuristics, ingest-time scanning, and explicit pinning) with rolling conflict history per key.
* **L1 (Working):** The uncompressed, verbatim text of the most recent `N` conversation turns.

---

## 3. Component Deep Dive

### `ContextManager`

The primary async interface for your application. It manages the lifecycle of the memory state, handles the prompt building logic, and safely spins up/shuts down the background worker.

### Sync managers

- **`SyncContextManager`** — sync-native path for scripts and WSGI. Same L0–L2 (optional L3) model and Entity Guard, with **inline** compression on the calling thread. No EventBus or background worker.
- **`SawtoothSyncWrapper`** — sync call sites that still need the async worker. Bridges through an AnyIO BlockingPortal so ingest stays non-blocking.

See [§5](#5-api--usage-guide) for the decision table and [§10](#10-api-reference) for method signatures.

### `CompressionWorker`

A dedicated background asynchronous task for legacy eager compression and DTE
pull consolidation. `ContextManager` monitors L1; in default DTE mode it folds
the oldest `chunk_size` messages locally, while `always_llm` sends that chunk to
the designated compression model.

### Dual-Target Externalization (DTE)

DTE is the default compression policy. It optimizes the two token-cost sources
separately:

1. **Observation Crush** deterministically compacts large `tool` messages before
   they enter L1. JSON arrays are sampled with their count and logs retain errors
   plus nearby context. The original remains available through
   `retrieve_observation(cache_id)` in a bounded in-process cache.
2. **Structured Folding** handles a soft-limit eviction without an LLM call.
   Entity Guard writes exact values to L1.5, optional L3 stores the raw trajectory,
   and L2 receives a compact `[FOLD ...]` outcome record.
3. **Intent-aware prompt assembly** scopes L3 retrieval and can omit redundant L2
   narrative when an entity lookup is already covered by L1.5.
4. **Pull consolidation** converts accumulated folds into narrative only when
   narrative debt or query intent warrants it and the configured background spend
   ratio permits it. A novelty filter removes ledger-covered text before the call.

Set `compression_mode="always_llm"` to retain the legacy behavior where every
soft-limit cycle immediately calls the compression model.

This design follows current agent-context research by treating observations and
history as separate compression targets, preferring recoverable externalization
over irreversible rewriting, and separating the compression mechanism from its
control policy.

### `MemoryState`

A thread-safe data structure that holds the current L1, L1.5, and L2 data. It utilizes `asyncio.Lock()` to prevent race conditions when the background worker attempts to modify the summary while the main thread is simultaneously reading the prompt.

### Entity Guard Pipeline

Sawtooth protects critical identifiers through a layered, local-first pipeline that runs entirely in-process:

1. **Regex extraction** — High-precision patterns for UUIDs, file paths, URIs, and user-defined formats. Captures all matches per pattern (e.g. `uuid`, `uuid_2`).
2. **Salience extraction** — Heuristic scoring (cue-word proximity, structural shape, entropy, rarity) catches unstructured identifiers like `INC-4421` or `ALPHA-991` without predefined regex.
3. **Protection manifest** — Locally discovered entities are injected into the compression LLM prompt as a `PROTECTED VALUES` block.
4. **Secure merge** — Local entities override LLM-extracted values on key collision.
5. **Post-merge verifier** — Re-injects protected values dropped from both `extracted_entities` and the narrative summary.

**Ingest-time scanning:** When `enable_ingest_entity_scan=True`, `add_message()` runs the same local extractors on incoming text so the live L1 window is protected before compression.

**Explicit pinning:** `pin_entity(key, value)` writes a critical value directly to L1.5 with `pinned` strategy provenance.

Extraction strategies tracked in telemetry: `deterministic`, `salience_heuristic`, `pinned`, `llm_synthesis`.

---

## 4. Configuration Reference

Sawtooth is configured via the `ContextManagerConfig` Pydantic model.

```python
from sawtooth_memory import ContextManagerConfig
from sawtooth_memory.config import OllamaConfig, CloudConfig, Provider
from pydantic import SecretStr

# Local Execution (Ollama)
local_config = ContextManagerConfig(
    compression_mode="dte",    # Zero-LLM soft-limit folding (default)
    soft_limit_tokens=1000,    # Trigger compression at 1k tokens
    hard_limit_tokens=2500,    # Failsafe truncation limit
    chunk_size=4,              # Compress 4 messages at a time
    tokenizer_model="gpt-4o",  # Tokenization algorithm
    ollama=OllamaConfig(
        base_url="http://localhost:11434",
        model="phi4"
    )
)

# Cloud Execution (OpenAI)
cloud_config = ContextManagerConfig(
    soft_limit_tokens=4000,
    hard_limit_tokens=8000,
    chunk_size=10,
    cloud=CloudConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        api_key=SecretStr("sk-...")
    )
)

# V2 shorthand — auto-routes cloud models via OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY
v2_config = ContextManagerConfig(background_model="gpt-4o-mini")

```

**Key Parameters:**

* `soft_limit_tokens`: The threshold that awakens the background worker.
* `hard_limit_tokens`: A strict ceiling. If the worker cannot compress fast enough (e.g., due to API rate limits), the system will brutally truncate old messages to prevent your main agent from crashing due to context window overflow.
* `chunk_size`: The number of messages shifted from L1 to L2 per compression cycle.
* `compression_mode`: `"dte"` (default) or `"always_llm"` (legacy eager compression).
* `enable_observation_crush` / `obs_crush_min_tokens`: Enable local compaction of
  large tool observations and set its threshold (default 800 tokens).
* `narrative_debt_trigger_tokens`: Evicted-token debt required before background
  narrative consolidation is eligible (default 2000).
* `background_spend_ratio`: Maximum cumulative background-compressor input tokens
  divided by compiled main-prompt tokens (default 0.1).
* `enable_sync_consolidation`: Permit inline consolidation during
  `SyncContextManager.build_prompt()` (default `False` to avoid unexpected
  caller-thread LLM latency). `SawtoothSyncWrapper` uses the async worker instead.
* `enable_novelty_filter` / `novelty_min_residual`: Strip known ledger values and
  duplicate lines; skip calls whose residual ratio is too small.
* `enable_intent_prompt_planner`: Dynamically budget L3 and omit redundant L2.
* `compression_guideline`: Optional ACON-style consolidator instruction.

**Entity Guard Parameters:**

* `enable_deterministic_ner`: Enable the local regex extraction layer (default: `True`).
* `custom_ner_patterns`: User-defined `key → regex` mappings extending default UUID/file-path/URI patterns.
* `enable_salience_extractor`: Enable heuristic extraction for unstructured identifiers (default: `True`).
* `salience_threshold`: Minimum salience score (0–1) to promote a candidate to L1.5 (default: `0.5`).
* `salience_max_entities`: Maximum heuristic entities per scan/compression cycle (default: `20`).
* `enable_ingest_entity_scan`: Run entity extraction on `add_message()` for live L1 protection (default: `True`).
* `enable_entity_verifier`: Re-inject protected values dropped by the compression LLM (default: `True`).

**L3 Semantic Storage Parameters:**

* `enable_l3_semantic_storage`: When `True`, evicted L1 text is chunked, batch-embedded, and persisted to pgvector during background compression. Requires a `PostgresStorageAdapter`.
* `enable_l3_prompt_retrieval`: When `True` (default), L3 chunks are automatically retrieved and injected into `build_prompt()`. Ignored if L3 storage is disabled.
* `l3_retrieval_top_k`: Maximum number of L3 semantic chunks to retrieve during `build_prompt()`. Defaults to 3.
* `l3_retrieval_max_tokens`: Token budget for the L3 retrieval block in `build_prompt()`. Defaults to 500.
* `embedding_backend`: `"hash"` for deterministic local vectors (tests/dev) or `"openai"` for production-quality embeddings.
* `embedding_model`: OpenAI embedding model name (default: `text-embedding-3-small`).
* `embedding_dimension`: Vector width; must match `PostgresStorageAdapter.embedding_dimension`.
* `l3_chunk_max_chars`: Maximum characters per semantic chunk before splitting (default: 2000).

---

## 5. API & Usage Guide

Sawtooth ships three first-class managers. Import them from the package root:

```python
from sawtooth_memory import (
    ContextManager,          # async, non-blocking worker
    SyncContextManager,      # sync, inline blocking compression
    SawtoothSyncWrapper,     # sync call sites + async worker via AnyIO portal
)
```

### Choosing a manager (sync environments formalized)

| Host environment | Recommended API | Compression | Events / journal | Notes |
|---|---|---|---|---|
| Scripts, notebooks, simple CLIs | `SyncContextManager` | **Blocking** inline | No | Default via `for_sync_script()`; L3 off unless you enable it |
| Flask / Django / sync WSGI that can tolerate brief stalls | `SyncContextManager` | **Blocking** inline | No | Set `enable_sync_consolidation=True` only if you want LLM consolidation during `build_prompt()` |
| Sync host that must keep ingest non-blocking | `SawtoothSyncWrapper` | **Non-blocking** background worker | Yes (optional) | Same semantics as async `ContextManager` |
| FastAPI, LangGraph, asyncio agents | `ContextManager` | **Non-blocking** worker | Yes (optional) | Prefer `async with` lifecycle |

**Rule of thumb:** start with `SyncContextManager`. Graduate to `SawtoothSyncWrapper` only when inline compression latency is unacceptable. Use async `ContextManager` whenever the host already has an event loop.

Shared method surface (sync managers omit `async` / `await`):

| Method | Purpose |
|---|---|
| `add_message(role, content)` | Append to L1; may trigger compression |
| `pin_entity(key, value)` | Force exact L1.5 retention (`pinned` strategy) |
| `retrieve_observation(cache_id)` | Recover raw tool text from Observation Crush cache |
| `build_prompt(*, retrieval_query=None)` | Compile L0 + L2 + L1.5 + L1 (+ optional L3) |
| `explain_prompt()` | Deterministic audit trail |
| `search_semantic_archive(query, top_k=5)` | Explicit L3 search |
| `l3_chunk_count()` | Indexed L3 chunk count |
| `get_stats()` / `health_check()` | Runtime diagnostics |
| `.state` | Live `MemoryState` (prefer read-only) |

### Sync-native API (scripts and WSGI)

```python
from sawtooth_memory import SyncContextManager, ContextManagerConfig

config = ContextManagerConfig.for_sync_script(soft_limit_tokens=1500)

with SyncContextManager(system_prompt="You are an expert.", config=config) as memory:
    memory.add_message("user", "Hello.")
    memory.add_message("assistant", "Hi there.")
    payload = memory.build_prompt()
    memory.pin_entity("tracking_code", "ALPHA-991")
    trace = memory.explain_prompt()
```

`ContextManagerConfig.for_sync_script()` turns L3 off by default. Override any field as needed.

### Sync non-blocking portal (`SawtoothSyncWrapper`)

```python
from sawtooth_memory import SawtoothSyncWrapper, ContextManagerConfig, get_event_bus

config = ContextManagerConfig(soft_limit_tokens=1500)

async def on_cycle(event):
    print(event.messages_compressed, event.l1_tokens_evicted)

get_event_bus().subscribe("compression.cycle_complete", on_cycle)

with SawtoothSyncWrapper("You are an expert.", config=config) as memory:
    memory.add_message("user", "Escalate INC-4421")
    memory.pin_entity("ticket_id", "INC-4421")
    payload = memory.build_prompt()
```

### Async API (FastAPI, LangGraph, asyncio agents)

Prefer `async with` so the background worker starts and stops cleanly.

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig

async def agent_loop():
    config = ContextManagerConfig(soft_limit_tokens=1500)

    async with ContextManager(system_prompt="You are an expert.", config=config) as cm:
        await cm.add_message("user", "Hello.")
        await cm.add_message("assistant", "Hi there.")
        payload = await cm.build_prompt()
```

### Pinning Critical Entities

```python
# Async
await cm.pin_entity("tracking_code", "ALPHA-991")

# Sync (SyncContextManager or SawtoothSyncWrapper)
memory.pin_entity("tracking_code", "ALPHA-991")
```

Pinned entities are tagged with strategy `pinned` in the JSONL journal and explainability traces.

### Entity Guard Configuration Example

```python
config = ContextManagerConfig(
    enable_deterministic_ner=True,
    custom_ner_patterns={
        "transaction_id": r"txn_[a-z0-9_]+",
    },
    enable_salience_extractor=True,
    salience_threshold=0.4,
    enable_ingest_entity_scan=True,
    enable_entity_verifier=True,
)
```

### Manual Lifecycle Management

If you cannot use `async with` (e.g., inside certain web framework state objects), start and stop manually:

```python
cm = ContextManager(system_prompt="...", config=config)
await cm.start()
# ... app logic ...
await cm.stop()  # Flushes pending compression and saves journal
```

For sync portal hosts, always use `with SawtoothSyncWrapper(...)` — it owns the BlockingPortal lifecycle.

---

## 6. Observability & Telemetry

Sawtooth provides deep visibility into memory operations, which is crucial for debugging enterprise agent systems.

### Explainability Traces

The `explain_prompt()` method returns a deterministic audit trail showing exactly what data is in the active payload and where it came from. Each L1.5 entity includes a strategy label: `deterministic`, `salience_heuristic`, `pinned`, or `llm_synthesis`.

```python
trace = cm.explain_prompt()
print(trace)
# {
#   "l0_system": {"content": "...", "origin": "Hardcoded System Initialization"},
#   "l2_archival": {"content": "...", "origin": "Background Ollama Compression (L1 -> L2)"},
#   "l1_5_entities": [
#       {
#           "entity_key": "ticket_id",
#           "entity_value": ["INC-4421"],
#           "origin": "Anchored via explicit tracking engine (Operation: insert) [Strategy: salience_heuristic]",
#           "confidence": "90% (Salience Heuristic)"
#       }
#   ],
#   "l1_working_messages": 2
# }

```

### The Event Bus

Subscribe via the package-level bus (there is no public `cm.event_bus` property). Handlers must be async callables; event types are string literals matching each event dataclass.

```python
from sawtooth_memory import get_event_bus, CompressionCycleCompleteEvent

async def on_compression(event: CompressionCycleCompleteEvent) -> None:
    print(f"Compressed {event.messages_compressed} messages.")
    print(f"L1 tokens evicted: {event.l1_tokens_evicted}")

get_event_bus().subscribe("compression.cycle_complete", on_compression)
```

Common event type strings:

| String | Dataclass |
|---|---|
| `l1.eviction` | `L1EvictionEvent` |
| `l1_5.entity_anchored` | `EntityAnchoredEvent` |
| `l2.summary_generated` | `L2SummaryGeneratedEvent` |
| `compression.cycle_started` | `CompressionCycleStartEvent` |
| `compression.cycle_complete` | `CompressionCycleCompleteEvent` |
| `compression.cycle_failed` | `CompressionCycleFailedEvent` |
| `monitor.soft_limit_reached` | `SoftLimitReachedEvent` |
| `monitor.hard_limit_reached` | `HardLimitReachedEvent` |
| `dte.fold_created` | `DTEFoldCreatedEvent` |
| `l3.vector_indexed` | `L3VectorIndexedEvent` |

---

## 7. LangGraph Integration

Sawtooth provides `SawtoothLangGraphAdapter`, which syncs LangGraph message state into the hierarchical memory stack and returns a sanitized, compressed prompt (including automatic orphan `ToolMessage` removal).

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from sawtooth_memory import ContextManager, ContextManagerConfig
from sawtooth_memory.integrations.langgraph import SawtoothLangGraphAdapter

config = ContextManagerConfig(...)
cm = ContextManager(system_prompt="System", config=config)
adapter = SawtoothLangGraphAdapter(context_manager=cm)

async def chat_node(state):
    await adapter.sync_state(state["messages"])
    safe_prompt = await adapter.get_compiled_prompt()
    # response = await llm.ainvoke(safe_prompt)
    return {"messages": [...]}

async with cm:
    graph = StateGraph(State).add_node("agent", chat_node).compile()
    async for event in graph.astream({"messages": [("user", "Hello")]}):
        print(event)

```

---

## 8. Persistence & Journaling

By default, Sawtooth writes an append-only JSONL audit journal for compression cycles and entity anchoring events.

When events are enabled, compression cycle completions and entity ledger mutations are appended to the path configured via `ContextManagerConfig.journal_path` (default: `.sawtooth_journal.jsonl`). The journal is intended for observability, debugging, and explainability traces — not automatic state recovery on restart.

For durable session persistence across process restarts or multi-container deployments, use `RedisStorageAdapter` or `PostgresStorageAdapter` via `ContextManagerConfig.storage_adapter`. See [Distributed Storage & L3 Semantic Archival](#9-distributed-storage--l3-semantic-archival).

---

## 9. Distributed Storage & L3 Semantic Archival

### Storage Backends

| Adapter | Scope | Best For |
|---------|-------|----------|
| `RedisStorageAdapter` | L0+L1 per session; L1.5+L2 per pool | High-speed ephemeral sessions |
| `PostgresStorageAdapter` | Full `MemoryState` JSONB + pgvector L3 | Durable multi-container deployments |

Configure via `ContextManagerConfig.storage_adapter` and `session_id`. Multi-agent pools use `pool_id` to share L1.5 and L2 across agents.

**Postgres setup:** Install the [pgvector](https://github.com/pgvector/pgvector) extension on your PostgreSQL server before using `PostgresStorageAdapter`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### L3 Semantic Vector Archival (Storage Layer)

L3 complements L2's narrative summary with **semantic vector chunks** of evicted L1 text. During each compression cycle the background worker:

1. Chunks the evicted L1 transcript (paragraph-aware splitting).
2. Batch-embeds all chunks in a single provider call.
3. Batch-inserts vectors into `sawtooth_semantic_vectors` via `executemany`.

Efficiency choices:
- Embeddings are batched per compression cycle (one HTTP call for OpenAI).
- Postgres writes use a single transaction with `executemany`.
- HNSW cosine index on embeddings for fast similarity search.
- Session-scoped B-tree index on `session_id`.

**Retrieval is automatically injected into `build_prompt()`** when `enable_l3_prompt_retrieval=True` (default). You can also query the storage layer directly:

```python
matches = await cm.search_semantic_archive("transaction ID dispute", top_k=5)
stats = await cm.l3_chunk_count()
trace = cm.explain_prompt()  # includes l3_semantic metadata
```

Enable L3 indexing:

```python
from sawtooth_memory import ContextManager, ContextManagerConfig, PostgresStorageAdapter

postgres = PostgresStorageAdapter(
    dsn="postgresql://user:pass@localhost/sawtooth",
    embedding_dimension=1536,
)

config = ContextManagerConfig(
    storage_adapter=postgres,
    session_id="user_123",
    enable_l3_semantic_storage=True,
    embedding_backend="openai",  # or "hash" for local/tests
    embedding_dimension=1536,
)
```

Telemetry: subscribe to `l3.vector_indexed` events on the event bus for chunk counts and embedding backend metadata.

---

## 10. API Reference

All symbols below are imported from `sawtooth_memory` unless noted. Optional framework bridges live under `sawtooth_memory.integrations.*` and require extras.

### Package exports

**Managers:** `ContextManager`, `SyncContextManager`, `SawtoothSyncWrapper`

**Config:** `ContextManagerConfig`, `OllamaConfig`, `CloudConfig`, `Provider`

**State / results:** `MemoryState`, `SemanticChunkResult`

**Storage:** `BaseStorageAdapter`, `RedisStorageAdapter`, `PostgresStorageAdapter`

**Events:** `EventBus`, `get_event_bus`, `reset_event_bus`, `make_journal_handler`, `SawtoothEvent`, plus typed event dataclasses listed in §6

**Embeddings:** `EmbeddingProvider`, `HashEmbeddingProvider`, `OpenAIEmbeddingProvider`, `create_embedding_provider`

**Exceptions:** `SawtoothError`, `CompressionError`, `OllamaConnectionError`, `MalformedOutputError`, `TokenLimitExceededError`

### `ContextManager` (async)

```text
ContextManager(
    system_prompt: str,
    config: ContextManagerConfig | None = None,
    *,
    enable_events: bool = True,
    journal_path: Path | None = None,
)
```

| Method | Signature | Description |
|---|---|---|
| `start` / `stop` | `async () -> None` | Lifecycle without context manager |
| `add_message` | `async (role, content) -> None` | Append L1; soft limit enqueues worker |
| `pin_entity` | `async (key, value) -> None` | Pin L1.5 entity |
| `retrieve_observation` | `(cache_id) -> str \| None` | Observation Crush lookup |
| `build_prompt` | `async (*, retrieval_query=None) -> list[dict]` | Compile OpenAI-style messages |
| `search_semantic_archive` | `async (query, top_k=5) -> list[SemanticChunkResult]` | Explicit L3 search |
| `l3_chunk_count` | `async () -> int` | Indexed chunk count |
| `explain_prompt` | `() -> dict` | Explainability trace |
| `get_stats` | `() -> dict` | Tokens, DTE, worker snapshot |
| `health_check` | `async () -> dict` | Readiness report |
| `state` | property `MemoryState` | Live state tree |

### `SyncContextManager` (sync, inline)

```text
SyncContextManager(
    system_prompt: str,
    config: ContextManagerConfig | None = None,  # defaults to for_sync_script()
)
```

Same method names as above without `async`. Compression runs on the calling thread. No EventBus and no background worker. Optional L3 methods bridge storage via a one-shot event loop helper.

Sync-only config flags of interest:

- `enable_sync_consolidation` — allow LLM consolidation during `build_prompt()`
- `ContextManagerConfig.for_sync_script(...)` — L3 disabled defaults for scripts

### `SawtoothSyncWrapper` (sync façade, non-blocking)

```text
SawtoothSyncWrapper(
    system_prompt: str,
    config: ContextManagerConfig | None = None,
    *,
    enable_events: bool = True,
    journal_path: Path | None = None,
)
```

Proxies the full async `ContextManager` through an AnyIO `BlockingPortal`. Method surface matches `SyncContextManager` (`add_message`, `pin_entity`, `retrieve_observation`, `build_prompt`, `explain_prompt`, L3 helpers, `state`, `get_stats`, `health_check`).

### `ContextManagerConfig` (selected fields)

| Field | Default | Notes |
|---|---|---|
| `soft_limit_tokens` / `hard_limit_tokens` | `1000` / `2500` | Compression triggers |
| `chunk_size` | `4` | Messages folded/summarized per cycle |
| `compression_mode` | `"dte"` | `"dte"` or `"always_llm"` |
| `background_model` | `None` | Auto-route Ollama vs cloud by model name |
| `ollama` / `cloud` | — | Explicit backend configs |
| `enable_sync_consolidation` | `False` | SyncContextManager only |
| `storage_adapter` / `session_id` / `pool_id` | — | Distributed persistence + multi-agent sharing |
| `enable_l3_semantic_storage` | `False` | Requires Postgres + semantic mixin |
| `enable_l3_prompt_retrieval` | `True` | Inject `[ARCHIVE_L3]` into prompts |
| `embedding_backend` | `"hash"` | `"hash"` or `"openai"` |
| Entity Guard flags | on/off | See §3 / §4 |

### Storage adapters

```python
from sawtooth_memory import RedisStorageAdapter, PostgresStorageAdapter

RedisStorageAdapter(
    redis_url="redis://localhost:6379/0",
    key_prefix="sawtooth:session:",
    ttl_seconds=86400,
)

PostgresStorageAdapter(
    dsn="postgresql://user:pass@localhost/sawtooth",
    embedding_dimension=1536,
    min_pool_size=2,
    max_pool_size=10,
)
```

Both implement `load_state` / `save_state` / `delete_state` / `load_pool_state` / `save_pool_state`. Postgres additionally implements L3 vector upsert/search. Call `await adapter.close()` on shutdown.

### Integrations (extras)

```python
# pip install "sawtooth-memory[langgraph]"
from sawtooth_memory.integrations.langgraph import (
    SawtoothLangGraphAdapter,
    build_sawtooth_graph,
    AgentState,
)

# pip install "sawtooth-memory[langchain]"
from sawtooth_memory.integrations.langchain_adapter import SawtoothChatMessageHistory
```

`SawtoothLangGraphAdapter.sync_state(messages)` ingests LangGraph messages; `get_compiled_prompt()` returns sanitized LangChain messages (orphan `ToolMessage`s removed).

`SawtoothChatMessageHistory` wraps `SawtoothSyncWrapper` for LangChain `BaseChatMessageHistory` compatibility (`init_portal` / `close_portal` required around the session).

### Embeddings

```python
from sawtooth_memory import create_embedding_provider

provider = create_embedding_provider("hash", dimension=384)
# or: create_embedding_provider("openai", model="text-embedding-3-small", api_key=...)
vectors = await provider.embed(["chunk one", "chunk two"])
```

---

## 11. Examples

| Script | Demonstrates |
|---|---|
| [`examples/simple_sync_script.py`](examples/simple_sync_script.py) | `SyncContextManager` linear script |
| [`examples/sync_nonblocking_wrapper.py`](examples/sync_nonblocking_wrapper.py) | `SawtoothSyncWrapper` + event bus |
| [`examples/basic_agent.py`](examples/basic_agent.py) | Async `ContextManager` + Entity Guard |
| [`examples/cloud_compressor.py`](examples/cloud_compressor.py) | Explicit `CloudConfig` / `Provider` |
| [`examples/multi_agent_pool.py`](examples/multi_agent_pool.py) | Shared `pool_id` across agents |
| [`examples/postgres_l3_agent.py`](examples/postgres_l3_agent.py) | Postgres + L3 retrieval (`SAWTOOTH_PG_DSN`) |
| [`examples/langgraph_integration.py`](examples/langgraph_integration.py) | LangGraph adapter + ToolMessage-safe prompt |

---
