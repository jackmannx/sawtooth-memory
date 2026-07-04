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
  (Returns in <0.01s)                        ▼
            │                         3. Worker awakens
            ▼                                │
  4. cm.build_prompt()                       ▼
            │                         4. Check Token Limits (Soft/Hard)
  (Retrieves active payload)                 │
            │                                ▼
            ▼                         5. Call Compression LLM (Ollama/Cloud)
  5. Send to Main Agent                      │
                                             ▼
                                      6. Extract L1.5 Entities
                                             │
                                             ▼
                                      7. Update L2 Summary
                                             │
                                             ▼
                                      8. Commit to Disk Journal

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
│ L1.5: Entity Ledger                                         │
│   [user_name]: "Alice"                                      │
│   [target_db]: "postgres://localhost:5432"                  │
├─────────────────────────────────────────────────────────────┤
│ L1: Working Memory (Recent Messages)                        │
│   User: "How do I index the users table?"                   │
│   AI: "You can use a B-Tree index..."                       │
└─────────────────────────────────────────────────────────────┘

```

* **L0 (System):** Immutable persona definitions, tool schemas, and core constraints.
* **L2 (Archive):** Highly compressed, token-efficient narrative of older conversation turns.
* **L1.5 (Ledger):** Exact string matching for UUIDs, transaction IDs, names, and explicit rules extracted during compression.
* **L1 (Working):** The uncompressed, verbatim text of the most recent `N` conversation turns.

---

## 3. Component Deep Dive

### `ContextManager`

The primary interface for your application. It manages the lifecycle of the memory state, handles the prompt building logic, and safely spins up/shuts down the background worker.

### `CompressionWorker`

A dedicated background asynchronous task. It monitors the total token count of the L1 Working Memory. When the token count exceeds the configured `soft_limit_tokens`, it slices off the oldest `chunk_size` messages and sends them to the designated compression LLM.

### `MemoryState`

A thread-safe data structure that holds the current L1, L1.5, and L2 data. It utilizes `asyncio.Lock()` to prevent race conditions when the background worker attempts to modify the summary while the main thread is simultaneously reading the prompt.

---

## 4. Configuration Reference

Sawtooth is configured via the `ContextManagerConfig` Pydantic model.

```python
from sawtooth_memory import ContextManagerConfig
from sawtooth_memory.config import OllamaConfig, CloudConfig, Provider
from pydantic import SecretStr

# Local Execution (Ollama)
local_config = ContextManagerConfig(
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

---

## 5. API & Usage Guide

### Initialization & Context Lifecycles

It is highly recommended to use the asynchronous context manager (`async with`) to ensure the background worker thread is properly initialized and gracefully shut down.

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig

async def agent_loop():
    config = ContextManagerConfig(...)

    async with ContextManager(system_prompt="You are an expert.", config=config) as cm:
        # Loop runs instantly
        await cm.add_message("user", "Hello.")
        await cm.add_message("assistant", "Hi there.")

        payload = await cm.build_prompt()

```

### Manual Lifecycle Management

If you cannot use `async with` (e.g., inside certain web framework state objects), you must manually start and stop the manager:

```python
cm = ContextManager(system_prompt="...", config=config)
await cm.start()

# ... app logic ...

await cm.stop() # CRITICAL: Flushes pending compression tasks and saves journal

```

---

## 6. Observability & Telemetry

Sawtooth provides deep visibility into memory operations, which is crucial for debugging enterprise agent systems.

### Explainability Traces

The `explain_prompt()` method returns a deterministic audit trail showing exactly what data is in the active payload and where it came from.

```python
trace = cm.explain_prompt()
print(trace)
# {
#   "system_prompt": "...",
#   "l2_summary_lineage": ["User asked about X", "AI explained Y"],
#   "l1_5_entities": [{"key": "ID", "value": "123", "origin": "Compression Turn 4"}],
#   "l1_active_messages": 2,
#   "total_tokens": 150
# }

```

### The Event Bus

You can subscribe to internal system events to log compression metrics or trigger webhooks.

```python
from sawtooth_memory.events import EventType

def on_compression(event):
    print(f"Compressed {event.data['messages_compressed']} messages.")
    print(f"Tokens saved: {event.data['tokens_saved']}")

cm.event_bus.subscribe(EventType.COMPRESSION_COMPLETED, on_compression)

```

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

By default, Sawtooth implements an append-only JSONL journal for state persistence and crash recovery.

When the `ContextManager` is running, every modification to the L1 buffer, L1.5 ledger, or L2 summary is appended to a local `.sawtooth_journal.jsonl` file.

If the application crashes unexpectedly, re-initializing a `ContextManager` with the same configuration will automatically replay the journal, restoring the exact state of the hierarchical memory stack prior to the crash.

*Note: In production environments spanning multiple stateless containers, you should implement a custom Journal class that writes to Redis or PostgreSQL instead of the local filesystem (Planned for Phase 3).*

---
