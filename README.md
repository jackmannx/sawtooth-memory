# Sawtooth Memory

[![Automated Test Suite](https://github.com/HtooTayZa/sawtooth-memory/actions/workflows/test.yaml/badge.svg)](https://github.com/HtooTayZa/sawtooth-memory/actions/workflows/test.yaml)
[![PyPI version](https://badge.fury.io/py/sawtooth-memory.svg)](https://badge.fury.io/py/sawtooth-memory)
[![Python Support](https://img.shields.io/pypi/pyversions/sawtooth-memory.svg)](https://pypi.org/project/sawtooth-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Async hierarchical memory middleware for LLM agents that mitigates "Lost in the Middle" effects via local or cloud-based context compression.

Sawtooth Memory prevents context-window degradation by continuously compressing older conversation state into structured long-term memory — without blocking the main agent execution loop.

Instead of storing entire conversations indefinitely or relying purely on retrieval, Sawtooth maintains a layered memory model:

- Recent messages remain verbatim
- Important operational entities persist exactly
- Older context compresses into narrative state summaries
- Compression runs fully asynchronously in the background

The result is bounded prompt growth with stable long-session behavior.

```text
agent loop
    │
    ▼
┌─────────────────────┐
│   ContextManager    │
│  ┌───────────────┐  │
│  │ L0 System     │  │  immutable persona + tool schemas
│  │ L2 Archive    │  │  compressed narrative memory
│  │ L1.5 Entities │  │  exact IDs, paths, UUIDs
│  │ L1 Working    │  │  recent raw conversation
│  └───────────────┘  │
└──────────┬──────────┘
           │
           ▼
     build_prompt()
           │
           ▼
        LLM API

```

The name "Sawtooth" comes from the token usage pattern created by periodic compression cycles.

---

## Why This Exists

Long-running agents eventually fail for predictable reasons:

- Context windows fill up with stale history
- Critical execution anchors get buried and ignored
- Naive summarization drops exact identifier strings
- RAG systems lose conversational flow continuity
- Synchronous context compression blocks or lags the main loop

Most memory systems optimize for either pure storage or external retrieval. Sawtooth optimizes for **prompt survivability**. It continuously reshapes conversation history into a compact working context while preserving exact operational state separately.

---

## Core Design

Sawtooth uses four structured memory tiers:

| Layer | Purpose        | Characteristics               |
| ----- | -------------- | ----------------------------- |
| L0    | System memory  | Immutable instruction layers  |
| L1    | Working memory | Recent verbatim messages      |
| L1.5  | Entity ledger  | Exact structured state anchor |
| L2    | Archive memory | Compressed narrative history  |

### L0 — System Memory

Contains system prompts, tool schemas, agent roles, and static rules. L0 remains immutable throughout the session.

### L1 — Working Memory

A sliding window of recent raw conversation turns representing the active reasoning surface. When usage crosses `soft_limit_tokens`, older terms are queued for background processing.

### L1.5 — Entity Ledger

Structured exact-value persistence. Because summarization is lossy, critical identifiers like UUIDs, database transaction keys, connection endpoints, and absolute file paths must remain exact. L1.5 preserves these elements in a clean structured schema to prevent them from disappearing into text narrative summaries.

### L2 — Archive Memory

An append-only, compressed long-horizon narrative history recording historical actions and agent outcomes. It is optimized for semantic continuity rather than verbatim replay.

---

# How Compression Works

Compression is fully asynchronous. The main agent loop never blocks waiting for a summarization API response.

When L1 exceeds the configured soft limit:

1. The oldest messages are sliced off into discrete chunks.
2. Chunk collections are offloaded onto a background `asyncio` worker queue.
3. Chat clutter is pruned and evaluated.
4. Cleaned components are compressed using your configured model.
5. Extracted text insights are merged cleanly back into the L2 Archive and L1.5 Entity Ledger.
6. The original raw messages are safely cleared from L1.

This produces a predictable, repeating "sawtooth" token profile instead of monotonic prompt growth.

---

# Design Features

- **Bounded Prompt Growth:** Keeps input payloads stable across lengthy multi-turn sessions.
- **Non-Blocking Execution:** Worker processing occurs off the primary execution timeline.
- **Failure Isolation:** Background summary errors never interrupt or crash your agent loop.
- **Framework Agnostic:** Generates standard message arrays compatible with any OpenAI-style client.
- **Local-First Capabilities:** Context reduction tasks can run 100% locally on an Ollama instance.

---

# Installation

Install the core package from PyPI:

```bash
pip install sawtooth-memory

```

To include optional LangGraph integration dependencies:

```bash
pip install "sawtooth-memory[langgraph]"

```

To install from source for local development:

```bash
git clone [https://github.com/HtooTayZa/sawtooth-memory](https://github.com/HtooTayZa/sawtooth-memory)
cd sawtooth-memory
pip install -e ".[dev,langgraph]"

```

### Runtime Requirements

- Python >= 3.11
- Either a local Ollama service running or accessible cloud backend API keys (OpenAI, Anthropic, Gemini).

```bash
ollama serve
ollama pull phi4

```

---

# Examples

You can find complete, runnable implementations inside the [`/examples`](https://www.google.com/search?q=https://github.com/HtooTayZa/sawtooth-memory/tree/main/examples) directory of the repository:

- `basic_agent.py` — Demonstrates standalone `ContextManager` message ingestion and memory state telemetry.
- `langgraph_integration.py` — Details how to wire up the `SawtoothLangGraphAdapter` seamlessly inside custom graph nodes.

---

# Quick Start

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig

config = ContextManagerConfig(
    soft_limit_tokens=3000,
    hard_limit_tokens=6000,
    chunk_size=10,
)

async def main():
    async with ContextManager(
        system_prompt="You are a data analysis assistant.",
        config=config,
    ) as memory:

        await memory.add_message("user", "Analyze Q3 revenue trends.")
        await memory.add_message("assistant", "Connecting to PostgreSQL database.")
        await memory.add_message("tool", '{"connection_id": "conn_994a82"}')

        # Retrieve the fully compiled, optimized message context
        messages = memory.build_prompt()

        # Pass the formatted array directly to your LLM framework client
        # response = await client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=messages,
        # )

        print(memory.get_stats())

if __name__ == "__main__":
    asyncio.run(main())

```

---

# Compiled Prompt Structure

`build_prompt()` returns standard OpenAI-format messages. The compound system context prompt is assembled dynamically matching this layout:

```text
[SYSTEM_L0]
You are a data analysis assistant.

[ARCHIVE_L2]
User requested Q3 analysis. Connected to PostgreSQL.
Detected revenue anomalies across enterprise channels.

[ENTITY_LEDGER_L1_5]
{
  "connection_id": "conn_994a82",
  "dataset": "sales_q3_2026"
}

```

Recent active conversation sequences follow directly underneath as raw, uncompressed verbatim turns.

---

# Configuration

Sawtooth Memory uses Pydantic configurations. You can back your background compilation loops using either local inference or public cloud engines.

### Local Backend Configuration (Ollama)

```python
from sawtooth_memory import ContextManagerConfig, OllamaConfig

config = ContextManagerConfig(
    soft_limit_tokens=3000,
    hard_limit_tokens=6000,
    chunk_size=10,
    tokenizer_model="gpt-4o",
    fallback_truncate=True,
    ollama=OllamaConfig(
        base_url="http://localhost:11434",
        model="phi4",
        timeout_seconds=90,
    ),
)

```

### Cloud Backend Configuration (OpenAI, Anthropic, Gemini)

```python
from sawtooth_memory import ContextManagerConfig
from sawtooth_memory.config import CloudConfig, Provider

config = ContextManagerConfig(
    soft_limit_tokens=3000,
    hard_limit_tokens=6000,
    chunk_size=10,
    fallback_truncate=True,
    cloud=CloudConfig(
        provider=Provider.ANTHROPIC,
        model="claude-3-5-haiku-latest",
        api_key="your-api-key-here",
        timeout_seconds=60,
        base_url=None,
    ),
)

```

### Core Configuration Fields

| Parameter           | Type           | Default    | Description                                                                                      |
| ------------------- | -------------- | ---------- | ------------------------------------------------------------------------------------------------ |
| `soft_limit_tokens` | `int`          | `3000`     | The token baseline that triggers asynchronous background compression tasks.                      |
| `hard_limit_tokens` | `int`          | `6000`     | The absolute safety limit for window sizing before strict processing takes effect.               |
| `chunk_size`        | `int`          | `10`       | The specific amount of early raw chat messages sliced off into each compression payload.         |
| `tokenizer_model`   | `str`          | `"gpt-4o"` | The underlying tokenizer schema encoding rule used for tracking lookups.                         |
| `fallback_truncate` | `bool`         | `True`     | Flags if the loop should drop elements smoothly if a cloud or local summarizer fails completely. |
| `ollama`            | `OllamaConfig` | _Factory_  | Configuration parameters defining the target local Ollama endpoint environment.                  |
| `cloud`             | `CloudConfig`  | `None`     | Endpoint authentication, provider types, and configuration metadata targeting cloud APIs.        |

---

# Comparison Matrix

| Framework Memory Variant      | Reduction Strategy                    | Structural Compression | Exact Identity Tracker | Asynchronous Non-Blocking |
| ----------------------------- | ------------------------------------- | ---------------------- | ---------------------- | ------------------------- |
| **ConversationSummaryMemory** | Rolling basic text summary            | Yes                    | No                     | No                        |
| **Mem0**                      | Memory item graph retrieval           | Partial                | No                     | Partial                   |
| **MemPalace**                 | External semantic vector retrieval    | No                     | No                     | No                        |
| **Sawtooth Memory**           | Tiered hierarchical context reduction | **Yes**                | **Yes**                | **Yes**                   |

---

# Roadmap

- [x] LangGraph framework adapter
- [ ] AutoGen framework integration adapter
- [ ] Redis-backed multi-process background worker transport
- [ ] Adaptive semantic salience metric scoring
- [ ] Recursive background archive layer compression
- [ ] Hybrid memory/vector RAG indexing
- [ ] Prometheus telemetry monitoring hooks
- [ ] Native TypeScript/Node framework package port

---

# Repository Structure

```text
sawtooth-memory/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug-report.yml          # Structured bug report input form
│   │   └── feature-request.yml     # Structured feature request input form
│   └── workflows/
│       └── test.yaml               # CI test pipeline
│
├── sawtooth_memory/
│   ├── integrations/
│   │   └── langgraph/
│   │       ├── adapter.py          # LangGraph adapter layer
│   │       └── graph.py            # Graph state definitions
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── adapters.py             # Internal LLM provider adapters
│   │   └── factory.py              # LLM client builder factory
│   │
│   ├── compressor.py               # Core compression/summarization logic
│   ├── config.py                   # Configuration validation schemas
│   ├── exceptions.py               # Package exceptions
│   ├── middleware.py               # Context middleware entrypoint
│   ├── monitor.py                  # Telemetry and token tracking metrics
│   ├── state.py                    # Multi-tier state representation
│   └── worker.py                   # Async background background loop
│
├── tests/
│   ├── conftest.py
│   ├── test_adapter.py
│   ├── test_cloud_compressor.py
│   ├── test_compressor.py
│   ├── test_graph.py
│   ├── test_middleware.py
│   ├── test_monitor.py
│   └── test_state.py
│
├── examples/
│   ├── basic_agent.py              # Basic standalone workflow snippet
│   └── langgraph_integration.py     # Simple LangGraph workflow snippet
│
├── .gitignore
├── .pre-commit-config.yaml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
└── SECURITY.md

```

---

# License

MIT License — see [`LICENSE`](https://www.google.com/search?q=LICENSE) for implementation conditions.

```

```
