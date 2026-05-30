# Sawtooth Memory

Async hierarchical memory middleware for LLM agents.

Sawtooth Memory mitigates context-window degradation by continuously compressing older conversation state into structured long-term memory — without blocking the agent execution loop.

Instead of storing entire conversations indefinitely or relying purely on retrieval, Sawtooth maintains a layered memory model:

- recent messages remain verbatim
- important entities persist exactly
- older context compresses into narrative state
- compression runs asynchronously in the background

The result is bounded prompt growth with stable long-session behavior.

```text
agent loop
    │
    ▼
┌─────────────────────┐
│   ContextManager    │
│  ┌───────────────┐  │
│  │ L0 System     │  immutable persona + tool schemas
│  │ L2 Archive    │  compressed narrative memory
│  │ L1.5 Entities │  exact IDs, paths, UUIDs
│  │ L1 Working    │  recent raw conversation
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

- context windows fill with stale history
- important information gets buried
- summarization loses exact values
- retrieval systems lose conversational continuity
- synchronous compression blocks the main loop

Most memory systems optimize for storage or retrieval.

Sawtooth optimizes for prompt survivability.

It continuously reshapes conversation history into a compact working context while preserving exact operational state separately.

---

## Core Design

Sawtooth uses four memory tiers.

| Layer | Purpose        | Characteristics              |
| ----- | -------------- | ---------------------------- |
| L0    | System memory  | Immutable                    |
| L1    | Working memory | Recent verbatim messages     |
| L1.5  | Entity ledger  | Exact structured state       |
| L2    | Archive memory | Compressed narrative history |

### L0 — System Memory

Contains:

- system prompts
- tool schemas
- agent rules
- static instructions

L0 never changes.

---

### L1 — Working Memory

Sliding window of recent raw conversation turns.

This is the active reasoning surface used by the model.

When token usage exceeds `soft_limit_tokens`, older messages are queued for asynchronous compression.

---

### L1.5 — Entity Ledger

Structured exact-value persistence.

This layer exists because summarization is lossy.

Things that must remain exact:

- UUIDs
- database IDs
- file paths
- API keys references
- table names
- timestamps
- active resources

Example:

```json
{
  "active_connection": "conn_994a82",
  "workspace_id": "ws_7f31",
  "current_dataset": "sales_q3_2026"
}
```

L1.5 prevents critical operational state from disappearing into narrative summaries.

---

### L2 — Archive Memory

Compressed long-horizon narrative memory.

Example:

```text
User requested Q3 revenue analysis.
Agent connected to PostgreSQL.
Detected a 14% revenue decline in enterprise accounts.
Generated anomaly report and exported CSV.
```

L2 is append-only and optimized for semantic continuity rather than exact replay.

---

# How Compression Works

Compression is asynchronous.

The main agent loop never waits for summarization.

When L1 exceeds the configured soft limit:

1. oldest messages are sliced into chunks
2. chunks are queued onto a background asyncio worker
3. noisy data is pruned
4. cleaned content is sent to a local Ollama model
5. extracted outputs are merged into:
   - L2 narrative memory
   - L1.5 entity state

6. original messages are removed from L1

This creates a repeating "sawtooth" token profile rather than monotonic prompt growth.

---

# Design Goals

## Bounded Prompt Growth

Prompt size remains stable during long-running sessions.

## Non-Blocking Compression

Compression runs off the main execution path.

## Failure Isolation

Compression failures never crash the agent loop.

## Framework Agnostic

Works with any OpenAI-compatible SDK.

## Local-First

All summarization can run entirely on local Ollama models.

---

# Installation

```bash
pip install sawtooth-memory (coming soon)
```

From source:

```bash
git clone https://github.com/HtooTayZa/sawtooth-memory
cd sawtooth-memory
pip install -e ".[dev]"
```

Optional LangGraph support:

```bash
pip install -e ".[langgraph]"
```

Requirements:

- Python 3.11+
- Either a local Ollama instance running OR api keys for cloud backends (OpenAI, Anthropic, Gemini)

```bash
ollama serve
ollama pull phi4
```

---

# Quick Start

```python
import asyncio

from sawtooth_memory import (
    ContextManager,
    ContextManagerConfig,
)

config = ContextManagerConfig(
    soft_limit_tokens=3000,
    hard_limit_tokens=6000,
    chunk_size=10,
)

async def main():

    async with ContextManager(
        system_prompt="You are a data analysis agent.",
        config=config,
    ) as memory:

        await memory.add_message(
            "user",
            "Analyze Q3 revenue trends."
        )

        await memory.add_message(
            "assistant",
            "Connecting to PostgreSQL."
        )

        await memory.add_message(
            "tool",
            '{"connection_id":"conn_994a82"}'
        )

        prompt = memory.build_prompt()

        # response = await client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=prompt,
        # )

        print(memory.get_stats())

asyncio.run(main())
```

---

# Compiled Prompt Structure

`build_prompt()` returns standard OpenAI-format messages.

The system message is assembled dynamically:

```text
[SYSTEM_L0]
You are a data analysis agent.

[ARCHIVE_L2]
User requested Q3 analysis.
Connected to PostgreSQL.
Detected revenue decline in enterprise segment.

[ENTITY_LEDGER_L1_5]
{
  "connection_id": "conn_994a82",
  "dataset": "sales_q3_2026"
}
```

Recent conversation turns remain verbatim beneath the system message.

---

# Failure Handling

If compression fails:

- the agent loop continues
- the worker records a degradation event
- old messages may be truncated depending on configuration

By default:

```python
fallback_truncate=True
```

This favors agent continuity over strict preservation.

Set:

```python
fallback_truncate=False
```

to raise `CompressionError` instead.

---

# What Sawtooth Is Not

Sawtooth is not:

- a vector database
- a retrieval framework
- a persistent knowledge graph
- a semantic search engine
- a replacement for RAG

It is prompt-state middleware.

Sawtooth manages conversational survivability inside bounded context windows.

It works alongside:

- RAG pipelines
- vector stores
- MCP tools
- LangGraph persistence
- external memory systems

---

# Comparison

| System                    | Strategy                 | Compression | Exact State Layer | Async   |
| ------------------------- | ------------------------ | ----------- | ----------------- | ------- |
| ConversationSummaryMemory | Rolling summary          | Yes         | No                | No      |
| Mem0                      | Retrieval memory         | Partial     | No                | Partial |
| MemPalace                 | Verbatim retrieval       | No          | No                | No      |
| Sawtooth                  | Hierarchical compression | Yes         | Yes               | Yes     |

---

# When To Use Sawtooth

Good fit:

- long-running autonomous agents
- coding agents
- research agents
- multi-tool workflows
- persistent orchestration loops
- local-first agent stacks

Probably unnecessary:

- short chats
- single-shot tasks
- stateless pipelines
- retrieval-heavy systems with minimal dialogue state

---

To cleanly update your **Configuration** section in the `README.md` to reflect the newly added multi-provider cloud support, you can replace that entire section with the following fully updated documentation.

It now showcases both the local-first Ollama path and the new production-ready cloud path side by side, making it clear and complete for your users.

---

# Configuration

Sawtooth Memory is configured using Pydantic models. You can back your context compression loop with either a local Ollama stack or cloud frontier models (OpenAI, Anthropic, or Gemini).

### Local Backend (Ollama)

To run entirely on local hardware, pass an `OllamaConfig` block.

```python
from sawtooth_memory import (
    ContextManagerConfig,
    OllamaConfig,
)

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

### Cloud Backend (OpenAI, Anthropic, Gemini)

To offload background compression tasks to a cloud API provider, configure a `CloudConfig` block instead. This mode utilizes native structured outputs and built-in exponential backoff for HTTP 429 rate limits.

```python
from sawtooth_memory import ContextManagerConfig
from sawtooth_memory.config import CloudConfig, Provider

config = ContextManagerConfig(
    soft_limit_tokens=3000,
    hard_limit_tokens=6000,
    chunk_size=10,
    fallback_truncate=True,

    # Configure any supported provider: Provider.OPENAI, Provider.ANTHROPIC, or Provider.GEMINI
    cloud=CloudConfig(
        provider=Provider.ANTHROPIC,
        model="claude-3-5-haiku-latest",
        api_key="your-api-key-here",
        timeout_seconds=60,
        # base_url is optional: use to route via Helicone, LiteLLM, or Azure OpenAI
        base_url=None,
    ),
)

```

### Configuration Parameters

| Parameter           | Type           | Default    | Description                                                                                       |
| ------------------- | -------------- | ---------- | ------------------------------------------------------------------------------------------------- |
| `soft_limit_tokens` | `int`          | `3000`     | Token threshold that triggers background conversation compression.                                |
| `hard_limit_tokens` | `int`          | `6000`     | Maximum token window size allowed before strict enforcement occurs.                               |
| `chunk_size`        | `int`          | `10`       | Number of older conversation messages sliced off into each compression worker chunk.              |
| `tokenizer_model`   | `str`          | `"gpt-4o"` | Tokenizer encoding scheme utilized for active memory tracking calculation.                        |
| `fallback_truncate` | `bool`         | `True`     | If `True`, falls back to tracking-truncation strings when compression fails, ensuring continuity. |
| `ollama`            | `OllamaConfig` | _Factory_  | Active backend properties dedicated to your local Ollama runtime loop.                            |
| `cloud`             | `CloudConfig`  | `None`     | Active properties dedicated to Cloud API orchestration rules.                                     |

```

```

---

# Roadmap

- [x] LangGraph adapter
- [ ] AutoGen adapter
- [ ] Redis-backed worker transport
- [ ] Adaptive salience scoring
- [ ] Recursive archive compression
- [ ] Hybrid retrieval integration
- [ ] Prometheus metrics
- [ ] TypeScript implementation

---

# Repository Structure

```text
sawtooth-memory/
├── .github/
│   └── workflows/
│       └── test.yml                # CI test pipeline
│
├── sawtooth_memory/
│   ├── integrations/
│   │   └── langgraph/
│   │       ├── adapter.py          # LangGraph adapter layer
│   │       └── graph.py            # Graph state definitions
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── compressor.py
│   │   └── factory.py
│   │
│   ├── compressor.py               # Compression + summarization pipeline
│   ├── config.py                   # Configuration models
│   ├── exceptions.py               # Custom exceptions
│   ├── middleware.py               # Context middleware entrypoint
│   ├── monitor.py                  # Telemetry and runtime monitoring
│   ├── state.py                    # Memory tier state management
│   └── worker.py                   # Background compression worker
│
├── tests/
│   ├── conftest.py
│   ├── test_adapter.py
│   ├── test_compressor.py
│   ├── test_graph.py
│   ├── test_middleware.py
│   ├── test_monitor.py
│   └── test_state.py
│
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
└── SECURITY.md
```

# Development

```bash
pytest
ruff check .
```

---

# License

MIT

```

```
