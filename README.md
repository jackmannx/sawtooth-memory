# Sawtooth-Memory

**Async context manager middleware for LLM agents.** Solves the ["Lost in the Middle"](https://arxiv.org/abs/2307.03172) problem by dynamically compressing context windows via a local Ollama model running on a background asyncio thread — with zero latency impact on your agent loop.

```
your agent loop
      │
      ▼
┌─────────────────────┐
│   ContextManager    │  ← drop-in middleware
│  ┌───────────────┐  │
│  │ L0  System    │  │  immutable persona + tool schemas
│  │ L2  Archive   │  │  compressed history narrative
│  │ L1.5 Entities │  │  exact IDs, paths, values
│  │ L1  Working   │  │  last N raw messages  ← compression triggers here
│  └───────────────┘  │
└──────────┬──────────┘
           │ build_prompt()
           ▼
      LLM API call
```

---

## Installation

```bash
pip install sawtooth-memory        # from PyPI (coming soon)
# or from source:
git clone https://github.com/your-org/sawtooth-memory
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+, [Ollama](https://ollama.ai) running locally.

```bash
ollama serve
ollama pull phi4   # or any 8B-class model
```

---

## Quick Start

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig

config = ContextManagerConfig(
    soft_limit_tokens=3000,   # trigger compression at 3k tokens in L1
    hard_limit_tokens=6000,   # emergency truncation cap
    chunk_size=10,            # compress 10 messages per background task
)

async def main():
    async with ContextManager("You are a data analysis agent.", config) as cm:
        # Add messages as your agent runs
        await cm.add_message("user", "Analyse Q3 revenue.")
        await cm.add_message("assistant", "Connecting to the database...")
        await cm.add_message("tool", '{"status": "ok", "conn_id": "conn_994a82"}')

        # Pass directly to your LLM SDK — works with any OpenAI-compatible API
        messages = cm.build_prompt()
        # response = await openai_client.chat.completions.create(
        #     model="gpt-4o", messages=messages
        # )

        # Inspect state
        print(cm.get_stats())
        # {
        #   "l0_tokens": 12, "l1_tokens": 47, "l1_message_count": 3,
        #   "l1_5_entity_count": 0, "l2_tokens": 0,
        #   "worker": {"processed": 0, "failed": 0, "queue_depth": 0}
        # }

asyncio.run(main())
```

---

## How It Works

### The Four-Tier Memory Model

| Tier | Name | Mutability | Contents |
|------|------|------------|----------|
| **L0** | System Prompt | Immutable | Agent persona, tool schemas |
| **L1** | Working Memory | Sliding window | Last N raw messages |
| **L1.5** | Entity Ledger | KV upsert | Exact IDs, paths, UUIDs |
| **L2** | Archival Memory | Append-only | Dense narrative summary |

### Asynchronous Compression Pipeline

When L1 exceeds `soft_limit_tokens`, the middleware **non-blockingly** slices the oldest `chunk_size` messages onto a background asyncio queue. The main agent thread continues without waiting.

The background worker:
1. **Prunes** base64 blobs, stack traces, and verbose JSON noise
2. **Sends** the cleaned chunk to a local Ollama model with a strict dual-extraction prompt
3. **Merges** the result: narrative → L2, entities → L1.5, raw messages → deleted

### Compiled Prompt Format

`build_prompt()` returns a standard OpenAI messages list. The system message is structured as:

```
[SYSTEM_L0]
You are a data analysis agent...

[ARCHIVE_L2]
User requested Q3 analysis. You connected to PostgreSQL and found a 14% drop...

[ENTITY_LEDGER_L1_5]
{
  "active_db_connection_id": "conn_994a82",
  "target_table": "sales_q3_2026"
}
```

Followed by raw `[WORKING_MEMORY_L1]` turns as normal user/assistant messages.

### Graceful Degradation

If Ollama is unreachable or crashes, the worker writes a truncation note to L2 and continues. The main agent thread is **never** blocked or crashed by a compression failure. Set `fallback_truncate=False` to raise `CompressionError` instead.

---

## Configuration Reference

```python
from sawtooth_memory import ContextManagerConfig, OllamaConfig

config = ContextManagerConfig(
    soft_limit_tokens=3000,      # int: L1 size that triggers background compression
    hard_limit_tokens=6000,      # int: emergency synchronous truncation cap
    chunk_size=10,               # int: messages per compression batch
    tokenizer_model="gpt-4o",   # str: tiktoken model for local counting
    fallback_truncate=True,      # bool: truncate vs raise on Ollama failure

    ollama=OllamaConfig(
        base_url="http://localhost:11434",
        model="phi4",            # any Ollama model; phi4/llama3.2 work well
        timeout_seconds=90,
    ),
)
```

---

## Project Structure

```
sawtooth_memory/
├── __init__.py       # public API surface
├── config.py         # ContextManagerConfig, OllamaConfig
├── state.py          # Pydantic v2 schemas: all four memory tiers
├── monitor.py        # tiktoken-based local token counting
├── compressor.py     # Ollama async HTTP client + dual-extraction prompt
├── worker.py         # asyncio background queue + pipeline + state merger
└── middleware.py     # ContextManager: main public API

tests/
├── test_state.py
├── test_monitor.py
├── test_compressor.py
└── test_middleware.py
```

---

## Roadmap

- [ ] LangChain / LangGraph adapter
- [ ] AutoGen adapter
- [ ] Redis queue transport (for multi-process agents)
- [ ] Sliding importance scoring (weight recent tool results more heavily)
- [ ] Prometheus metrics endpoint
- [ ] TypeScript port

---
