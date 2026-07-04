# Sawtooth Memory

[![Automated Test Suite](https://github.com/jackmannx/sawtooth-memory/actions/workflows/test.yaml/badge.svg)](https://github.com/jackmannx/sawtooth-memory/actions/workflows/test.yaml)
[![PyPI version](https://badge.fury.io/py/sawtooth-memory.svg)](https://badge.fury.io/py/sawtooth-memory)
[![Python Support](https://img.shields.io/pypi/pyversions/sawtooth-memory.svg)](https://pypi.org/project/sawtooth-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A high-performance, asynchronous non-blocking hierarchical memory framework for LLM Agents.**

## The Problem

Standard LLM memory systems (like LangChain's `ConversationSummaryMemory`) process conversation history sequentially on the main application thread. Every time a user sends a message, the entire application freezes while the system waits for an LLM to generate a new historical summary. Furthermore, these summaries suffer from the "Lost in the Middle" hallucination effect, frequently deleting specific UUIDs, names, or rules to save tokens.

## The Solution

**Sawtooth Memory** eliminates this latency and data loss. It immediately stores the user's message and returns control to the application in milliseconds, offloading the heavy summarization to an asynchronous background worker. To prevent hallucinations, it extracts critical facts into an immutable ledger before summarizing.

---

## Architecture & Data Flow

### 1. The Non-Blocking Execution Model

```text
  Standard Memory (Blocking)            Sawtooth Memory (Async)
  ──────────────────────────            ───────────────────────

  [ Application ]                       [ Application ]
         │                                     │
         ▼                                     ▼
  [ Save Context ]                      [ ContextManager ]
         │                                     │
         ▼                                     ├───────────────────┐ (Instant Return)
  [ LLM Summarizes ]                           ▼                   ▼
  (App freezes for 5-10s)               [ Next User Turn ]  [ Background Worker ]
         │                                                         │
         ▼                                                         ▼
  [ Next User Turn ]                                        [ LLM Summarizes ]

```

### 2. The Hierarchical Memory Stack

When your agent is ready to respond, Sawtooth stitches together an optimized context payload from distinct layers, ensuring critical facts are never summarized away.

```text
    Agent Loop
        │
        ▼
┌─────────────────────┐
│   ContextManager    │
│  ┌───────────────┐  │
│  │ L0 System     │  │  immutable persona + tool schemas
│  │ L2 Archive    │  │  compressed narrative memory
│  │ L1.5 Entities │  │  exact IDs, rolling conflict history
│  │ L1 Working    │  │  recent raw conversation
│  └───────────────┘  │
└──────────┬──────────┘
           │
           ▼
     build_prompt() / get_compiled_prompt()
           │
           ▼
        LLM API

```

- **Phase 2 Update to L1.5:** The Entity Ledger now utilizes a rolling window history. Instead of overwriting older values, it preserves conflicts and automatically injects a `<key>__history` variable into the prompt so the LLM can see the chronological provenance of changing variables.

---

## Key Features

- **Zero-Latency Ingestion:** Messages are appended to L1 instantly. A local `tiktoken` monitor checks thresholds without making API calls.
- **Dual LLM Compression Backends:** Run compression locally via `OllamaCompressor` or in the cloud using `CloudCompressor` (with modular adapters for OpenAI, Anthropic, and Gemini).
- **Deterministic NER Engine:** A zero-latency local regex pipeline extracts UUIDs, file paths, and URIs _before_ the LLM sees the text, securely populating the Entity Ledger (L1.5) and overriding potential LLM hallucinations.
- **Turn-Based Batching & Debouncing:** Prevent background queue flooding using `max_unsummarized_turns` to trigger compression safely by turn count, alongside token limits.
- **Graceful Degradation:** If the system hits the `hard_limit_tokens` before the asynchronous background worker finishes a cycle, a fallback protocol forcefully truncates the oldest L1 messages on the main thread to prevent API crashes.

---

## Performance Benchmarks

By moving compression to the background, Sawtooth achieves massive latency reductions on the main thread while maintaining 100% recall accuracy.

**Local GPU Benchmark (NVIDIA RTX 5060 | Model: phi4-mini | 20-Message Conversation)**

| Performance Metric       | Standard Summary Memory | Sawtooth Hierarchical | Architectural Advantage        |
| ------------------------ | ----------------------- | --------------------- | ------------------------------ |
| **Main Thread Latency**  | 64.15 seconds           | **5.70 seconds**      | **11.3x Faster Execution**     |
| **Final Prompt Payload** | 506 tokens              | **454 tokens**        | **10% Lower Token Cost**       |
| **UUID / Fact Recall**   | Variable / Hallucinates | **100% Retained**     | **Guaranteed via L1.5 Ledger** |

For full methodology, cloud comparisons, and reproducibility steps, view our [Performance Benchmarks](BENCHMARKS.md).

---

## Installation

```bash
pip install sawtooth-memory

```

_Optional dependencies:_

```bash
# Cloud compression providers (install the SDK you use)
pip install langchain-openai langchain-anthropic langchain-google-genai

# LangChain message history adapter
pip install sawtooth-memory[langchain]

# LangGraph integration
pip install sawtooth-memory[langgraph]

# Distributed session storage
pip install sawtooth-memory[redis]

# Durable Postgres + pgvector storage
pip install sawtooth-memory[postgres]

# Everything
pip install sawtooth-memory[all]
```

---

## Quickstart (V2 Configuration)

The V2 configuration introduces dynamic validation, allowing you to set a single `background_model` parameter that automatically routes to the respective local or cloud backend. Cloud models (`gpt-*`, `claude-*`, `gemini-*`) read API keys from standard environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig

async def main():
    # V2 Simplified Configuration
    config = ContextManagerConfig(
        background_model="gpt-4o-mini",   # Auto-routes to CloudCompressor (or "phi4" for local Ollama)
        soft_limit_tokens=1000,           # Token threshold to trigger background compression
        hard_limit_tokens=2000,           # Emergency truncation limit
        max_unsummarized_turns=10,        # Turn-based batching threshold
        enable_deterministic_ner=True     # Enable local regex extraction for the Entity Ledger
    )

    async with ContextManager(system_prompt="You are a helpful assistant.", config=config) as cm:

        # Optional: Run a health check to verify backend routing and worker status
        await cm.health_check()

        # 1. Instantly ingest messages (Zero-latency on the main thread)
        await cm.add_message("user", "My transaction ID is txn_998877_alpha")
        await cm.add_message("assistant", "I have noted your transaction ID.")

        # 2. Build the optimized prompt to send to your main LLM
        prompt = await cm.build_prompt()
        print(prompt)

if __name__ == "__main__":
    asyncio.run(main())

```

For explicit cloud configuration without environment variables:

```python
from sawtooth_memory.config import CloudConfig, Provider

config = ContextManagerConfig(
    cloud=CloudConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        api_key="sk-...",
    ),
)
```

---

## Advanced Features

### 1. Deterministic NER (Named Entity Recognition)

By setting `enable_deterministic_ner=True`, Sawtooth intercepts incoming text and uses a fast regex engine to extract critical string identifiers directly into the Entity Ledger. You can also inject custom patterns:

```python
config = ContextManagerConfig(
    enable_deterministic_ner=True,
    custom_ner_patterns={
        "aws_arn": r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]+:\d{12}:[a-zA-Z0-9\-\_/]+"
    }
)

```

### 2. LangGraph Integration & ToolMessage Sanitization

Sawtooth provides a native `SawtoothLangGraphAdapter` to sync state seamlessly.

**V2 Safety Feature:** Strict cloud APIs (like Anthropic/OpenAI) will crash if a `ToolMessage` is sent without its parent `AIMessage` (the tool call request). The LangGraph adapter includes an advanced **3-pass sanitization logic** that automatically detects and drops orphaned `ToolMessage`s when their parent `AIMessage` has been compressed and evicted to L2 Archival Memory.

```python
from langgraph.graph import StateGraph
from sawtooth_memory.integrations.langgraph import SawtoothLangGraphAdapter

# Initialize the adapter with your Sawtooth ContextManager
adapter = SawtoothLangGraphAdapter(cm)

# Automatically syncs state, deduplicates message IDs, and sanitizes orphaned tools
sanitized_messages = await adapter.sync_and_sanitize(langgraph_state_messages)

```

### 3. Modern LangChain & LCEL Integration

Sawtooth provides a native, pure-Python adapter that fully implements LangChain's modern `BaseChatMessageHistory` interface. This allows you to drop Sawtooth directly into any LCEL Runnable or Agent executor, bringing background compression and deterministic NER to standard LangChain pipelines without blocking the main thread.

```python
from langchain_core.messages import HumanMessage
from sawtooth_memory.integrations.langchain_adapter import SawtoothChatMessageHistory

# Drop-in replacement for any LangChain memory module
history = SawtoothChatMessageHistory(
    system_prompt="You are a financial analyst.",
    config=config
)

history.add_message(HumanMessage(content="Analyze these Q3 numbers."))

# Automatically compiles the L0, L1.5, L2, and L1 tiers safely across thread boundaries
lc_messages = history.messages

```

### 4. Synchronous API Wrapper (Flask, Django, CLI)

If you are building in a traditional synchronous environment (like a standard Flask or Django view) where you cannot use asyncio or await, Sawtooth provides an enterprise-grade SawtoothSyncWrapper. It uses an AnyIO BlockingPortal to isolate the asynchronous background worker on a safe daemon thread, preventing event loop collisions while maintaining zero-latency writes.

```python
from sawtooth_memory.sync_wrapper import SawtoothSyncWrapper

def my_flask_route():
    # Use standard 'with' - no async required!
    with SawtoothSyncWrapper("You are a helpful assistant.", config=config) as memory:

        # 1. Instantly write to the background thread
        memory.add_message("user", "Hello world!")

        # 2. Safely read the compiled state machine
        prompt = memory.build_prompt()

        return prompt


```

### 5. Recall Explainability Traces

Sawtooth eliminates the "black-box" of agent memory by providing deterministic audit trails. You can query the memory system to see exactly why a fact was retained in the prompt.

```python
trace = cm.explain_prompt()

import json
print(json.dumps(trace, indent=2))

```

**Output:**

```json
{
  "l0_system": {
    "content": "You are a helpful assistant.",
    "origin": "Hardcoded System Initialization"
  },
  "l2_archival": {
    "content": "User provided transaction ID txn_998877_alpha.",
    "origin": "Background Ollama Compression (L1 -> L2)"
  },
  "l1_5_entities": [
    {
      "prompt_component": "[ENTITY_LEDGER_L1_5]",
      "entity_key": "user_transaction_id",
      "entity_value": "txn_998877_alpha",
      "origin": "Anchored via explicit tracking engine (Operation: insert) [Strategy: deterministic]"
    }
  ],
  "l1_working_messages": 2
}
```

### 6. Distributed Storage Backends (Horizontal Scaling)

By default, Sawtooth manages process state locally. For multi-container stateless applications (e.g., load-balanced FastAPI apps or Kubernetes pods), Sawtooth provides an abstract storage layer to decouple memory data from active server process memory RAM.

The `RedisStorageAdapter` serializes your state trees to high-speed JSON structures natively, allowing multiple distinct node pods to process background worker loops seamlessly without cross-session data overwrites.

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig
from sawtooth_memory.storage.redis_adapter import RedisStorageAdapter

async def main():
    # Initialize the high-speed distributed storage backend
    redis_storage = RedisStorageAdapter(
        redis_url="redis://localhost:6379/0",
        key_prefix="sawtooth:session:",
        ttl_seconds=86400  # Automatically expire inactive sessions after 24 hours
    )

    config = ContextManagerConfig(
        background_model="gpt-4o-mini",
        storage_adapter=redis_storage,
        session_id="user_session_994"  # Route state changes dynamically via custom keys
    )

    async with ContextManager(system_prompt="You are a cluster node agent.", config=config) as cm:
        await cm.add_message("user", "Save this cluster token: secret_pass_123")

        # Hydrates state directly across node instances instantly!
        prompt = await cm.build_prompt()
```

### 7. Semantic Vector L3 Archival Memory (Storage Layer)

Sawtooth can index evicted L1 conversation text into a **pgvector-backed L3 semantic archive** during background compression. Vectors are stored separately from the JSONB `MemoryState` payload to keep session snapshots lean.

**Important:** L3 retrieval is available via `search_semantic_archive()` but is **not yet injected into `build_prompt()`**. L2 narrative summaries remain the only archival content in the compiled prompt until RAG retrieval is wired in a future release.

Requirements:
- `PostgresStorageAdapter` with the PostgreSQL `vector` extension installed
- `enable_l3_semantic_storage=True` on `ContextManagerConfig`
- Matching `embedding_dimension` on both the adapter and config

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig
from sawtooth_memory.storage.postgres_adapter import PostgresStorageAdapter

async def main():
    postgres = PostgresStorageAdapter(
        dsn="postgresql://user:pass@localhost:5432/sawtooth",
        embedding_dimension=384,
    )

    config = ContextManagerConfig(
        background_model="gpt-4o-mini",
        storage_adapter=postgres,
        session_id="user_session_994",
        enable_l3_semantic_storage=True,
        embedding_backend="hash",       # "openai" for production embeddings
        embedding_dimension=384,
        l3_chunk_max_chars=2000,
    )

    async with ContextManager(system_prompt="You are a cluster node agent.", config=config) as cm:
        await cm.add_message("user", "Router firmware is v2.4.1 and drops packets nightly.")
        # After background compression, evicted L1 text is chunked, embedded, and stored in L3.

        # Storage-layer retrieval (not yet wired into build_prompt):
        matches = await cm.search_semantic_archive("router firmware packets", top_k=3)
        for chunk in matches:
            print(f"[{chunk.similarity:.2f}] {chunk.text}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Roadmap

- **Phase 1: Core Architecture**
- [x] L1/L2 Hierarchical Buffer
- [x] Asynchronous Background Worker
- [x] Local (Ollama) & Cloud compatibility

- **Phase 2: Observability & Stability**
- [x] EventBus Subsystem & JSONL Auditing Journal
- [x] Explainability Traces & Performance Benchmarking Harness
- [x] Deterministic NER Engine
- [x] LangGraph ToolMessage Sanitization
- [x] Turn-Based Batching & Debouncing
- [x] Modern LangChain (LCEL) History Adapter
- [x] AnyIO Synchronous Blocking Portal (Flask/Django Support)

- **Phase 3: Advanced Architectures**
- [x] Redis Distributed Storage Adapter (High-Speed Session Pooling)
- [x] Postgres Storage Adapter (Persistent Relational Cache with pgvector)
- [x] Multi-Agent Memory Pooling (Shared contextual state)
- [x] Semantic Vector L3 Archival Memory (RAG integration — storage layer complete; retrieval not yet wired into `build_prompt()`)

---

## Contributing

We welcome pull requests. See our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to run the test suite and ensure code quality.

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
