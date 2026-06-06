
# Sawtooth Memory

[![Automated Test Suite](https://github.com/HtooTayZa/sawtooth-memory/actions/workflows/test.yaml/badge.svg)](https://github.com/HtooTayZa/sawtooth-memory/actions/workflows/test.yaml)
[![PyPI version](https://badge.fury.io/py/sawtooth-memory.svg)](https://badge.fury.io/py/sawtooth-memory)
[![Python Support](https://img.shields.io/pypi/pyversions/sawtooth-memory.svg)](https://pypi.org/project/sawtooth-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A high-performance, non-blocking hierarchical memory framework for LLM Agents.**

## The Problem
Standard LLM memory systems (like LangChain's `ConversationSummaryMemory`) process conversation history sequentially on the main application thread. Every time a user sends a message, the entire application freezes while the system waits for an LLM to generate a new historical summary. Furthermore, these summaries suffer from the "Lost in the Middle" hallucination effect, frequently deleting specific UUIDs, names, or rules to save tokens.

## The Solution
**Sawtooth Memory** eliminates this latency and data loss. It immediately stores the user's message and returns control to the application in milliseconds, offloading the heavy summarization to an asynchronous background worker. To prevent hallucinations, it extracts critical facts into an immutable ledger before summarizing.

---
## Documentation

For deep architectural deep-dives, comprehensive API specifications, and advanced lifecycle configurations, please refer to the official documentation:

[View Detailed Architecture & API Reference (DOCUMENTATION.md)](DOCUMENTATION.md)

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

---

## Performance Benchmarks

By moving compression to the background, Sawtooth achieves massive latency reductions on the main thread while maintaining 100% recall accuracy.

**Local GPU Benchmark (NVIDIA RTX 5060 | Model: phi4-mini | 20-Message Conversation)**

| Performance Metric | Standard Summary Memory | Sawtooth Hierarchical | Architectural Advantage |
| --- | --- | --- | --- |
| **Main Thread Latency** | 64.15 seconds | **5.70 seconds** | **11.3x Faster Execution** |
| **Final Prompt Payload** | 506 tokens | **454 tokens** | **10% Lower Token Cost** |
| **UUID / Fact Recall** | Variable / Hallucinates | **100% Retained** | **Guaranteed via L1.5 Ledger** |

For full methodology, cloud comparisons, and reproducibility steps, view our [Read the Performance Benchmarks](BENCHMARKS.md).

---

## Installation

```bash
pip install sawtooth-memory

```

*Optional dependencies for cloud providers:*

```bash
pip install langchain-openai langchain-anthropic langchain-google-genai

```

---

## Quickstart

### 1. The Standard Agent Loop

Initialize the `ContextManager` and let the background worker handle the heavy lifting. Sawtooth is universally compatible with local air-gapped models (Ollama) and cloud APIs.

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig
from sawtooth_memory.config import OllamaConfig

async def main():
    config = ContextManagerConfig(
        soft_limit_tokens=1000,
        hard_limit_tokens=2000,
        ollama=OllamaConfig(base_url="http://localhost:11434", model="phi4")
    )

    async with ContextManager(system_prompt="You are a helpful assistant.", config=config) as cm:

        # 1. Instantly ingest messages (Main thread is never blocked)
        await cm.add_message("user", "My transaction ID is txn_998877_alpha")
        await cm.add_message("assistant", "I have noted your transaction ID.")

        # 2. Build the optimized prompt to send to your main LLM
        prompt = cm.build_prompt()
        print(prompt)

if __name__ == "__main__":
    asyncio.run(main())

```

### 2. Recall Explainability Traces

Sawtooth eliminates the "black-box" of agent memory by providing deterministic audit trails. You can query the memory system to see exactly why a fact was retained in the prompt.

```python
trace = cm.explain_prompt()

import json
print(json.dumps(trace, indent=2))

```

**Output:**

```json
{
  "system_prompt": "You are a helpful assistant.",
  "l2_summary_lineage": [
    "User initiated troubleshooting for router.",
    "User provided MAC address."
  ],
  "l1_5_entities": [
    {
      "key": "user_transaction_id",
      "value": "txn_998877_alpha",
      "origin": "Anchored via L1.5 explicit instruction"
    }
  ],
  "l1_active_messages": 4,
  "total_tokens": 342
}

```

### 3. Integrations: LangGraph

Sawtooth provides a native `SawtoothMemorySaver` adapter, acting as a drop-in checkpointer replacement for LangGraph architectures.

```python
from langgraph.graph import StateGraph
from sawtooth_memory.integrations.langgraph import SawtoothMemorySaver

graph_builder = StateGraph(State)
# ... add nodes and edges ...

memory_saver = SawtoothMemorySaver(cm)
graph = graph_builder.compile(checkpointer=memory_saver)

```

---

## Roadmap

* [x] **Phase 1: Core Architecture**
* [x] L1/L2 Hierarchical Buffer
* [x] Asynchronous Background Worker
* [x] Local (Ollama) & Cloud compatibility


* [x] **Phase 2: Observability & Telemetry**
* [x] EventBus Subsystem
* [x] Explainability Traces
* [x] Persistent JSONL Auditing Journal
* [x] Performance Benchmarking Harness


* [ ] **Phase 3: Advanced Architectures (Up Next)**
* [ ] Multi-Agent Memory Pooling (Shared contextual state)
* [ ] Semantic Vector L3 Archival Memory (RAG integration)
* [ ] Redis/Postgres Adapter for Distributed Deployments



---

## Contributing

We welcome pull requests. See our [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) for guidelines on how to run the test suite and ensure code quality.

---
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---
