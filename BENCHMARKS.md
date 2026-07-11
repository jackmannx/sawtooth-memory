# Sawtooth Memory: Performance Benchmarks

This document describes the benchmark suite for `sawtooth-memory`, including methodology, published results, and how to reproduce them locally.

---

## Benchmark Architecture

The suite is organized in four layers:

| Layer | Location | Purpose | CI |
|-------|----------|---------|-----|
| **Microbenchmarks** | `benchmarks/micro/` | Hot-path latency (TokenMonitor, NER, ledger, `add_message`, `build_prompt`) | Every PR |
| **Scenario benchmarks** | `benchmarks/scenarios/` | E2E latency methodology, recall suite, scale, storage | Every PR |
| **Harness** | `benchmarks/harness.py` | Comparative reports (Sawtooth vs blocking summary) | Every PR (mock) |
| **Live benchmarks** | `scripts/benchmark_memory.py` | Real Ollama GPU runs | Manual |

---

## Core Metrics

We track metrics separately to avoid conflating main-thread and background costs:

| Metric | What it measures |
|--------|------------------|
| **User-perceived turn latency (p95)** | Worst-case time the application thread spends in `add_message()` / `build_prompt()` per turn |
| **Per-turn blocked latency** | Time blocked in sequential summary memory on each `save_context()` call |
| **Background drain cost** | One-time cost to flush the compression worker at session end (`cm.stop()`) |
| **Final prompt token footprint** | Size of the compiled payload sent to the primary LLM |
| **Needle recall (%)** | Retention of injected facts across L1, L1.5 ledger, and L2 archive |

### Methodology

**Blocking baseline** emulates LangChain-style `ConversationSummaryMemory`: every turn blocks the main thread until Ollama finishes summarizing.

**Sawtooth** measures only main-thread time during active conversation. Compression runs asynchronously in a background worker. A separate `drain_ms` metric captures the one-time flush at session end.

This replaces an earlier approach that compared total session wall time, which unfairly penalized Sawtooth by including background worker drain in the same bucket as per-turn blocking.

---

## Published Results (Live Ollama)

**Run date:** 2026-07-11  
**Git SHA:** `5b31fa0`  
**Package version:** `0.2.1`

### Test Environment

| Setting | Value |
|---------|-------|
| **Host** | Linux x86_64 (Pop!_OS 6.17.9) |
| **GPU** | NVIDIA GeForce RTX 5060 Laptop (8 GB VRAM) |
| **LLM backend** | Ollama `phi4-mini` |
| **Conversation** | 10 turns / 20 messages, medium message size |
| **Sawtooth config** | `soft_limit_tokens=400`, `hard_limit_tokens=8000`, `chunk_size=4` |

### Performance Matrix

| Performance Metric | Blocking Summary Memory | Sawtooth Hierarchical | Notes |
| :--- | :--- | :--- | :--- |
| **User-perceived turn latency (p95)** | 24.3 seconds | **<0.1 ms** | What the app thread waits per turn |
| **Mean blocked per turn** | 7.8 seconds | **0.03 ms** | Average main-thread cost |
| **Per-turn blocked (p50)** | 4.0 seconds | **0.02 ms** | Median turn latency |
| **Per-turn blocked (max)** | 30.1 seconds | **0.09 ms** | Worst single `add_message()` |
| **Total main-thread blocked (10 turns)** | 78.2 seconds | **<2 ms** | Cumulative app freeze |
| **Session-end background drain** | — | **1.1 seconds** | One-time `cm.stop()` flush |
| **Final prompt tokens** | 563 | 866 | Sawtooth includes L1.5 ledger + archive tiers |
| **Golden needle recall** | 0% | **100%** | `txn_998877_alpha_omega` lost vs retained |
| **Needle suite recall (4 facts)** | n/a | **100%** | IDs, paths, URIs, ARNs all retained |

### Key Insights

**Why Sawtooth feels instant during conversation**

Blocking summary memory calls Ollama synchronously on every turn. On this hardware, that means 2–30 seconds of main-thread freeze per turn (mean 7.8s, p95 24.3s). Over 10 turns, the application thread is blocked for **78.2 seconds total**.

Sawtooth's `add_message()` returns in **~0.03 ms** on average. Compression happens in a background worker. The only Sawtooth-specific latency cost is a **1.1 second drain** at session end when pending compression is flushed.

**Token cost vs. recall trade-off**

Sawtooth's final prompt is larger (866 vs 563 tokens) because it preserves structured memory tiers — especially the L1.5 entity ledger. This is intentional: the extra tokens buy **guaranteed fact retention** that summary-only memory cannot provide.

**Recall accuracy**

The blocking baseline **lost** the golden needle (`txn_998877_alpha_omega`) after progressive summarization. Sawtooth retained all 4 injected facts (transaction ID, file path, URI, AWS ARN) across compression cycles.

---

## Running Benchmarks

### Setup

```bash
cd sawtooth-memory
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,langgraph]"
```

### Comparative harness (recommended)

**Mock mode** (no GPU, fast smoke test):

```bash
.venv/bin/python scripts/run_benchmarks.py harness --turns 10 --mode mock
```

**Live mode** (real Ollama, produces published numbers):

```bash
ollama pull phi4-mini
BENCHMARK_MODE=live .venv/bin/python scripts/benchmark_memory.py --turns 10
```

Results are written to `benchmarks/results/latest.json`. View them:

```bash
cat benchmarks/results/latest.json | python3 -m json.tool
```

### Full suite

```bash
.venv/bin/python scripts/run_benchmarks.py all
```

### Microbenchmarks only

```bash
.venv/bin/python scripts/run_benchmarks.py micro
```

With regression check against the committed baseline:

```bash
.venv/bin/pytest benchmarks/micro \
  --benchmark-only \
  --benchmark-compare=benchmarks/baselines/micro_baseline.json \
  --benchmark-compare-fail=mean:25%
```

### Scenario benchmarks (mock E2E)

```bash
.venv/bin/python scripts/run_benchmarks.py scenarios
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_MODE` | `mock` | `mock` or `live` |
| `BENCHMARK_LOCAL_MODEL` | `phi4-mini` | Ollama model for live runs |
| `BENCHMARK_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |

---

## Recall Needle Suite

The recall battery injects facts at known turns and verifies retention after compression:

| Category | Example value | Turn |
|----------|---------------|------|
| Custom ID | `txn_998877_alpha_omega` | 2 |
| File path | `/etc/nginx/sites-enabled/api.conf` | 4 |
| URI | `https://internal.corp/runbooks/inc-4421` | 6 |
| AWS ARN | `arn:aws:s3:us-east-1:123456789012:bucket/prod-data` | 8 |

Recall is scored across the compiled prompt, L1.5 entity ledger, and L2 archival narrative.

---

## CI

The [Benchmark Suite](.github/workflows/benchmark.yaml) workflow runs on every PR that touches `benchmarks/` or `sawtooth_memory/`:

- Microbenchmark regression gate (25% mean tolerance vs baseline)
- Scenario integration benchmarks
- Mock harness smoke test

---

## Updating Baselines

After intentional performance changes:

```bash
.venv/bin/pytest benchmarks/micro --benchmark-only --benchmark-save=baseline
cp .benchmarks/Linux-CPython-3.12-64bit/0001_baseline.json benchmarks/baselines/micro_baseline.json
```

Commit the updated `benchmarks/baselines/micro_baseline.json` with your PR.

When publishing new live GPU numbers, re-run the live harness and update the **Published Results** section above with the new `latest.json` output.
