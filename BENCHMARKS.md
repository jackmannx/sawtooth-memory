# Sawtooth Memory: Performance Benchmarks

This document describes the rigorous benchmark suite for `sawtooth-memory`, including methodology, metrics, and how to reproduce results locally.

---

## Benchmark Architecture

The suite is organized in four layers:

| Layer | Location | Purpose | CI |
|-------|----------|---------|-----|
| **Microbenchmarks** | `benchmarks/micro/` | Hot-path latency (TokenMonitor, NER, ledger, `add_message`, `build_prompt`) | Every PR |
| **Scenario benchmarks** | `benchmarks/scenarios/` | E2E latency methodology, recall suite, scale, storage | Every PR |
| **Harness** | `benchmarks/harness.py` | Comparative reports (Sawtooth vs blocking summary) | Every PR (mock) |
| **Live benchmarks** | `scripts/benchmark_memory.py` | Real Ollama GPU/API runs | Manual |

---

## Core Metrics

We track metrics separately to avoid conflating main-thread and background costs:

1. **User-perceived turn latency (p95)** — Time blocked on `add_message()` / `build_prompt()` per turn
2. **Per-turn blocked latency (blocking baseline)** — Time blocked in sequential summary memory
3. **Background drain cost** — Time to flush the compression worker at session end (`cm.stop()`)
4. **Final prompt token footprint** — Compiled payload size sent to the primary LLM
5. **Needle recall (%)** — Retention of injected facts across L1, L1.5 ledger, and L2 archive

### Methodology correction

Earlier benchmarks compared **total session wall time**, which unfairly included background worker drain for Sawtooth while counting per-turn blocking for LangChain-style memory.

The harness now reports **user-perceived turn p95** for apples-to-apples comparison of what the application thread experiences during active conversation.

---

## Running Benchmarks

### Full suite

```bash
pip install -e ".[dev,langgraph]"
python scripts/run_benchmarks.py all
```

### Microbenchmarks only

```bash
python scripts/run_benchmarks.py micro
```

With regression check against the committed baseline:

```bash
pytest benchmarks/micro \
  --benchmark-only \
  --benchmark-compare=benchmarks/baselines/micro_baseline.json \
  --benchmark-compare-fail=mean:25%
```

### Scenario benchmarks (mock E2E)

```bash
python scripts/run_benchmarks.py scenarios
```

### Comparative harness (mock — no GPU required)

```bash
python scripts/run_benchmarks.py harness --turns 10 --mode mock
```

Output is written to `benchmarks/results/latest.json`.

### Live Ollama benchmark (requires local GPU + Ollama)

```bash
ollama pull phi4-mini
BENCHMARK_MODE=live python scripts/benchmark_memory.py --turns 10
```

Environment variables:

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

## Example Results (Mock Harness)

Typical mock harness output on CI hardware:

| Metric | Blocking Summary | Sawtooth |
|--------|------------------|----------|
| User-perceived turn p95 | ~5 ms (simulated) | **<1 ms** |
| Needle suite recall | n/a | **100%** |

Live Ollama results vary by GPU and model. Re-run locally and commit updated numbers to this document when publishing marketing benchmarks.

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
pytest benchmarks/micro --benchmark-only --benchmark-save=baseline
cp .benchmarks/Linux-CPython-3.12-64bit/0001_baseline.json benchmarks/baselines/micro_baseline.json
```

Commit the updated `benchmarks/baselines/micro_baseline.json` with your PR.
