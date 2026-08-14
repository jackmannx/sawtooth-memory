# Flagship Demo: Naive Memory vs. Sawtooth

A 40-turn production-incident coding-agent session, replayed through two
memories under the same token budget:

1. **`NaiveMemory`** — a flat message list, FIFO-truncated once it exceeds
   the budget. This is the default shape of most hand-rolled agent memory.
2. **Sawtooth's `SyncContextManager`** — DTE compression mode, which folds
   evicted messages into a deterministic entity ledger instead of just
   dropping them.

Four facts are planted in the first few turns (a production DB connection
ID, an incident ticket, a file path, a commit hash) and then the agent is
asked to recall them after ~35 turns of unrelated debugging chatter has
pushed those early turns out of the window.

No API key, no local model, and no network access required — DTE folding
is fully deterministic, so this runs the same way every time.

## Run it

```bash
pip install -e ".[dev]"   # or: pip install sawtooth-memory
python examples/flagship_demo/run_demo.py
```

## What you'll see

The naive memory drops whichever planted facts fell outside its window —
in this scenario, the connection ID and ticket number, since they were
mentioned only in the earliest turns. Sawtooth keeps all four, exactly,
because they were extracted into the L1.5 entity ledger before their
source messages were evicted.

## Files

- `scenario.py` — the fixed 40-turn transcript and the facts to recall.
- `naive_baseline.py` — the flat-list, FIFO-truncated baseline memory.
- `run_demo.py` — runs the scenario through both memories and reports recall.

## What this demo is (and isn't) proving

This demo isolates **fact retention under a fixed token budget**, not
latency. For real latency numbers (blocking LLM summarization vs.
Sawtooth's non-blocking background worker, measured against a live Ollama
backend), see [`BENCHMARKS.md`](../../BENCHMARKS.md) at the repo root.
