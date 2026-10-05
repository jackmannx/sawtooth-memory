# Bug Report: Sawtooth Memory

Findings from a manual bug-hunting pass over `sawtooth_memory/` (as of commit
`0b675db`, v0.3.0). Three exploration sweeps (concurrency/orchestration,
compression/entity-ledger, storage/integrations) produced 31 candidate
issues; every claim below was independently re-verified by reading the cited
source before being included here. Claims that didn't hold up are listed in
[Investigated and Ruled Out](#investigated-and-ruled-out) so they aren't
re-flagged in a future pass.

Findings 1 and 2 have since been fixed (see each section's "Fix applied"
note); the rest are still open analysis, no code changed for those.

---

## Summary

| # | Severity | Area | One-line description | Status |
|---|----------|------|------------------------|--------|
| 1 | Critical | Multi-agent pooling | Pool sync is a read-modify-write race; concurrent agents can clobber each other's deltas | **Fixed** |
| 2 | High | Multi-agent pooling | Entities from crushed tool observations never propagate to the shared pool | **Fixed** |
| 3 | Medium | Observability | Unexpected worker exceptions produce no event/journal entry |
| 4 | Medium | DTE accounting | Consolidation's debt-counter reset can race a concurrent fold |
| 5 | Medium-High | LangChain integration | Plain adapter silently drops `ToolMessage`/`SystemMessage` |
| 6 | Medium | L3 semantic retrieval | Partial embedding responses silently discard a whole indexing batch |
| 7 | Low-Medium | Sync wrapper | Background thread leaks if shutdown raises |
| 8 | Medium | Thread safety | `SyncContextManager` not safe to share across threads despite WSGI framing |

---

## 1. Multi-agent pool sync has a read-modify-write race

**Severity:** Critical
**Files:** `sawtooth_memory/middleware.py:425-471` (`_sync_pool_state_from_storage`, `_sync_fold_to_pool`), mirrored in `sawtooth_memory/sync_manager.py:506-553`; lock mechanism in `sawtooth_memory/storage/postgres_adapter.py:201-230`

**The code:**
```python
# middleware.py:453-468 (_sync_fold_to_pool)
pool_state = await adapter.load_pool_state(pool_id)      # READ (own transaction)
...
apply_fold_delta_to_pool(..., shared_entities=shared_entities, shared_archive=shared_archive)  # MODIFY (in Python memory)
await adapter.save_pool_state(pool_id, shared_entities, shared_archive)  # WRITE (own transaction, SELECT...FOR UPDATE)
```

**Failure scenario:** Agent A and Agent B share a pool. Both call
`_sync_fold_to_pool` around the same time:
1. A loads the pool (entities = `{X: "v1"}`).
2. B loads the pool (entities = `{X: "v1"}` — A hasn't saved yet).
3. A applies its fold delta locally (`{X: "v1", Y: "a"}`) and saves.
4. B applies its fold delta locally, built from its *stale* read
   (`{X: "v1", Z: "b"}`), and saves — overwriting A's `Y: "a"`.

**Why it's wrong:** `save_pool_state`'s `SELECT ... FOR UPDATE`
(`postgres_adapter.py:211-217`) only locks the row immediately before the
final `INSERT ... ON CONFLICT DO UPDATE`. It does nothing to protect the
*read* that happened in a separate, already-committed transaction moments
earlier. The CHANGELOG for v0.2.1 claims "Pool L1.5/L2 merge-on-sync (no
clobber)" was fixed — this shows the fix covers the storage layer's write
path but not the three-step load→modify→save sequence at the call sites.

**Suggested fix direction:** Either (a) wrap the full load-modify-save
sequence in a single transaction with the row lock acquired at the start of
the read, or (b) move the merge logic into the storage layer so
`save_pool_state` takes a *delta* (not a fully-resolved new state) and merges
it server-side under its own lock, eliminating the client-side stale-read
window entirely. (b) is more robust against future call sites making the
same mistake.

**Fix applied:** Took direction (b). Added `BaseStorageAdapter.merge_pool_state(pool_id, merge_fn)`
(`storage/base.py`) — callers pass a plain function that mutates an
`EntityLedger`/`ArchivalMemory` in place; the adapter owns holding a lock
across the full read-modify-write. The base class ships a non-atomic
load-then-save default (so third-party adapters that only implement
`load_pool_state`/`save_pool_state` keep working unmodified — adding a new
*abstract* method would have been a breaking change for any custom
adapter). `PostgresStorageAdapter.merge_pool_state` holds a single
transaction with `SELECT ... FOR UPDATE` spanning the read through the
write (`storage/postgres_adapter.py`); `RedisStorageAdapter.merge_pool_state`
uses WATCH/MULTI/EXEC optimistic locking with retry, since Redis has no
row-lock equivalent (`storage/redis_adapter.py`). All four call sites that
used to do their own load→modify→save now go through `merge_pool_state`:
`middleware.py::_sync_fold_to_pool`, `worker.py::_finalize_fold`,
`worker.py`'s consolidate branch of `on_success`, and
`sync_manager.py`'s `_sync_fold_to_pool`/`_compact_pool_folds`. Regression
test: `tests/test_multi_agent_pooling.py::test_pool_merge_default_is_racy_but_locked_adapter_is_not`
(proves the race exists under a forced interleave with the unlocked default,
and is closed once a lock spans the full sequence, matching what the two
shipped adapters now do).

---

## 2. Crushed tool-observation entities never reach the shared pool

**Severity:** High
**Files:** `sawtooth_memory/middleware.py:339-357` (`add_message`), `sawtooth_memory/fold_unit.py:41-45` (`create_fold_unit`), `sawtooth_memory/dte_runtime.py:112-126` (`apply_fold_delta_to_pool`)

**The code path:**
```python
# middleware.py:339-357
stored_content = apply_observation_crush(role, content, ...)   # crushes large tool output
msg = Message(role=role, content=stored_content)                # Message now holds the CRUSHED text
self._state.l1_working.append(msg)
...
await self._scan_message_entities(content)   # uses ORIGINAL content — entity lands in LOCAL ledger ✓
```
```python
# fold_unit.py:41-45 (runs later, when msg is evicted)
text = messages_text if messages_text is not None else messages_to_text(messages)  # messages[i].content is the CRUSHED text
extraction = ner_pipeline.extract_with_metadata(text)   # finds nothing — the ID isn't in the crushed placeholder
entity_keys = tuple(sorted(extraction.entities))          # empty for this entity
```
```python
# dte_runtime.py:122-126
for key in entity_keys:              # entity_keys came from the crushed-text NER pass above
    value = local_entities.get_latest(key)
    if value is not None:
        shared_entities.upsert({key: value})   # entity missing from entity_keys → never reaches the pool
```

**Failure scenario:** A tool returns a large JSON blob containing
`connection_id: "conn-123-abc"`. `add_message` crushes it to
`"[OBSERVATION_CRUSHED id=obs_abc123 ...]"` before storing it, but the
ingest-time scan (which runs on the *original*, uncrushed `content` argument)
still captures `conn-123-abc` into the local L1.5 ledger — so far so good.
Later, when that message is evicted and folded, `create_fold_unit` re-runs
NER over the *already-crushed* stored text and finds nothing, so
`conn-123-abc` is absent from `entity_keys`. `apply_fold_delta_to_pool` only
pushes keys listed in `entity_keys` to the shared pool, so other agents in
the pool never see this entity — even though it's sitting right there in the
originating agent's own ledger.

**Why it's wrong:** This isn't data loss from the *local* agent's
perspective (the README's "100% recall" claim is about the local ledger,
which is fine here), but it silently breaks the "Multi-Agent Memory Pooling"
feature for any entity that arrived via a crushed tool observation.

**Suggested fix direction:** Have `create_fold_unit` accept the set of
entity keys already known to be in the local ledger for the messages being
folded (from the ingest-time scan), and union them into `entity_keys` rather
than relying solely on a fresh NER pass over potentially-crushed text.

**Fix applied:** Added `Message.ingested_entity_keys: list[str]` (`state.py`).
`_scan_message_entities` (both `middleware.py` and `sync_manager.py`, which
each run their own ingest-time scan) now tags the keys it found directly
onto the `Message` it just scanned. `create_fold_unit` (`fold_unit.py`)
unions every evicted message's `ingested_entity_keys` into `entity_keys`
alongside the fresh NER pass's results, so an entity found only on the
original (pre-crush) content still ends up in `entity_keys` — and therefore
still reaches `apply_fold_delta_to_pool`. Regression test:
`tests/test_multi_agent_pooling.py::test_crushed_observation_entity_still_reaches_pool`
(a UUID that only appears in a tool observation large enough to get crushed
away is confirmed present in the shared pool after the fold cycle).

---

## 3. Unexpected worker exceptions leave zero audit trail

**Severity:** Medium
**File:** `sawtooth_memory/worker.py:184-200` (`_loop`), contrast with `worker.py:297-319` (`on_failure`)

**The code:**
```python
# worker.py:184-200
while True:
    task = await self._queue.get()
    ...
    try:
        await self._process(task)
        self._processed += 1
    except Exception as exc:          # catches EVERYTHING, including bugs unrelated to LLM calls
        self._failed += 1
        logger.error(f"CompressionWorker: unhandled error processing task: {exc}", exc_info=True)
        # no event emitted, no journal entry written
    finally:
        self._queue.task_done()
```

**Why it's wrong:** The *expected*-failure path (an LLM/compression call
failing) is handled correctly — `on_failure` (`worker.py:297-319`) emits a
`CompressionCycleFailedEvent`, which the journal handler
(`events/handlers.py`) picks up. But if anything else throws inside
`_process` (a bug in state mutation, a storage error outside the handled
try/except blocks, etc.), it's caught by this outer generic handler instead,
which only logs — no `CompressionCycleFailedEvent`, no journal write. Since
`task.state` is mutated by reference and may already be partially updated
before the exception, this is exactly the scenario where an audit trail
matters most, and it's the one case where there isn't one.

**Suggested fix direction:** In the `except Exception` branch of `_loop`,
emit a `CompressionCycleFailedEvent` (with `error_type="unexpected"` or
similar) so unexpected failures are journaled the same way expected ones are.

---

## 4. Consolidation's debt-counter reset can race a concurrent fold

**Severity:** Medium
**File:** `sawtooth_memory/worker.py:217-248` (`on_success`, "consolidate" branch)

**The code:**
```python
# worker.py:221-248
if task.task_kind == "consolidate":
    task.state.l2_archival.narrative = remove_fold_lines(task.state.l2_archival.narrative)
    if self._storage_adapter and self._pool_id:
        try:
            pool_state = await self._storage_adapter.load_pool_state(self._pool_id)   # AWAIT #1
            ...
            await self._storage_adapter.save_pool_state(...)                           # AWAIT #2
        except Exception as exc:
            logger.warning(...)
    task.state.dte.narrative_debt_tokens = 0        # hard reset, not -=
    task.state.dte.folds_since_narrative = 0        # hard reset, not -=
    ...
```

**Failure scenario:** Between the two `await` points above, the event loop
can switch to the main coroutine processing `add_message()`. If that call
triggers a new fold (`middleware.py`'s `_trigger_compression` /
`_force_truncate` → `fold_unit.create_fold_unit`, which does
`state.dte.narrative_debt_tokens += tokens_evicted` and
`folds_since_narrative += 1`), those increments land on the *same* shared
`state.dte` object. When `on_success` resumes after its second `await` and
unconditionally zeroes both counters, the new fold's contribution is wiped
out.

**Why it's wrong:** No narrative text is lost — the new fold's stub is still
appended to `l2_archival.narrative`. But the debt/count bookkeeping that
`prepare_consolidation` (`dte_runtime.py`) uses to decide whether the *next*
consolidation should fire now silently under-counts, which can delay
consolidation past the point where a real LLM summarization pass should have
run.

**Suggested fix direction:** Snapshot the debt/count values being
consolidated *before* the awaits, and decrement by that snapshot rather than
hard-resetting to 0 — so any contributions from folds that landed during the
await window are preserved for the next cycle.

---

## 5. The plain LangChain adapter silently drops `ToolMessage`/`SystemMessage`

**Severity:** Medium-High
**File:** `sawtooth_memory/integrations/langchain_adapter.py:87-95`

**The code:**
```python
def add_message(self, message: BaseMessage) -> None:
    self.init_portal()
    assert self._sync_wrapper is not None
    if isinstance(message, HumanMessage) or message.type == "human":
        self._sync_wrapper.add_message("user", str(message.content))
    elif isinstance(message, AIMessage) or message.type == "ai":
        self._sync_wrapper.add_message("assistant", str(message.content))
    # no branch, no else, no log for ToolMessage / SystemMessage
```

**Failure scenario:** A LangChain tool-calling agent produces an
`AIMessage` with `tool_calls`, then appends the corresponding `ToolMessage`
(the tool's result) via `history.add_message(ToolMessage(...))`. That call
is a silent no-op. The next `build_prompt()` reconstructs history without
the tool's result, so the LLM is asked to continue a tool-call conversation
with the result of that call missing — the LLM may hallucinate what the
tool returned or get confused about conversation state.

**Why it's wrong:** This is a real regression relative to the dedicated
LangGraph adapter (`integrations/langgraph/adapter.py`), which explicitly
maps `SystemMessage`/`ToolMessage` and even runs a sanitization pass for
orphaned tool messages. The plain `BaseChatMessageHistory` adapter has none
of that.

**Suggested fix direction:** Add `SystemMessage` → `"system"` and
`ToolMessage` → `"tool"` branches (mirroring the mapping already defined in
the LangGraph adapter), and either support a `"tool"` role end-to-end or
raise/log clearly if it's genuinely unsupported — don't drop silently.

---

## 6. Partial embedding responses silently discard a whole indexing batch

**Severity:** Medium
**Files:** `sawtooth_memory/embeddings/openai.py:70-76`, `sawtooth_memory/l3_indexer.py:131-134`, `sawtooth_memory/worker.py:408-418` (`_index_l3_semantic`)

**The code:**
```python
# embeddings/openai.py:70-76
ordered: list[list[float] | None] = [None] * len(texts)
for item in payload.get("data", []):
    idx = item.get("index")
    if idx is not None and 0 <= idx < len(texts):
        ordered[idx] = item.get("embedding", [])
return [vec if vec is not None else [] for vec in ordered]   # missing index → empty vector, no error
```
```python
# l3_indexer.py:131-134
embeddings = await self._embedder.embed(chunks)
if len(embeddings) != len(chunks):      # only checks COUNT, not that each vector is well-formed
    raise ...
```
```python
# worker.py:408-418
try:
    chunks_indexed = await indexer.index(self._session_id, messages_text, state)
except Exception as exc:
    logger.warning("CompressionWorker: L3 semantic indexing failed (%s).", exc, exc_info=True)
    return 0        # the whole batch silently fails
```

**Failure scenario:** OpenAI's embeddings endpoint returns fewer `data`
entries than requested (rate limiting, transient glitch) without the HTTP
call itself failing. `openai.py` pads the missing slot with `[]` instead of
raising. `l3_indexer.py`'s count check passes (the list is the right
*length*), so the malformed vector proceeds to the pgvector insert, which
fails on a dimension mismatch. That exception propagates up and is caught by
`worker.py`'s broad `except Exception`, which discards the *entire* batch
(not just the one bad chunk) with only a warning log — no retry, no partial
success, nothing surfaced to the caller. L3 semantic recall for that
conversation window silently degrades.

**Suggested fix direction:** Raise in `openai.py` when an index is missing
from the response instead of padding with `[]`, so the failure is explicit
and close to its source. Separately, consider making `_index_l3_semantic`
retry transient embedding failures once before giving up, since this is
exactly the kind of error (rate limit, brief outage) that a retry would
usually resolve.

---

## 7. `SawtoothSyncWrapper` leaks its background thread on a shutdown error

**Severity:** Low-Medium
**File:** `sawtooth_memory/sync_wrapper.py:80-91` (`__exit__`)

**The code:**
```python
def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    cm = self._cm
    portal = self._portal
    if cm and portal:
        portal.call(cm.stop)              # if this raises, execution never reaches below
    if self._portal_ctx:
        self._portal_ctx.__exit__(exc_type, exc_val, exc_tb)   # shuts down the background thread
        self._portal = None
        self._cm = None
```

**Failure scenario:** `cm.stop()` raises — e.g. a storage adapter times out
while flushing final state during shutdown. The exception propagates straight
out of `__exit__` immediately after `portal.call(cm.stop)`, so
`self._portal_ctx.__exit__(...)` is never reached. That call is what tears
down the dedicated AnyIO background event-loop thread, so it leaks for the
remaining life of the process.

**Suggested fix direction:** Wrap `portal.call(cm.stop)` in
`try`/`finally`, with the portal teardown in `finally`, so the background
thread is always shut down regardless of whether `stop()` succeeded.

---

## 8. `SyncContextManager` isn't safe to share across threads despite its own WSGI framing

**Severity:** Medium
**Files:** `sawtooth_memory/sync_manager.py` (module docstring + class), `sawtooth_memory/state.py:140-158` (`EntityLedger.upsert`)

*(Found during verification of other claims — not in the original three
exploration reports.)*

**The context:** `sync_manager.py`'s module docstring: *"Sync-native
ContextManager for scripts and WSGI applications. Runs compression inline on
the calling thread. No asyncio event loop, background worker, or AnyIO
blocking portal is required."* WSGI request handling is inherently
multi-threaded (most WSGI servers use a thread pool). Because this class does
all its work directly on the calling thread with no event loop indirection,
there is nothing serializing access if the *same instance* is shared across
worker threads (e.g., one `SyncContextManager` per pooled session, reused by
whichever thread handles the next request for that session).

**The code:**
```python
# state.py:140-158 (EntityLedger.upsert)
for key, value in new_entities.items():
    ...
    else:
        history = self.entities[key]
        if history and history[-1] == value:   # READ
            continue
        history.append(value)                   # WRITE — two separate statements, not atomic as a pair
        if len(history) > self.max_history_per_key:
            self.entities[key] = history[-self.max_history_per_key:]
```

**Failure scenario:** Two threads both handling requests for the same
pooled session call `add_message()` concurrently on the same
`SyncContextManager` instance. CPython's GIL can switch threads between
these two statements: both threads can read `history[-1]` and find it
doesn't match their incoming value *before* either has appended, so the
"skip duplicate" dedup guarantee is lost and the entry is appended twice (or,
depending on interleaving, one thread's update can be interleaved in a way
that doesn't reflect either caller's intent cleanly).

**Why it's wrong:** This directly contradicts the class's own documented use
case. Nothing in the docs tells an integrator whether one instance must be
confined to a single thread/request, or whether the class is expected to
be safe to share — and as written, it isn't.

**Suggested fix direction:** Either document explicitly that one
`SyncContextManager` instance must not be shared across threads (one per
request/thread, with pooling happening at the storage layer instead), or add
a `threading.Lock` around the mutating methods if cross-thread sharing is
meant to be supported.

---

## Investigated and Ruled Out

These were raised by the initial exploration sweeps but did not hold up under
direct source verification. Recorded here so they aren't re-flagged later.

- **"Debounce lock released too early causes duplicate/queue-flooding
  compression."** (`middleware.py:781`) — This is working as designed for
  DTE mode: the expensive work (fold creation, entity extraction, narrative
  stub) already happens synchronously *before* the debounce lock is
  released; only cheap finalize work (L3 indexing, pool sync) is deferred to
  the worker. The worker also processes its queue strictly sequentially
  regardless of how many tasks are enqueued, so there's no duplicate-work or
  corruption risk — just, at most, a backlog.

- **"Race condition between `_trigger_compression` and `_force_truncate`
  corrupting shared state."** — The hard-limit and soft-limit checks in
  `add_message` are `if`/`elif`, so only one of the two can run per message;
  they can't interleave within a single call. The more specific, real
  version of this concern is captured as finding #4 above (the consolidation
  debt-counter race), which *does* hold up because it spans actual `await`
  points.

- **"EntityLedger concurrent-mutation race in the async `ContextManager`."**
  — `EntityLedger.upsert()` contains no `await`, and the async
  `ContextManager`/worker both run as tasks on a single cooperatively
  scheduled event loop. A synchronous method with no internal yield point
  cannot be preempted mid-execution by another task on the same loop, so
  there's no race here. (The thread-level version of this same concern, for
  `SyncContextManager` specifically, is real — see finding #8.)

- **"Postgres FK constraint violation on vector upsert."**
  (`postgres_adapter.py:253-270`) — The session-row-ensure `INSERT ... ON
  CONFLICT DO NOTHING` and the vector batch insert run inside the *same*
  transaction, so Postgres guarantees the parent row exists before the
  child insert is evaluated; there's no violation window. Minor leftover
  note, not a bug: if L3 indexing runs before a session's first
  `save_state()`, it leaves a `state_payload='{}'` stub row — worth
  confirming `MemoryState` hydrates an empty payload gracefully, but not
  worth tracking as a defect on its own.

- **`SawtoothSyncWrapper` portal lifecycle on exception (general version).**
  — The sync-facing proxy methods (`add_message`, `pin_entity`,
  `build_prompt`, etc.) all funnel through `portal.call(...)`, which
  serializes execution onto the single background event-loop thread — so
  these are thread-safe as designed. The one real gap here is the
  `__exit__` shutdown path specifically, captured as finding #7.

- **Unshielded `asyncio.create_task()` calls for event emission** (multiple
  sites in `middleware.py`, `worker.py`, `monitor.py`) — A real, known
  asyncio footgun: a task with no stored reference is technically eligible
  for garbage collection before it completes, per the asyncio docs. In
  practice these tasks resolve almost immediately (a single `emit()` call),
  making this low-probability. Worth a cheap defensive fix later (store
  references in a set, the same pattern already used in
  `events/bus.py:26`), but not severe enough to prioritize over the findings
  above.
