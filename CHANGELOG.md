# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-16
### Added
- Expanded package-root public API: `SawtoothSyncWrapper`, `CloudConfig`, `Provider`, storage adapters, event bus symbols, and embedding helpers.
- Formal sync-environment guidance (`SyncContextManager` vs `SawtoothSyncWrapper` vs `ContextManager`) with method parity on the sync portal (`pin_entity`, `retrieve_observation`, `.state`).
- Detailed API Reference and examples index in `DOCUMENTATION.md`.
- Deeper examples: sync non-blocking wrapper, cloud compressor, multi-agent pool, Postgres+L3.
- Salience Entity Guard: local heuristic extractor for unstructured identifiers (ticket IDs, tracking codes, reference numbers).
- Multi-match regex extraction (e.g. `uuid`, `uuid_2`) for multiple occurrences per pattern.
- Protection manifest injection into compression prompts and post-merge entity verifier.
- Ingest-time entity scanning on `add_message()` and explicit `pin_entity()` API.
- Strategy provenance telemetry: `salience_heuristic` and `pinned` extraction sources.
- Dual-Target Externalization (DTE) default compression mode with observation crush and fold units.
- Sync-native `SyncContextManager` with inline compression for scripts and WSGI hosts.
- `STABILITY.md` API contract and deprecation policy.
- Release workflow (PyPI publish on version tags) and Python 3.11–3.13 CI matrix.

### Changed
- **Default compression mode is now `dte`.** Set `compression_mode="always_llm"` for pre-0.3.0 eager summarization behavior.
- L3 module docs now reflect `build_prompt()` retrieval injection.
- Event bus documentation corrected to use string event types + `get_event_bus()`.
- `SECURITY.md` support table covers current 0.3.x releases.

### Migration
- Existing integrations require no code changes unless you relied on implicit LLM-on-soft-limit behavior.
- To restore legacy eager summarization: `ContextManagerConfig(compression_mode="always_llm", ...)`.
- Prefer `from sawtooth_memory import ...` over deep submodule imports for stable API access.

## [0.2.2] - 2026-07-11
### Added
- Wired L3 semantic retrieval into `build_prompt()` with automatic injection and opt-out config flag.
- Token-budgeted `[ARCHIVE_L3]` block in the compiled prompt.
- Explainability updates for L3 retrieved chunks in `explain_prompt()`.

## [0.2.1] - 2026-07-11
### Added
- L3 semantic storage layer (Postgres/pgvector, `SemanticIndexer`, embedding providers).
- Benchmark harness + CI workflow.
- Micro and scenario benchmark suites.
- `BENCHMARKS.md` rewrite for rigorous multi-layer methodology.

### Changed
- Added `ruff` linting to CI.
- Shared compression JSON parsing across adapters.
- Event-bus test isolation.

### Fixed
- Compression debounce lock after hard truncate.
- Pool L1.5/L2 merge-on-sync (no clobber).
- Redis L3 metadata persistence.
- L3 indexing on hard-truncate and fallback paths.
- Embedding provider shutdown on ContextManager stop.
- Journal path wiring.
- Merge regressions from cleanup PRs.

## [0.2.0] - 2026-07-01
### Added
- Asynchronous background worker for compression.
- Dual LLM Compression Backends (Ollama and Cloud).
- Deterministic NER Engine.

## [0.1.0] - 2026-06-01
### Added
- Initial release of Sawtooth Memory.
