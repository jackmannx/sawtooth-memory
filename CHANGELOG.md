# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - YYYY-MM-DD
### Added
- Pending Phase 4

### Fixed
- Pending Phase 4

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
