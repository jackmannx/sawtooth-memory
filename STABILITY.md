# API Stability Policy

This document defines what Sawtooth Memory guarantees across releases and how
breaking changes are handled.

---

## Stability tiers

| Tier | Scope | Guarantee |
|------|-------|-----------|
| **Stable** | Symbols listed in `sawtooth_memory.__all__` | Semver; breaking changes only in major releases |
| **Provisional** | `sawtooth_memory.integrations.*` | May change with LangChain / LangGraph version bumps |
| **Internal** | All other modules and private methods (`_prefix`) | No compatibility guarantee |

The canonical stable surface is `sawtooth_memory.__all__`. Import from the
package root whenever possible:

```python
from sawtooth_memory import ContextManager, ContextManagerConfig
```

Avoid relying on deep submodule paths (e.g. `sawtooth_memory.middleware`) unless
you accept internal-tier risk.

---

## Semantic versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

| Bump | When |
|------|------|
| **Major** (`1.0.0`) | Removed or renamed stable symbols; incompatible changes to return shapes of `build_prompt()`, `explain_prompt()`, or `health_check()` |
| **Minor** (`0.3.0`) | New stable exports, new config fields with safe defaults, new optional behavior |
| **Patch** (`0.3.1`) | Bug fixes with no intentional behavior change |

Pre-1.0 releases (`0.x.y`) may include behavior changes on minor bumps when
documented in CHANGELOG (e.g. default config changes).

---

## Deprecation process

1. Mark the symbol with a `@deprecated` docstring and emit `DeprecationWarning`
   at runtime when feasible.
2. Document under `### Deprecated` in `CHANGELOG.md`.
3. Keep the deprecated API for **at least one minor release** before removal.
4. Remove only in a **major** release (or after two minors for low-risk removals).

---

## Breaking change definition

The following are considered breaking for stable API consumers:

- Removal or rename of any name in `__all__`
- Change to default values that alter runtime behavior without an opt-out
  (document in CHANGELOG **Migration** section)
- Change to the structure returned by `explain_prompt()` or `health_check()`
  where keys are removed or repurposed
- Change to `build_prompt()` message dict shape (`role` / `content` keys)

The following are **not** breaking:

- New optional keyword arguments on public methods
- New keys added to diagnostic dicts (`get_stats()`, `explain_prompt()`)
- New symbols added to `__all__`
- Bug fixes that restore documented behavior

---

## Manager API parity

Three public managers expose a shared sync-facing surface:

| Method | `SyncContextManager` | `SawtoothSyncWrapper` | `ContextManager` |
|--------|----------------------|----------------------|-------------------|
| `add_message` | sync | sync (via portal) | async |
| `pin_entity` | sync | sync | async |
| `retrieve_observation` | sync | sync | sync |
| `build_prompt` | sync | sync | async |
| `explain_prompt` | sync | sync | sync |
| `search_semantic_archive` | sync | sync | async |
| `l3_chunk_count` | sync | sync | async |
| `state` | property | property | property |
| `get_stats` | sync | sync | sync |
| `health_check` | sync | sync | async |

`ContextManager` additionally requires `async with` lifecycle and exposes
`start()` / `stop()` for explicit control.

---

## Configuration stability

`ContextManagerConfig` fields may grow on minor releases. Existing field names
and types will not change on patch releases.

Notable default (0.3.0+): `compression_mode="dte"`. Use
`compression_mode="always_llm"` to restore pre-0.3.0 eager summarization
behavior.

---

## Reporting issues

- **Bugs:** GitHub Issues using the bug report template
- **Security:** See [SECURITY.md](SECURITY.md) — do not file public issues for
  vulnerabilities
