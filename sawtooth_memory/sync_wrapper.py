"""
sync_wrapper.py — Synchronous façade over async ContextManager.

Provides a thread-safe sync API for hosts that cannot use ``async with``
natively (Flask, Django views, synchronous CLIs) but still need the
**non-blocking background compression worker**. Uses AnyIO's BlockingPortal.

For simpler scripts that tolerate blocking inline compression, prefer
``SyncContextManager`` (no portal / daemon event loop).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anyio.from_thread import BlockingPortal, start_blocking_portal

from .config import ContextManagerConfig
from .middleware import ContextManager
from .state import MemoryState, MessageRole
from .storage.semantic import SemanticChunkResult


class SawtoothSyncWrapper:
    """
    Sync façade over the full async ``ContextManager`` stack.

    Runs a dedicated background event loop via AnyIO's BlockingPortal so
    ``add_message`` returns quickly while compression continues on the worker.

    Prefer ``SyncContextManager`` when you want a pure-sync inline path with
    no background threads.
    """

    def __init__(
        self,
        system_prompt: str,
        config: Optional[ContextManagerConfig] = None,
        *,
        enable_events: bool = True,
        journal_path: Optional[Path] = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._config = config
        self._enable_events = enable_events
        self._journal_path = journal_path

        self._portal_ctx: Any = None
        self._portal: Optional[BlockingPortal] = None
        self._cm: Optional[ContextManager] = None

    def _require_live(self) -> Tuple[BlockingPortal, ContextManager]:
        portal = self._portal
        cm = self._cm
        if not portal or not cm:
            raise RuntimeError(
                "SawtoothSyncWrapper must be used within a 'with' context."
            )
        return portal, cm

    def __enter__(self) -> "SawtoothSyncWrapper":
        """Boot the blocking portal and start the async ContextManager inside it."""
        self._portal_ctx = start_blocking_portal()
        self._portal = self._portal_ctx.__enter__()

        async def _setup() -> ContextManager:
            cm = ContextManager(
                system_prompt=self._system_prompt,
                config=self._config,
                enable_events=self._enable_events,
                journal_path=self._journal_path,
            )
            await cm.start()
            return cm

        self._cm = self._portal.call(_setup)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the ContextManager, drain telemetry, and close the portal."""
        cm = self._cm
        portal = self._portal

        if cm and portal:
            portal.call(cm.stop)

        if self._portal_ctx:
            self._portal_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._portal = None
            self._cm = None

    # ------------------------------------------------------------------
    # Core Synchronous API (Thread-Safe Proxies)
    # ------------------------------------------------------------------

    def add_message(self, role: MessageRole, content: str) -> None:
        """Append a message to L1 working memory (non-blocking compression)."""
        portal, cm = self._require_live()
        portal.call(cm.add_message, role, content)

    def retrieve_observation(self, cache_id: str) -> str | None:
        """Return a raw tool observation previously crushed into a cache stub."""
        _, cm = self._require_live()
        return cm.retrieve_observation(cache_id)

    def pin_entity(self, key: str, value: str) -> None:
        """Pin an exact key/value into the L1.5 entity ledger."""
        portal, cm = self._require_live()
        portal.call(cm.pin_entity, key, value)

    def build_prompt(
        self, *, retrieval_query: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Compile all memory tiers into an OpenAI-compatible messages list."""
        portal, cm = self._require_live()

        async def _safe_build() -> List[Dict[str, str]]:
            return await cm.build_prompt(retrieval_query=retrieval_query)

        return portal.call(_safe_build)

    def explain_prompt(self) -> Dict[str, Any]:
        """Return an explainability / audit trace of the compiled prompt."""
        portal, cm = self._require_live()

        async def _safe_explain() -> Dict[str, Any]:
            return cm.explain_prompt()

        return portal.call(_safe_explain)

    def search_semantic_archive(
        self, query: str, top_k: int = 5
    ) -> List[SemanticChunkResult]:
        """Retrieve L3 semantic chunks similar to *query*."""
        portal, cm = self._require_live()

        async def _safe_search() -> List[SemanticChunkResult]:
            return await cm.search_semantic_archive(query, top_k)

        return portal.call(_safe_search)

    def l3_chunk_count(self) -> int:
        """Return the number of indexed L3 semantic chunks for this session."""
        portal, cm = self._require_live()
        return portal.call(cm.l3_chunk_count)

    @property
    def state(self) -> MemoryState:
        """Live MemoryState. Prefer read-only access; mutations go through APIs."""
        _, cm = self._require_live()
        return cm.state

    # ------------------------------------------------------------------
    # Observability Proxies
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of token usage and worker health."""
        portal, cm = self._require_live()

        async def _safe_stats() -> Dict[str, Any]:
            return cm.get_stats()

        return portal.call(_safe_stats)

    def health_check(self) -> Dict[str, Any]:
        """Verify runtime configuration and backend readiness."""
        portal, cm = self._require_live()
        return portal.call(cm.health_check)
