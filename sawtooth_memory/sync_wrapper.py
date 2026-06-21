"""
sync_wrapper.py — Synchronous Interface Wrapper for Sawtooth-Memory.

Provides a thread-safe, synchronous blocking portal for applications
that cannot use 'async with' natively (e.g., Flask, Django, synchronous CLI tools).
Utilizes AnyIO's BlockingPortal to bridge the sync/async boundary securely.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from anyio.from_thread import start_blocking_portal, BlockingPortal

from .config import ContextManagerConfig
from .middleware import ContextManager
from .state import MessageRole


class SawtoothSyncWrapper:
    """
    Enterprise-grade synchronous wrapper for Sawtooth-Memory.

    Creates a dedicated background event loop using AnyIO's BlockingPortal.
    This prevents thread blockages and ensures thread-safe Pydantic mutations
    by forcing all memory reads and writes through the background loop.
    """

    def __init__(
        self,
        system_prompt: str,
        config: Optional[ContextManagerConfig] = None,
        *,
        enable_events: bool = True,
        journal_path: Optional[Path] = None,
    ) -> None:
        # Store attributes explicitly instead of using *args/**kwargs to satisfy mypy
        self._system_prompt = system_prompt
        self._config = config
        self._enable_events = enable_events
        self._journal_path = journal_path

        self._portal_ctx: Any = None
        self._portal: Optional[BlockingPortal] = None
        self._cm: Optional[ContextManager] = None

    def __enter__(self) -> "SawtoothSyncWrapper":
        """
        Boots the blocking portal and initializes the async ContextManager inside it.
        """
        self._portal_ctx = start_blocking_portal()
        self._portal = self._portal_ctx.__enter__()

        async def _setup() -> ContextManager:
            # Construct explicitly to satisfy static type checkers
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
        """
        Gracefully tears down the ContextManager, drains telemetry, and closes the portal.
        """
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
        """Add a message to Working Memory securely via the background thread."""
        cm = self._cm
        portal = self._portal

        if not portal or not cm:
            raise RuntimeError(
                "SawtoothSyncWrapper must be used within a 'with' context."
            )

        portal.call(cm.add_message, role, content)

    def build_prompt(self) -> List[Dict[str, str]]:
        """
        Compile all memory tiers into an OpenAI-compatible messages list.
        Executed strictly on the background thread to prevent concurrent read/write errors.
        """
        cm = self._cm
        portal = self._portal

        if not portal or not cm:
            raise RuntimeError(
                "SawtoothSyncWrapper must be used within a 'with' context."
            )

        async def _safe_build() -> List[Dict[str, str]]:
            return await cm.build_prompt()

        return portal.call(_safe_build)

    def explain_prompt(self) -> Dict[str, Any]:
        """Recall Explainability Traces and historical telemetry."""
        cm = self._cm
        portal = self._portal

        if not portal or not cm:
            raise RuntimeError(
                "SawtoothSyncWrapper must be used within a 'with' context."
            )

        async def _safe_explain() -> Dict[str, Any]:
            return cm.explain_prompt()

        return portal.call(_safe_explain)

    # ------------------------------------------------------------------
    # Observability Proxies
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot of current token usage and worker health."""
        cm = self._cm
        portal = self._portal

        if not portal or not cm:
            raise RuntimeError(
                "SawtoothSyncWrapper must be used within a 'with' context."
            )

        async def _safe_stats() -> Dict[str, Any]:
            return cm.get_stats()

        return portal.call(_safe_stats)

    def health_check(self) -> Dict[str, Any]:
        """Verifies runtime configurations synchronously."""
        cm = self._cm
        portal = self._portal

        if not portal or not cm:
            raise RuntimeError(
                "SawtoothSyncWrapper must be used within a 'with' context."
            )

        return portal.call(cm.health_check)
