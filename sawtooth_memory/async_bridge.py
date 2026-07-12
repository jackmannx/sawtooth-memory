"""
async_bridge.py — Minimal asyncio bridge for sync callers with async storage.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_coro_once(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run a single coroutine to completion without a persistent event loop.

    Used by SyncContextManager at storage/L3 boundaries only.
    """
    return asyncio.run(coro)
