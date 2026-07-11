"""Fast mock compression backend for deterministic E2E benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any


class MockCompressor:
    """Deterministic async compressor with configurable artificial delay."""

    def __init__(self, delay_ms: float = 1.0) -> None:
        self.delay_ms = delay_ms
        self.call_count = 0

    async def compress(self, text: str) -> dict[str, Any]:
        self.call_count += 1
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return {
            "narrative_summary": (
                f"Compressed narrative ({len(text)} chars, digest={digest})."
            ),
            "extracted_entities": {},
        }

    async def close(self) -> None:
        return None
