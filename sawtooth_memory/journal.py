"""Async JSONL journal for compression cycles – fully non-blocking."""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class JournalEntry:
    """Single compression journal entry – matches Phase 2.2 spec exactly."""

    cycle_id: str
    timestamp: str  # ISO-8601 UTC
    l1_tokens_evicted: int
    l1_5_entities_retained: Dict[str, Any]
    l2_summary_generated: str
    # Additional optional fields
    messages_compressed: Optional[int] = None
    total_duration_ms: Optional[int] = None
    final_l1_tokens: Optional[int] = None

    def to_json(self) -> str:
        """Serialize to JSON string (deterministic order)."""
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)


class AsyncCompressionJournal:
    """
    Async journal writer with automatic rotation.

    - Writes are non-blocking (uses aiofiles)
    - JSONL format (one JSON object per line) for append-only and easy parsing
    - Automatic rotation based on size or entry count
    - Thread-safe (asyncio)
    """

    def __init__(
        self,
        path: Path,
        max_size_mb: int = 100,
        max_entries: int = 10000,
        flush_interval_seconds: float = 1.0,
    ):
        self.path = Path(path)
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self.flush_interval = flush_interval_seconds

        self._queue: asyncio.Queue[JournalEntry] = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None
        self._running = False
        self._entry_count = 0
        self._current_size = 0

        # Create parent directory
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_stats()

    def _init_stats(self) -> None:
        """Initialize entry count and file size from existing journal."""
        if self.path.exists():
            self._current_size = self.path.stat().st_size
            # Count entries (approx by counting newlines – fast)
            try:
                with open(self.path, "rb") as f:
                    self._entry_count = sum(1 for _ in f)
            except Exception:
                self._entry_count = 0

    async def start(self) -> None:
        """Start background writer task."""
        if self._running:
            return
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.debug(f"Journal started: {self.path}")

    async def stop(self) -> None:
        """Stop writer and flush remaining entries."""
        self._running = False
        if self._writer_task:
            await self._writer_task
        logger.debug("Journal stopped")

    async def write(
        self,
        cycle_id: str,
        l1_tokens_evicted: int,
        l1_5_entities_retained: Dict[str, Any],
        l2_summary_generated: str,
        timestamp: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Queue a journal entry (non-blocking).

        Args:
            cycle_id: Unique ID for compression cycle
            l1_tokens_evicted: Tokens removed from L1
            l1_5_entities_retained: Dict of entities preserved
            l2_summary_generated: Summary text
            timestamp: UTC datetime (default now)
            extra: Additional fields like messages_compressed, duration
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        entry = JournalEntry(
            cycle_id=cycle_id,
            timestamp=timestamp.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            l1_tokens_evicted=l1_tokens_evicted,
            l1_5_entities_retained=l1_5_entities_retained,
            l2_summary_generated=l2_summary_generated,
            **(extra or {}),
        )

        await self._queue.put(entry)

    async def _writer_loop(self) -> None:
        """Background loop that writes entries to disk."""
        buffer = []
        last_flush = asyncio.get_running_loop().time()

        while self._running or not self._queue.empty():
            try:
                # Wait for an entry with timeout to allow periodic flush
                entry = await asyncio.wait_for(
                    self._queue.get(), timeout=self.flush_interval
                )
                buffer.append(entry)
            except asyncio.TimeoutError:
                pass

            # Flush if buffer has entries or time elapsed
            now = asyncio.get_running_loop().time()
            if buffer and (
                len(buffer) >= 10 or now - last_flush >= self.flush_interval
            ):
                await self._flush_buffer(buffer)
                buffer.clear()
                last_flush = now

            # Check rotation
            if self._should_rotate():
                await self._rotate()

        # Final flush
        if buffer:
            await self._flush_buffer(buffer)

    async def _flush_buffer(self, entries: List[JournalEntry]) -> None:
        """Write buffer to file atomically."""
        if not entries:
            return

        # Build lines
        lines = [entry.to_json() + "\n" for entry in entries]

        try:
            # Write with aiofiles (non-blocking)
            async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
                await f.writelines(lines)

            self._entry_count += len(entries)
            self._current_size += sum(len(line.encode("utf-8")) for line in lines)
        except Exception as e:
            logger.error(f"Failed to write journal entries: {e}")
            # Don't re-raise – we don't want to crash the agent

    def _should_rotate(self) -> bool:
        """Check rotation conditions."""
        size_mb = self._current_size / (1024 * 1024)
        return size_mb >= self.max_size_mb or self._entry_count >= self.max_entries

    async def _rotate(self) -> None:
        """Rotate journal: move current file to backup."""
        if not self.path.exists():
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = self.path.with_suffix(f".{timestamp}.jsonl.bak")

        try:
            # Close any open handles and rename
            await asyncio.to_thread(self.path.rename, backup)
            self._entry_count = 0
            self._current_size = 0
            logger.info(f"Journal rotated: {backup}")
        except Exception as e:
            logger.error(f"Rotation failed: {e}")

    async def read_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read most recent entries (for debugging)."""
        if not self.path.exists():
            return []

        try:
            async with aiofiles.open(self.path, "r", encoding="utf-8") as f:
                lines = await f.readlines()
            # Last `limit` lines, parse JSON
            entries = []
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return list(reversed(entries))  # most recent first
        except Exception as e:
            logger.error(f"Failed to read journal: {e}")
            return []
