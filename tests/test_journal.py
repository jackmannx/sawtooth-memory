import asyncio
from datetime import datetime, timezone

import pytest

from sawtooth_memory.journal import AsyncCompressionJournal

pytestmark = pytest.mark.asyncio


async def test_journal_write_and_read(tmp_path):
    journal_file = tmp_path / "test_journal.jsonl"

    # Low flush interval to ensure it writes quickly during the test
    journal = AsyncCompressionJournal(journal_file, flush_interval_seconds=0.1)
    await journal.start()

    await journal.write(
        cycle_id="test-cycle-123",
        l1_tokens_evicted=150,
        l1_5_entities_retained={"user_id": "abc"},
        l2_summary_generated="User discussed testing.",
        timestamp=datetime(2026, 6, 2, tzinfo=timezone.utc),
        extra={"messages_compressed": 4},
    )

    # Give the background task a tiny moment to flush
    await asyncio.sleep(0.2)
    await journal.stop()

    assert journal_file.exists()

    # Verify the contents
    entries = await journal.read_recent()
    assert len(entries) == 1
    assert entries[0]["cycle_id"] == "test-cycle-123"
    assert entries[0]["l1_tokens_evicted"] == 150
    assert entries[0]["l1_5_entities_retained"]["user_id"] == "abc"
    assert entries[0]["messages_compressed"] == 4


async def test_journal_rotation(tmp_path):
    journal_file = tmp_path / "rotate_journal.jsonl"

    # Set max entries to 2 to force a rotation quickly
    journal = AsyncCompressionJournal(
        journal_file, max_entries=2, flush_interval_seconds=0.1
    )
    await journal.start()

    # Write 3 entries (should trigger 1 rotation)
    for i in range(3):
        await journal.write(
            cycle_id=f"cycle-{i}",
            l1_tokens_evicted=10,
            l1_5_entities_retained={},
            l2_summary_generated="dummy",
        )
        await asyncio.sleep(0.15)  # wait for flush

    await journal.stop()

    # There should now be the active file PLUS one backup file (.bak)
    files = list(tmp_path.glob("*.jsonl*"))
    assert len(files) == 2

    bak_files = list(tmp_path.glob("*.bak"))
    assert len(bak_files) == 1
