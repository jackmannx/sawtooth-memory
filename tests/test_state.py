"""tests/test_state.py — Unit tests for Pydantic state schemas."""

import pytest
from pydantic import ValidationError

from sawtooth_memory.state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    Message,
    SystemPrompt,
    WorkingMemory,
)


class TestMessage:
    def test_valid_roles(self):
        for role in ["user", "assistant", "system", "tool"]:
            msg = Message(role=role, content="hello")
            assert msg.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(role="bot", content="hello")

    def test_to_openai_dict(self):
        msg = Message(role="user", content="Hi")
        d = msg.to_openai_dict()
        assert d == {"role": "user", "content": "Hi"}

    def test_default_token_count(self):
        msg = Message(role="user", content="test")
        assert msg.token_count == 0


class TestWorkingMemory:
    def test_append_updates_token_count(self):
        wm = WorkingMemory()
        msg = Message(role="user", content="hello", token_count=10)
        wm.append(msg)
        assert wm.token_count == 10
        assert len(wm.messages) == 1

    def test_slice_oldest_removes_and_recounts(self):
        wm = WorkingMemory()
        msgs = [Message(role="user", content=f"msg{i}", token_count=5) for i in range(5)]
        for m in msgs:
            wm.append(m)

        assert wm.token_count == 25

        chunk = wm.slice_oldest(3)
        assert len(chunk) == 3
        assert len(wm.messages) == 2
        assert wm.token_count == 10

    def test_slice_more_than_available(self):
        wm = WorkingMemory()
        wm.append(Message(role="user", content="only one", token_count=5))
        chunk = wm.slice_oldest(10)
        assert len(chunk) == 1
        assert wm.token_count == 0


class TestEntityLedger:
    def test_upsert_merges_new_keys(self):
        """Distinct keys from separate upsert calls are all retained."""
        ledger = EntityLedger()
        ledger.upsert({"db_id": "abc123"})
        ledger.upsert({"file": "/tmp/out.txt"})
        assert ledger.get_latest("db_id") == "abc123"
        assert ledger.get_latest("file") == "/tmp/out.txt"

    def test_upsert_appends_on_collision(self):
        """A new value for an existing key is appended, not overwritten."""
        ledger = EntityLedger()
        ledger.upsert({"key": "old"})
        ledger.upsert({"key": "new"})
        # Latest value is the new one (backwards-compatible semantic).
        assert ledger.get_latest("key") == "new"
        # But the old value is preserved in history.
        assert ledger.get_history("key") == ["old", "new"]

    def test_upsert_deduplicates_identical_values(self):
        """Reinserting the same value for a key is a no-op (no duplicates)."""
        ledger = EntityLedger()
        ledger.upsert({"token": "abc"})
        ledger.upsert({"token": "abc"})  # identical — should not duplicate
        assert ledger.get_history("token") == ["abc"]

    def test_upsert_rolling_window_enforced(self):
        """Values beyond max_history_per_key evict the oldest entry."""
        ledger = EntityLedger(max_history_per_key=3)
        for i in range(5):
            ledger.upsert({"conn": f"conn-{i}"})
        history = ledger.get_history("conn")
        assert len(history) == 3
        # Oldest entries are evicted; latest three are retained.
        assert history == ["conn-2", "conn-3", "conn-4"]

    def test_get_latest_returns_none_for_missing_key(self):
        ledger = EntityLedger()
        assert ledger.get_latest("nonexistent") is None

    def test_get_history_returns_empty_list_for_missing_key(self):
        ledger = EntityLedger()
        assert ledger.get_history("nonexistent") == []

    def test_to_json_str_single_value(self):
        """Single-value keys render without a history companion entry."""
        import json
        ledger = EntityLedger()
        ledger.upsert({"x": "1"})
        parsed = json.loads(ledger.to_json_str())
        assert parsed["x"] == "1"
        assert "x__history" not in parsed

    def test_to_json_str_multi_value_includes_history_key(self):
        """Keys with multiple values include a ``<key>__history`` entry."""
        import json
        ledger = EntityLedger()
        ledger.upsert({"conn": "A"})
        ledger.upsert({"conn": "B"})
        ledger.upsert({"conn": "C"})
        parsed = json.loads(ledger.to_json_str())
        # Latest value is rendered as the primary entry.
        assert parsed["conn"] == "C"
        # Earlier values are visible in the history companion.
        assert "conn__history" in parsed
        assert "A" in parsed["conn__history"]
        assert "B" in parsed["conn__history"]

    def test_to_json_str_legacy_direct_init(self):
        """EntityLedger constructed with pre-populated entities dict still serialises."""
        import json
        ledger = EntityLedger(entities={"x": ["1"]})
        parsed = json.loads(ledger.to_json_str())
        assert parsed == {"x": "1"}


class TestArchivalMemory:
    def test_append_narrative_first(self):
        arch = ArchivalMemory()
        arch.append_narrative("First note.")
        assert arch.narrative == "First note."

    def test_append_narrative_subsequent(self):
        arch = ArchivalMemory()
        arch.append_narrative("First.")
        arch.append_narrative("Second.")
        assert "First." in arch.narrative
        assert "Second." in arch.narrative

    def test_append_empty_string_noop(self):
        arch = ArchivalMemory()
        arch.append_narrative("   ")
        assert arch.narrative == ""


class TestMemoryState:
    def test_default_empty_tiers(self):
        state = MemoryState(l0_system=SystemPrompt(content="You are helpful."))
        assert state.l1_working.messages == []
        assert state.l1_5_entities.entities == {}
        assert state.l2_archival.narrative == ""
        assert state.l3_semantic.chunk_count == 0
        assert state.l3_semantic.last_indexed_at is None

    def test_l0_content(self):
        state = MemoryState(l0_system=SystemPrompt(content="Agent persona."))
        assert state.l0_system.content == "Agent persona."


class TestSemanticVectorMemory:
    def test_defaults(self):
        from sawtooth_memory.state import SemanticVectorMemory

        l3 = SemanticVectorMemory()
        assert l3.chunk_count == 0
        assert l3.last_indexed_at is None
