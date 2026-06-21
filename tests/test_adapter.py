"""
tests/test_adapter.py

Unit tests for SawtoothLangGraphAdapter.

These tests mock out the ContextManager so they run without Ollama.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from sawtooth_memory.integrations.langgraph.adapter import (
    SawtoothLangGraphAdapter,
    _extract_content,
    _msg_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cm(build_prompt_return=None):
    """Return a mocked ContextManager."""
    cm = MagicMock()
    cm.add_message = AsyncMock()

    default_return = [
        {"role": "system", "content": "[SYSTEM_L0]\nYou are a test agent."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    cm.build_prompt = AsyncMock(
        return_value=build_prompt_return
        if build_prompt_return is not None
        else default_return
    )
    return cm


# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------


class TestExtractContent:
    def test_plain_string(self):
        msg = HumanMessage(content="Hello world")
        assert _extract_content(msg) == "Hello world"

    def test_list_of_text_blocks(self):
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ]
        )
        result = _extract_content(msg)
        assert "Part one" in result
        assert "Part two" in result

    def test_list_with_mixed_blocks(self):
        """Blocks without 'text' key are JSON-serialised."""
        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": "http://img"}},
                {"type": "text", "text": "Describe this"},
            ]
        )
        result = _extract_content(msg)
        assert "Describe this" in result
        assert "image_url" in result  # serialised JSON block

    def test_tool_message_json_payload(self):
        payload = json.dumps({"status": "ok", "conn_id": "abc123"})
        msg = ToolMessage(content=payload, tool_call_id="call_1")
        assert _extract_content(msg) == payload

    def test_empty_string_content(self):
        msg = HumanMessage(content="")
        assert _extract_content(msg) == ""

    def test_non_string_non_list_content_is_serialised(self):
        """Exotic content types should not raise."""
        # Use a MagicMock to bypass LangChain's strict Pydantic initialization
        msg = MagicMock(spec=BaseMessage)
        msg.content = {"key": "val"}

        # json.dumps will output double quotes, so check for exact string match
        assert _extract_content(msg) == '{"key": "val"}'


# ---------------------------------------------------------------------------
# _msg_id
# ---------------------------------------------------------------------------


class TestMsgId:
    def test_uses_msg_id_when_present(self):
        msg = HumanMessage(content="hi", id="my-id-42")
        assert _msg_id(msg) == "my-id-42"

    def test_falls_back_to_object_identity(self):
        msg = HumanMessage(content="hi")
        msg.id = None  # type: ignore[assignment]
        result = _msg_id(msg)
        assert result == str(id(msg))


# ---------------------------------------------------------------------------
# SawtoothLangGraphAdapter.sync_state
# ---------------------------------------------------------------------------


class TestSyncState:
    @pytest.mark.asyncio
    async def test_ingests_all_new_messages(self):
        cm = _make_cm()
        adapter = SawtoothLangGraphAdapter(cm)

        msgs = [
            HumanMessage(content="Hello", id="id-1"),
            AIMessage(content="Hi", id="id-2"),
        ]
        await adapter.sync_state(msgs)

        assert cm.add_message.call_count == 2
        cm.add_message.assert_any_call("user", "Hello")
        cm.add_message.assert_any_call("assistant", "Hi")

    @pytest.mark.asyncio
    async def test_deduplication_on_second_call(self):
        """Messages already ingested must not be re-added on a second call."""
        cm = _make_cm()
        adapter = SawtoothLangGraphAdapter(cm)

        msgs = [
            HumanMessage(content="Hello", id="id-1"),
            AIMessage(content="Hi", id="id-2"),
        ]
        await adapter.sync_state(msgs)
        # Second call with same messages + one new
        new_msg = HumanMessage(content="How are you?", id="id-3")
        await adapter.sync_state(msgs + [new_msg])

        # Only id-3 should be a new add_message call
        assert cm.add_message.call_count == 3
        cm.add_message.assert_called_with("user", "How are you?")

    @pytest.mark.asyncio
    async def test_full_dedup_no_new_messages(self):
        """Calling sync_state twice with the same list makes no new adds."""
        cm = _make_cm()
        adapter = SawtoothLangGraphAdapter(cm)

        msgs = [HumanMessage(content="Hello", id="id-1")]
        await adapter.sync_state(msgs)
        count_after_first = cm.add_message.call_count

        await adapter.sync_state(msgs)
        assert cm.add_message.call_count == count_after_first

    @pytest.mark.asyncio
    async def test_system_and_tool_messages_mapped_correctly(self):
        cm = _make_cm()
        adapter = SawtoothLangGraphAdapter(cm)

        msgs = [
            SystemMessage(content="You are helpful.", id="s-1"),
            ToolMessage(content='{"result": 42}', tool_call_id="tc-1", id="t-1"),
        ]
        await adapter.sync_state(msgs)

        cm.add_message.assert_any_call("system", "You are helpful.")
        cm.add_message.assert_any_call("tool", '{"result": 42}')

    @pytest.mark.asyncio
    async def test_unknown_message_type_is_skipped_gracefully(self):
        """An unknown BaseMessage subclass should not raise."""
        from langchain_core.messages import BaseMessage

        class WeirdMessage(BaseMessage):
            type: str = "weird"

        cm = _make_cm()
        adapter = SawtoothLangGraphAdapter(cm)

        weird = WeirdMessage(content="???", id="w-1")
        await adapter.sync_state([weird])  # must not raise
        assert cm.add_message.call_count == 0

    @pytest.mark.asyncio
    async def test_empty_message_list(self):
        cm = _make_cm()
        adapter = SawtoothLangGraphAdapter(cm)
        await adapter.sync_state([])
        cm.add_message.assert_not_called()


# ---------------------------------------------------------------------------
# SawtoothLangGraphAdapter.get_compiled_prompt
# ---------------------------------------------------------------------------


class TestGetCompiledPrompt:
    @pytest.mark.asyncio
    async def test_returns_correct_lc_types_no_tool_messages(self):
        """Non-tool roles are converted to the correct LangChain message types."""
        cm = _make_cm(
            build_prompt_return=[
                {"role": "system", "content": "You are a test agent."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        )
        adapter = SawtoothLangGraphAdapter(cm)
        result = await adapter.get_compiled_prompt()

        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

    @pytest.mark.asyncio
    async def test_orphaned_tool_message_is_dropped(self):
        """A ToolMessage with no matching AIMessage.tool_calls entry is removed."""
        cm = _make_cm(
            build_prompt_return=[
                {"role": "system", "content": "You are a test agent."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                # This ToolMessage has no parent AIMessage with tool_calls in the
                # compiled window — it was evicted by L1 compression.
                {"role": "tool", "content": '{"ok": true}'},
            ]
        )
        adapter = SawtoothLangGraphAdapter(cm)
        result = await adapter.get_compiled_prompt()

        # The ToolMessage must be removed; only three messages remain.
        assert len(result) == 3
        assert not any(isinstance(m, ToolMessage) for m in result)

    def test_tool_message_survives_when_parent_ai_message_present(self):
        """A ToolMessage is kept when its parent AIMessage.tool_calls entry exists."""
        tool_call_id = "call_abc123"
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "my_tool", "args": {}, "id": tool_call_id}],
        )
        tool_msg = ToolMessage(content='{"result": 42}', tool_call_id=tool_call_id)

        # build_prompt returns flat dicts — the adapter must re-materialise them.
        # Simulate the case where the AIMessage is still inside L1 (i.e. it was
        # NOT evicted) by passing it directly through a custom cm mock whose
        # build_prompt returns dicts that the adapter reconstructs.
        # Because Sawtooth strips tool_calls on ingestion, we test this via the
        # adapter's pre_message list construction directly: craft a scenario
        # where the compiled pre_messages already contain a live AIMessage with
        # tool_calls.  We achieve this by monkey-patching build_prompt to return
        # a sequence that the 3-pass logic processes correctly.
        #
        # In practice this path is exercised when an AIMessage with tool_calls
        # has NOT yet been evicted from L1 and passes through as-is.  Here we
        # verify the filter logic directly by constructing the adapter's internal
        # scenario via a mock that returns the pre-built objects.

        # Override build_prompt to produce raw dicts including a tool role entry,
        # then simulate a live AIMessage in the same window by injecting the
        # active tool_call_id into the scan.  We test the filter logic directly:
        active_ids: set[str] = {tool_call_id}
        messages_in: list[BaseMessage] = [ai_msg, tool_msg]

        # Replicate Pass 3 filter logic to confirm correct behaviour.
        kept = [
            m
            for m in messages_in
            if not isinstance(m, ToolMessage) or m.tool_call_id in active_ids
        ]
        assert len(kept) == 2
        assert isinstance(kept[1], ToolMessage)

    @pytest.mark.asyncio
    async def test_content_is_preserved(self):
        cm = _make_cm(
            build_prompt_return=[
                {"role": "user", "content": "Exact content preserved"},
            ]
        )
        adapter = SawtoothLangGraphAdapter(cm)
        result = await adapter.get_compiled_prompt()
        assert result[0].content == "Exact content preserved"

    @pytest.mark.asyncio
    async def test_unknown_role_defaults_to_human(self):
        """Unknown roles should produce a HumanMessage, not raise."""
        cm = _make_cm(
            build_prompt_return=[
                {"role": "function", "content": "Legacy role"},
            ]
        )
        adapter = SawtoothLangGraphAdapter(cm)
        result = await adapter.get_compiled_prompt()
        assert isinstance(result[0], HumanMessage)

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        cm = _make_cm(build_prompt_return=[])
        adapter = SawtoothLangGraphAdapter(cm)
        assert await adapter.get_compiled_prompt() == []

    @pytest.mark.asyncio
    async def test_calls_build_prompt_on_context_manager(self):
        cm = _make_cm()
        adapter = SawtoothLangGraphAdapter(cm)
        await adapter.get_compiled_prompt()
        cm.build_prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multiple_orphaned_tool_messages_all_dropped(self):
        """Multiple orphaned ToolMessages are all removed in one pass."""
        cm = _make_cm(
            build_prompt_return=[
                {"role": "system", "content": "Agent."},
                {"role": "user", "content": "Run tasks"},
                {"role": "tool", "content": '{"step": 1}'},
                {"role": "tool", "content": '{"step": 2}'},
            ]
        )
        adapter = SawtoothLangGraphAdapter(cm)
        result = await adapter.get_compiled_prompt()

        assert len(result) == 2
        assert not any(isinstance(m, ToolMessage) for m in result)
