"""
tests/test_graph.py

Tests for graph.py: node factories, AgentState schema, and the compiled
graph happy-path and retry behaviour.

All LLM and ContextManager interactions are mocked — no live API or Ollama
instance is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sawtooth_memory.integrations.langgraph.adapter import SawtoothLangGraphAdapter
from sawtooth_memory.integrations.langgraph.graph import (
    AgentState,
    build_sawtooth_graph,
    make_compression_node,
    make_llm_node,
    _is_transient_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(compiled_prompt=None):
    """Return a SawtoothLangGraphAdapter backed by a mocked ContextManager."""
    cm = MagicMock()
    cm.add_message = AsyncMock()

    default_prompt = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "Hello"},
    ]

    cm.build_prompt = AsyncMock(
        return_value=compiled_prompt if compiled_prompt is not None else default_prompt
    )
    return SawtoothLangGraphAdapter(cm)


def _make_llm(response_content: str = "Test response"):
    """Return a mock ChatModel whose ainvoke returns an AIMessage."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_content))
    return llm


# ---------------------------------------------------------------------------
# _is_transient_error
# ---------------------------------------------------------------------------


class TestIsTransientError:
    def test_429_is_transient(self):
        err = MagicMock(status_code=429)
        assert _is_transient_error(err) is True

    def test_500_is_transient(self):
        err = MagicMock(status_code=500)
        assert _is_transient_error(err) is True

    def test_503_is_transient(self):
        err = MagicMock(status_code=503)
        assert _is_transient_error(err) is True

    def test_400_is_not_transient(self):
        err = MagicMock(status_code=400)
        assert _is_transient_error(err) is False

    def test_404_is_not_transient(self):
        err = MagicMock(status_code=404)
        assert _is_transient_error(err) is False

    def test_string_rate_limit_in_message(self):
        """Clients that embed the status code in the message text."""
        err = Exception("HTTP 429 Too Many Requests: rate limit exceeded")
        assert _is_transient_error(err) is True

    def test_non_http_exception_is_not_transient(self):
        err = ValueError("bad value")
        assert _is_transient_error(err) is False


# ---------------------------------------------------------------------------
# make_compression_node
# ---------------------------------------------------------------------------


class TestCompressionNode:
    @pytest.mark.asyncio
    async def test_returns_llm_context_key(self):
        adapter = _make_adapter()
        node = make_compression_node(adapter)

        state: AgentState = {
            "messages": [HumanMessage(content="Hello", id="m-1")],
            "llm_context": [],
        }
        result = await node(state)

        assert "llm_context" in result
        assert isinstance(result["llm_context"], list)
        assert len(result["llm_context"]) > 0

    @pytest.mark.asyncio
    async def test_compiled_prompt_has_correct_types(self):
        adapter = _make_adapter(
            compiled_prompt=[
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "Hi"},
            ]
        )
        node = make_compression_node(adapter)
        state: AgentState = {
            "messages": [HumanMessage(content="Hi", id="m-1")],
            "llm_context": [],
        }
        result = await node(state)

        from langchain_core.messages import SystemMessage, HumanMessage as HM

        assert isinstance(result["llm_context"][0], SystemMessage)
        assert isinstance(result["llm_context"][1], HM)

    @pytest.mark.asyncio
    async def test_sync_state_called_with_full_message_list(self):
        adapter = _make_adapter()
        # Spy on sync_state
        original_sync = adapter.sync_state
        called_with = []

        async def spy_sync(msgs):
            called_with.append(list(msgs))
            return await original_sync(msgs)

        adapter.sync_state = spy_sync

        node = make_compression_node(adapter)
        msgs = [HumanMessage(content="Hello", id="m-1")]
        state: AgentState = {"messages": msgs, "llm_context": []}
        await node(state)

        assert called_with[0] == msgs

    @pytest.mark.asyncio
    async def test_does_not_raise_on_empty_messages(self):
        adapter = _make_adapter(compiled_prompt=[])
        node = make_compression_node(adapter)
        state: AgentState = {"messages": [], "llm_context": []}
        result = await node(state)
        assert result["llm_context"] == []


# ---------------------------------------------------------------------------
# make_llm_node
# ---------------------------------------------------------------------------


class TestLlmNode:
    @pytest.mark.asyncio
    async def test_invokes_llm_with_llm_context(self):
        llm = _make_llm("The answer is 42.")
        node = make_llm_node(llm)

        context = [SystemMessage(content="Sys"), HumanMessage(content="Q")]
        state: AgentState = {"messages": [], "llm_context": context}
        result = await node(state)

        llm.ainvoke.assert_awaited_once_with(context)
        assert result["messages"][0].content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_returns_messages_key_for_operator_add(self):
        """The return dict must use the 'messages' key so operator.add appends."""
        llm = _make_llm()
        node = make_llm_node(llm)
        state: AgentState = {
            "messages": [],
            "llm_context": [HumanMessage(content="hi")],
        }
        result = await node(state)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        """The LLM node must retry transient 429 errors before succeeding."""
        call_count = 0

        async def flaky_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                err = Exception("HTTP 429 rate limit")
                err.status_code = 429  # type: ignore[attr-defined]
                raise err
            return AIMessage(content="Eventually OK")

        llm = MagicMock()
        llm.ainvoke = flaky_invoke

        node = make_llm_node(llm)
        state: AgentState = {
            "messages": [],
            "llm_context": [HumanMessage(content="hi")],
        }
        result = await node(state)
        assert result["messages"][0].content == "Eventually OK"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_transient_error_is_not_retried(self):
        """A 400 Bad Request should propagate immediately without retrying."""
        call_count = 0

        async def bad_invoke(messages):
            nonlocal call_count
            call_count += 1
            err = Exception("HTTP 400 Bad Request")
            err.status_code = 400  # type: ignore[attr-defined]
            raise err

        llm = MagicMock()
        llm.ainvoke = bad_invoke

        node = make_llm_node(llm)
        state: AgentState = {
            "messages": [],
            "llm_context": [HumanMessage(content="hi")],
        }
        with pytest.raises(Exception, match="400"):
            await node(state)

        assert call_count == 1  # no retry


# ---------------------------------------------------------------------------
# build_sawtooth_graph — full pipeline
# ---------------------------------------------------------------------------


class TestBuildSawtoothGraph:
    @pytest.mark.asyncio
    async def test_happy_path_end_to_end(self):
        """Full graph: compression node feeds llm_context into llm_node."""
        adapter = _make_adapter(
            compiled_prompt=[
                {"role": "system", "content": "You are a test agent."},
                {"role": "user", "content": "Hello"},
            ]
        )
        llm = _make_llm("Hello back!")
        graph = build_sawtooth_graph(llm=llm, adapter=adapter)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Hello", id="m-1")],
            "llm_context": [],
        }
        result = await graph.ainvoke(initial_state)

        # The AI response must have been appended to messages
        all_msgs = result["messages"]
        ai_msgs = [m for m in all_msgs if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "Hello back!"

    @pytest.mark.asyncio
    async def test_llm_context_is_overwritten_not_appended(self):
        """llm_context must be replaced (not accumulated) on each cycle."""
        adapter = _make_adapter(
            compiled_prompt=[{"role": "user", "content": "Compressed"}]
        )
        llm = _make_llm("Response")
        graph = build_sawtooth_graph(llm=llm, adapter=adapter)

        state: AgentState = {
            "messages": [HumanMessage(content="Hi", id="m-1")],
            "llm_context": [HumanMessage(content="OLD context", id="old")],
        }
        result = await graph.ainvoke(state)

        # After the run llm_context reflects the latest compressed prompt
        # (the graph output preserves the last value written by sawtooth_node)
        assert result["llm_context"][0].content == "Compressed"

    @pytest.mark.asyncio
    async def test_original_messages_preserved(self):
        """operator.add must not lose the initial messages."""
        adapter = _make_adapter()
        llm = _make_llm("Pong")
        graph = build_sawtooth_graph(llm=llm, adapter=adapter)

        initial_human = HumanMessage(content="Ping", id="m-1")
        result = await graph.ainvoke({"messages": [initial_human], "llm_context": []})

        contents = [m.content for m in result["messages"]]
        assert "Ping" in contents  # original preserved
        assert "Pong" in contents  # AI reply appended
