"""
graph.py — LangGraph integration for Sawtooth-Memory.

Provides:
- ``AgentState``               : typed graph state with ``messages`` and
                                  ephemeral ``llm_context`` keys.
- ``sawtooth_compression_node``: pre-LLM node that syncs messages and
                                  compiles the compressed prompt.
- ``llm_node``                 : LLM invocation node with tenacity retries
                                  for transient HTTP errors.
- ``build_sawtooth_graph``     : factory that wires the two nodes together
                                  and returns a compiled ``StateGraph``.

Node execution order:
    [START] ──> sawtooth_compression_node ──> llm_node ──> [END]

Usage::

    from sawtooth_memory import ContextManager, ContextManagerConfig
    from integrations.langgraph.adapter import SawtoothLangGraphAdapter
    from integrations.langgraph.graph import build_sawtooth_graph

    config = ContextManagerConfig(soft_limit_tokens=3000)

    async with ContextManager("You are a helpful agent.", config) as cm:
        adapter = SawtoothLangGraphAdapter(cm)
        graph   = build_sawtooth_graph(llm=my_chat_model, adapter=adapter)
        result  = await graph.ainvoke(
            {"messages": [HumanMessage("Hello!")], "llm_context": []}
        )
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from .adapter import SawtoothLangGraphAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """
    LangGraph state schema for Sawtooth-powered agents.

    ``messages``
        The canonical, append-only conversation log.  LangGraph's default
        ``operator.add`` reducer appends new messages on every node return.

    ``llm_context``
        Ephemeral key overwritten on every compression cycle.  The
        ``sawtooth_compression_node`` writes to it; the ``llm_node``
        reads from it.  It is *never* fed back into the ContextManager —
        only the raw ``messages`` array is.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    llm_context: list[BaseMessage]


# ---------------------------------------------------------------------------
# Tenacity retry helpers
# ---------------------------------------------------------------------------


def _is_transient_error(exc: BaseException) -> bool:
    """
    Return True for HTTP errors that are worth retrying.

    Covers the most common transient failures from OpenAI-compatible APIs:
    - 429 Rate Limit
    - 500 / 502 / 503 / 504 server-side errors

    Falls back on inspecting the exception string when no ``status_code``
    attribute is present (e.g. httpx, requests, aiohttp all surface the
    status code differently).
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in {429, 500, 502, 503, 504}

    # Some clients embed the code in the message string
    msg = str(exc).lower()
    return any(
        code in msg for code in ("429", "500", "502", "503", "504", "rate limit")
    )


_llm_retry = retry(
    retry=retry_if_exception(_is_transient_error),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)

# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def make_compression_node(adapter: SawtoothLangGraphAdapter):
    """
    Return an async node function bound to *adapter*.

    The returned coroutine:
    1. Reads ``state["messages"]`` (the full append-only log).
    2. Syncs the delta to the ContextManager (already-seen messages are
       skipped via ID deduplication inside the adapter).
    3. Builds the optimised compressed prompt array.
    4. Returns ``{"llm_context": <compiled list>}`` to overwrite the
       ephemeral key in the graph state.

    Note: This node only blocks on the local token-counting / sliding-window
    enqueue step inside ``add_message``.  It does **not** block on any
    Ollama background compression call, preserving zero-latency semantics.
    """

    async def sawtooth_compression_node(state: AgentState) -> dict[str, Any]:
        logger.debug(
            "sawtooth_compression_node: syncing %d message(s)",
            len(state["messages"]),
        )
        await adapter.sync_state(state["messages"])
        compiled = await adapter.get_compiled_prompt()
        logger.debug(
            "sawtooth_compression_node: compiled prompt has %d message(s)",
            len(compiled),
        )
        return {"llm_context": compiled}

    sawtooth_compression_node.__name__ = "sawtooth_compression_node"
    return sawtooth_compression_node


def make_llm_node(llm: BaseChatModel):
    """
    Return an async node function bound to *llm*.

    The returned coroutine:
    1. Reads ``state["llm_context"]`` — the compressed prompt from the
       previous node.
    2. Invokes the LLM with tenacity retries for transient HTTP errors
       (429, 500-range).
    3. Returns ``{"messages": [ai_message]}`` which is appended to the
       canonical log via ``operator.add``.
    """

    @_llm_retry
    async def _invoke_with_retry(messages: list[BaseMessage]) -> AIMessage:
        return await llm.ainvoke(messages)

    async def llm_node(state: AgentState) -> dict[str, Any]:
        context = state["llm_context"]
        if not context:
            logger.warning("llm_node: llm_context is empty — graph misconfigured?")

        logger.debug("llm_node: invoking LLM with %d message(s)", len(context))
        response: AIMessage = await _invoke_with_retry(context)
        logger.debug("llm_node: received response from LLM")
        return {"messages": [response]}

    llm_node.__name__ = "llm_node"
    return llm_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_sawtooth_graph(
    llm: BaseChatModel,
    adapter: SawtoothLangGraphAdapter,
) -> Any:
    """
    Build and compile a LangGraph ``StateGraph`` with Sawtooth compression.

    Args:
        llm:     Any LangChain chat model (must support ``ainvoke``).
        adapter: A ``SawtoothLangGraphAdapter`` bound to an active
                 ``ContextManager`` instance (i.e. already started via
                 ``async with`` or ``cm.start()``).

    Returns:
        A compiled LangGraph runnable.  Invoke with::

            result = await graph.ainvoke(
                {"messages": [HumanMessage("...")], "llm_context": []}
            )

    Node wiring:
        START ──> sawtooth_compression_node ──> llm_node ──> END
    """
    compression_node = make_compression_node(adapter)
    llm_node = make_llm_node(llm)

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("sawtooth_compression_node", compression_node)
    builder.add_node("llm_node", llm_node)

    builder.add_edge(START, "sawtooth_compression_node")
    builder.add_edge("sawtooth_compression_node", "llm_node")
    builder.add_edge("llm_node", END)

    return builder.compile()
