"""
adapter.py — SawtoothLangGraphAdapter

Bidirectional bridge between LangChain's typed message objects and
Sawtooth-Memory's ContextManager. Performs message-ID deduplication
so it is safe to call sync_state() on every graph iteration (including
cycles) without double-ingesting the same messages.

Role mapping (bidirectional):
    HumanMessage  <->  "user"
    AIMessage     <->  "assistant"
    SystemMessage <->  "system"
    ToolMessage   <->  "tool"
"""

from __future__ import annotations

import json
import logging
from typing import Sequence, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from sawtooth_memory import ContextManager
from sawtooth_memory.state import MessageRole

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role <-> LangChain message type mapping tables
# ---------------------------------------------------------------------------

_LC_TO_ROLE: dict[type[BaseMessage], MessageRole] = {
    HumanMessage: "user",
    AIMessage: "assistant",
    SystemMessage: "system",
    ToolMessage: "tool",
}

_ROLE_TO_LC: dict[MessageRole, type[BaseMessage]] = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}


def _extract_content(msg: BaseMessage) -> str:
    """
    Safely extract a plain-text string from any LangChain message.

    LangChain's ``content`` field can be:
    - A plain ``str``          — returned as-is
    - A ``list`` of blocks     — each block serialised to JSON / text and
                                  joined with newlines (covers multimodal
                                  content, tool call payloads, etc.)
    - Anything else            — serialised via ``json.dumps``
    """
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Covers {"type": "text", "text": "..."} and tool-call dicts
                parts.append(
                    block.get("text")
                    or block.get("content")
                    or json.dumps(block, ensure_ascii=False)
                )
            else:
                parts.append(json.dumps(block, default=str, ensure_ascii=False))
        return "\n".join(filter(None, parts))
    # Fallback for any exotic content type
    return json.dumps(content, default=str, ensure_ascii=False)


def _msg_id(msg: BaseMessage) -> str:
    """
    Return a stable identifier for a message.

    LangGraph assigns a ``msg.id`` string to each message it creates; fall
    back to Python's object identity for messages constructed outside the
    graph (e.g. in tests).
    """
    return msg.id if msg.id else str(id(msg))


class SawtoothLangGraphAdapter:
    """
    Stateful adapter that connects a single LangGraph session to a
    ``ContextManager`` instance.

    One adapter per graph session / thread.  Create it alongside the
    ContextManager, then pass *the same instance* into both graph nodes.

    Example::

        cm = ContextManager("You are a helpful agent.", config)
        adapter = SawtoothLangGraphAdapter(cm)

        async with cm:
            graph = build_graph(adapter)
            result = await graph.ainvoke({"messages": [HumanMessage("Hello")]})
    """

    def __init__(self, context_manager: ContextManager) -> None:
        self._context_manager = context_manager
        # Tracks message IDs that have already been fed into the ContextManager
        # so that cyclical graph iterations don't double-ingest history.
        self._processed_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def sync_state(self, messages: Sequence[BaseMessage]) -> None:
        """
        Ingest new messages into the ContextManager.

        Only messages whose IDs are *not* already in ``processed_ids`` are
        forwarded.  Safe to call on every graph iteration — already-seen
        messages are silently skipped.

        Args:
            messages: The full ``state["messages"]`` list from the graph.
        """
        new_count = 0
        for msg in messages:
            msg_id = _msg_id(msg)
            if msg_id in self._processed_ids:
                continue

            role = _LC_TO_ROLE.get(type(msg))
            if role is None:
                # Unknown message type — log and skip rather than crashing.
                logger.warning(
                    "sync_state: skipping unsupported message type %s (id=%s)",
                    type(msg).__name__,
                    msg_id,
                )
                # Mark as processed so we don't warn on every cycle.
                self._processed_ids.add(msg_id)
                continue

            content = _extract_content(msg)
            await self._context_manager.add_message(role, content)
            self._processed_ids.add(msg_id)
            new_count += 1

        logger.debug(
            "sync_state: ingested %d new message(s), total processed=%d",
            new_count,
            len(self._processed_ids),
        )

    async def get_compiled_prompt(self) -> list[BaseMessage]:
        """
        Build an optimised, compressed prompt and convert it to LangChain
        message objects.

        Includes a sanitization pass that removes orphaned ``ToolMessage``
        entries — i.e. any ``ToolMessage`` whose originating ``AIMessage``
        (identified by a matching ``tool_calls[i].id``) is no longer present
        in the compiled sequence because it was evicted and compressed into
        the L2 archive.

        Sending orphaned ``ToolMessage`` objects to strict cloud APIs
        (OpenAI, Anthropic) causes an HTTP 400 because the API enforces
        that every ``tool_call_id`` in a ``ToolMessage`` must reference an
        ``id`` that appears in a preceding ``AIMessage.tool_calls`` array.

        Returns:
            An ordered list of LangChain ``BaseMessage`` objects representing
            the compiled L0 / L1.5 / L2 / active-L1 context windows, ready
            to pass directly to ``llm.ainvoke()``.
        """
        raw_prompt: list[dict[str, str]] = await self._context_manager.build_prompt()

        # ── Pass 1: materialise raw dicts into typed LangChain objects ──────
        # ToolMessage construction is deferred to Pass 2 so we can first
        # collect the full set of active tool_call_ids from any AIMessage.
        pre_messages: list[BaseMessage] = []

        for item in raw_prompt:
            role = item.get("role", "user")
            content: str = item.get("content", "")
            msg_cls = _ROLE_TO_LC.get(cast(MessageRole, role))
            if msg_cls is None:
                logger.warning(
                    "get_compiled_prompt: unknown role %r — defaulting to HumanMessage",
                    role,
                )
                msg_cls = HumanMessage

            if msg_cls is ToolMessage:
                # Use a sentinel tool_call_id for now; Pass 2 will filter
                # these out if the originating AIMessage is absent.
                pre_messages.append(
                    ToolMessage(content=content, tool_call_id="__sawtooth_pending__")
                )
            else:
                pre_messages.append(msg_cls(content=content))

        # ── Pass 2: collect active tool_call_ids from present AIMessages ────
        # An AIMessage produced by Sawtooth's memory reconstruction will have
        # no tool_calls (they are not stored in L1 state), so the active set
        # will be empty for all Sawtooth-reconstructed messages.  An AIMessage
        # passed through without compression (e.g. still in the L1 window)
        # retains its tool_calls array and its children survive the filter.
        active_tool_call_ids: set[str] = set()
        for msg in pre_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    # tool_calls items are dicts with an "id" key, or
                    # ToolCall typed-dicts depending on the langchain version.
                    tc_id = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", None)
                    )
                    if tc_id:
                        active_tool_call_ids.add(tc_id)

        # ── Pass 3: build final list, dropping orphaned ToolMessages ────────
        lc_messages: list[BaseMessage] = []
        dropped = 0
        for msg in pre_messages:
            if isinstance(msg, ToolMessage):
                if msg.tool_call_id not in active_tool_call_ids:
                    # The parent AIMessage has been evicted from L1 and
                    # compressed into L2.  Its semantic content is already
                    # captured in the archival narrative, so we can safely
                    # omit this ToolMessage to keep the payload schema-valid.
                    dropped += 1
                    logger.debug(
                        "get_compiled_prompt: dropped orphaned ToolMessage "
                        "(tool_call_id=%r — parent AIMessage not in window)",
                        msg.tool_call_id,
                    )
                    continue
            lc_messages.append(msg)

        if dropped:
            logger.info(
                "get_compiled_prompt: sanitized %d orphaned ToolMessage(s) "
                "whose parent AIMessage was evicted from L1.",
                dropped,
            )

        logger.debug(
            "get_compiled_prompt: compiled %d message(s) (%d orphan(s) removed)",
            len(lc_messages),
            dropped,
        )
        return lc_messages
