"""
langchain_adapter.py — Modern LangChain BaseChatMessageHistory Adapter.

Allows Sawtooth-Memory to serve as a drop-in message history provider for
modern LCEL Runnables, LangGraph, and standard Agent executors.
"""

from pathlib import Path
from typing import List, Optional, Sequence
from langchain_core.chat_history import BaseChatMessageHistory  # type: ignore[import-not-found]
from langchain_core.messages import (  # type: ignore[import-not-found]
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from ..config import ContextManagerConfig
from ..sync_wrapper import SawtoothSyncWrapper


class SawtoothChatMessageHistory(BaseChatMessageHistory):
    """
    Modern LCEL-compliant Chat Message History adapter for Sawtooth-Memory.

    Provides flawless thread safety by wrapping the SawtoothSyncWrapper portal,
    automatically processing async background token compression and NER extraction.
    """

    def __init__(
        self,
        system_prompt: str,
        config: Optional[ContextManagerConfig] = None,
        enable_events: bool = True,
        journal_path: Optional[Path] = None,
    ) -> None:
        """Initialize standard Python attributes."""
        self.system_prompt = system_prompt
        self.config = config
        self.enable_events = enable_events
        self.journal_path = journal_path

        # Internal state tracking without Pydantic dependencies
        self._sync_wrapper: Optional[SawtoothSyncWrapper] = None

    def init_portal(self) -> None:
        """Explicitly boot the underlying thread portal context loop."""
        if self._sync_wrapper is None:
            wrapper = SawtoothSyncWrapper(
                system_prompt=self.system_prompt,
                config=self.config,
                enable_events=self.enable_events,
                journal_path=self.journal_path,
            )
            self._sync_wrapper = wrapper.__enter__()

    def close_portal(self) -> None:
        """Gracefully shut down the background portal and flush logs."""
        if self._sync_wrapper is not None:
            self._sync_wrapper.__exit__(None, None, None)
            self._sync_wrapper = None

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Fetch thread-safe memory state snapshots from the background portal."""
        self.init_portal()
        assert self._sync_wrapper is not None

        raw_messages = self._sync_wrapper.build_prompt()
        lc_messages: List[BaseMessage] = []

        for msg in raw_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        return lc_messages

    def add_message(self, message: BaseMessage) -> None:
        """Persist a single LangChain message to the state machine securely."""
        self.init_portal()
        assert self._sync_wrapper is not None

        if isinstance(message, HumanMessage) or message.type == "human":
            self._sync_wrapper.add_message("user", str(message.content))
        elif isinstance(message, AIMessage) or message.type == "ai":
            self._sync_wrapper.add_message("assistant", str(message.content))

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Bulk addition hook used by LCEL RunnableWithMessageHistory."""
        for msg in messages:
            self.add_message(msg)

    def clear(self) -> None:
        """Clears memory states by cycling the underlying thread context portal."""
        self.close_portal()
        self.init_portal()
