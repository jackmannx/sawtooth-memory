"""
tests/test_langchain_memory.py

Verifies the modern LCEL BaseChatMessageHistory adapter mechanics.
"""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sawtooth_memory.config import ContextManagerConfig
from sawtooth_memory.integrations.langchain_adapter import SawtoothChatMessageHistory


def test_sawtooth_chat_message_history_lifecycle(tmp_path: Path):
    """Verify that object mode accurately parses dictionaries to LangChain BaseMessages."""
    config = ContextManagerConfig(soft_limit_tokens=1000)
    journal_file = tmp_path / "lc_audit.jsonl"

    history = SawtoothChatMessageHistory(
        system_prompt="You are a history agent.",
        config=config,
        enable_events=True,
        journal_path=journal_file,
    )

    # 1. Initial State (Should contain only the system prompt compiled)
    msgs = history.messages
    assert len(msgs) == 1
    assert isinstance(msgs[0], SystemMessage)
    assert "You are a history agent." in msgs[0].content

    # 2. Add messages via the standard LangChain hooks
    history.add_messages([HumanMessage(content="Ping"), AIMessage(content="Pong")])

    # 3. Verify State
    msgs = history.messages
    assert len(msgs) == 3

    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content == "Ping"

    assert isinstance(msgs[2], AIMessage)
    assert msgs[2].content == "Pong"

    # 4. Verify Clear operation wipes execution structures cleanly
    history.clear()

    cleared_msgs = history.messages
    assert len(cleared_msgs) == 1
    assert isinstance(cleared_msgs[0], SystemMessage)
