"""
state.py — Pydantic v2 schemas for Sawtooth-Memory's tiered context model.

Tiers:
  L0  — SystemPrompt     : Immutable. Agent persona + tool schemas.
  L1  — WorkingMemory    : Mutable. Last N raw messages.
  L1.5— EntityLedger     : KV store. Exact deterministic values (IDs, paths, etc).
  L2  — ArchivalMemory   : Append-only. Dense narrative of compressed history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """A single turn in Working Memory (L1)."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    token_count: int = 0

    def to_openai_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class SystemPrompt(BaseModel):
    """L0 — Immutable system prompt. Set once, never compressed."""

    content: str
    token_count: int = 0


class WorkingMemory(BaseModel):
    """L1 — Sliding window of recent raw messages."""

    messages: list[Message] = Field(default_factory=list)
    token_count: int = 0

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self.token_count += message.token_count

    def slice_oldest(self, n: int) -> list[Message]:
        """Remove and return the oldest n messages, updating token count."""
        chunk = self.messages[:n]
        self.messages = self.messages[n:]
        self.token_count = sum(m.token_count for m in self.messages)
        return chunk


class EntityLedger(BaseModel):
    """L1.5 — Key-value store of exact deterministic values extracted from history."""

    entities: dict[str, str] = Field(default_factory=dict)

    def upsert(self, new_entities: dict[str, str]) -> None:
        self.entities.update(new_entities)

    def to_json_str(self) -> str:
        import json

        return json.dumps(self.entities, indent=2)


class ArchivalMemory(BaseModel):
    """L2 — Append-only narrative. Chronological summary of compressed history."""

    narrative: str = ""
    token_count: int = 0

    def append_narrative(self, new_text: str) -> None:
        if not new_text.strip():
            return
        self.narrative = (
            f"{self.narrative}\n{new_text}" if self.narrative else new_text
        )


class MemoryState(BaseModel):
    """Root state object. Holds all four memory tiers."""

    l0_system: SystemPrompt
    l1_working: WorkingMemory = Field(default_factory=WorkingMemory)
    l1_5_entities: EntityLedger = Field(default_factory=EntityLedger)
    l2_archival: ArchivalMemory = Field(default_factory=ArchivalMemory)
