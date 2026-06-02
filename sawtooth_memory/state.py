"""
state.py — Pydantic v2 schemas for Sawtooth-Memory's tiered context model.

Tiers:
  L0  — SystemPrompt     : Immutable. Agent persona + tool schemas.
  L1  — WorkingMemory    : Mutable. Last N raw messages.
  L1.5— EntityLedger     : KV store. Exact deterministic values (IDs, paths, etc).
  L2  — ArchivalMemory   : Append-only. Dense narrative of compressed history.

Event integration:
  EntityLedger can be given an optional event callback that is invoked whenever
  an entity is inserted or updated. The callback receives (key, value, operation)
  where operation is "insert" or "update". This allows higher layers to emit
  telemetry without coupling the state module to the event system.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """A single turn in Working Memory (L1)."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    """L1.5 — High-fidelity key-value cache of exact deterministic values.

    Storage model
    -------------
    Each key maps to a **list** of observed values in chronological order
    (oldest first).  When the same key is extracted across multiple
    compression waves (e.g. ``connection_id`` first seen as ``"conn-A"``
    and later as ``"conn-B"``), both values are preserved rather than the
    earlier one being silently overwritten.

    Backwards-compatible accessors
    --------------------------------
    ``get_latest(key)``   — returns the most-recently observed value (the
                            old "current" semantic).
    ``get_history(key)``  — returns the full ordered value list.
    ``to_json_str()``     — renders each key as its *latest* value so the
                            prompt compiler produces the same compact KV
                            block it always has, but adds a ``_history``
                            sub-key whenever multiple distinct values exist
                            so agents can reference earlier values if needed.

    Rolling window
    --------------
    To prevent unbounded growth during very long sessions, each key retains
    at most ``max_history_per_key`` values (default 10).  When the list is
    full the oldest entry is evicted before the new value is appended.

    Event callback (Phase 2)
    ------------------------
    An optional callback can be set via ``set_event_callback()``. It will be
    invoked whenever an entity is inserted (new key) or updated (new value
    appended to an existing key). The callback signature is:
        callback(key: str, value: str, operation: Literal["insert", "update"])
    This allows higher layers (e.g., CompressionWorker) to emit telemetry events
    without coupling the state module to the event system.
    """

    entities: dict[str, list[str]] = Field(default_factory=dict)
    max_history_per_key: int = Field(default=10, exclude=True)
    _event_callback: Optional[
        Callable[[str, str, Literal["insert", "update"]], None]
    ] = Field(default=None, exclude=True)

    def set_event_callback(
        self,
        callback: Optional[Callable[[str, str, Literal["insert", "update"]], None]],
    ) -> None:
        """
        Set a callback that will be invoked whenever an entity is inserted or updated.

        Args:
            callback: Callable with signature (key, value, operation) or None to disable.
        """
        self._event_callback = callback

    def upsert(self, new_entities: dict[str, str]) -> None:
        """Merge *new_entities* into the ledger with conflict preservation.

        For every key in *new_entities*:
        - If the key is new, create a single-element list and invoke the
          event callback with operation "insert".
        - If the key already exists and the incoming value differs from the
          most-recently stored value, append the new value (up to
          ``max_history_per_key`` unique entries, evicting the oldest when
          the window is full). Invoke the event callback with operation "update".
        - If the incoming value is identical to the current latest value,
          the call is a no-op for that key (avoids duplicating identical
          extractions from overlapping compression waves).
        """
        for key, value in new_entities.items():
            value = str(value)
            if key not in self.entities:
                # New key → insert
                self.entities[key] = [value]
                if self._event_callback:
                    self._event_callback(key, value, "insert")
            else:
                history = self.entities[key]
                # Skip exact duplicate of the most-recent value.
                if history and history[-1] == value:
                    continue
                # Existing key gets a new value → update
                history.append(value)
                # Enforce the rolling window: evict the oldest entry.
                if len(history) > self.max_history_per_key:
                    self.entities[key] = history[-self.max_history_per_key :]
                if self._event_callback:
                    self._event_callback(key, value, "update")

    def get_latest(self, key: str) -> str | None:
        """Return the most-recently extracted value for *key*, or ``None``."""
        history = self.entities.get(key)
        return history[-1] if history else None

    def get_history(self, key: str) -> list[str]:
        """Return the full ordered history of values for *key* (oldest first)."""
        return list(self.entities.get(key, []))

    def to_json_str(self) -> str:
        """Serialise the ledger for prompt injection.

        Each key is rendered as its *latest* value so the compiled prompt
        remains compact and backward-compatible with the existing prompt
        format.  Keys that have accumulated multiple distinct values also
        receive a ``<key>__history`` companion entry so agents (and
        downstream tooling) can inspect the full provenance without having
        to parse anything special.
        """
        import json

        flat: dict[str, str] = {}
        for key, history in self.entities.items():
            if not history:
                continue
            flat[key] = history[-1]
            if len(history) > 1:
                # Surface earlier values as a lightweight audit trail.
                flat[f"{key}__history"] = " → ".join(history[:-1])

        return json.dumps(flat, indent=2)


class ArchivalMemory(BaseModel):
    """L2 — Append-only narrative. Chronological summary of compressed history."""

    narrative: str = ""
    token_count: int = 0

    def append_narrative(self, new_text: str) -> None:
        if not new_text.strip():
            return
        self.narrative = f"{self.narrative}\n{new_text}" if self.narrative else new_text


class MemoryState(BaseModel):
    """Root state object. Holds all four memory tiers."""

    l0_system: SystemPrompt
    l1_working: WorkingMemory = Field(default_factory=WorkingMemory)
    l1_5_entities: EntityLedger = Field(default_factory=EntityLedger)
    l2_archival: ArchivalMemory = Field(default_factory=ArchivalMemory)
