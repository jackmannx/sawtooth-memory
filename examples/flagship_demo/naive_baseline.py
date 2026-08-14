"""
naive_baseline.py — The memory pattern most agent loops ship with by default:
a flat message list that gets FIFO-truncated once it outgrows a token budget.

No entity tracking, no compression, no ledger. This is what "just append to
a list" looks like once a session runs long enough to matter — the thing
Sawtooth's DTE/entity-ledger stack exists to replace.
"""

from dataclasses import dataclass, field

import tiktoken

_ENCODING = tiktoken.encoding_for_model("gpt-4o")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass
class NaiveMemory:
    """Flat message list with FIFO eviction once max_tokens is exceeded."""

    system_prompt: str
    max_tokens: int = 900
    messages: list[dict] = field(default_factory=list)
    dropped_messages: list[dict] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        while self._token_count() > self.max_tokens and len(self.messages) > 1:
            self.dropped_messages.append(self.messages.pop(0))

    def _token_count(self) -> int:
        return sum(_count_tokens(m["content"]) for m in self.messages)

    def build_prompt(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]
