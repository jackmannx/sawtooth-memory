"""Synthetic conversation generators and recall needle definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class RecallNeedle:
    """A fact injected into a conversation that must survive compression."""

    key: str
    value: str
    turn_index: int
    category: str


NEEDLE_SUITE: tuple[RecallNeedle, ...] = (
    RecallNeedle(
        key="transaction_id",
        value="txn_998877_alpha_omega",
        turn_index=2,
        category="custom_id",
    ),
    RecallNeedle(
        key="config_path",
        value="/etc/nginx/sites-enabled/api.conf",
        turn_index=4,
        category="file_path",
    ),
    RecallNeedle(
        key="runbook_uri",
        value="https://internal.corp/runbooks/inc-4421",
        turn_index=6,
        category="uri",
    ),
    RecallNeedle(
        key="aws_arn",
        value="arn:aws:s3:us-east-1:123456789012:bucket/prod-data",
        turn_index=8,
        category="custom_pattern",
    ),
)


def _filler_user(turn: int, size: str) -> str:
    if size == "small":
        return f"Tell me about quantum entanglement. (Turn {turn})"
    if size == "medium":
        return (
            f"Explain quantum field theory, wave-function collapse, and quark confinement "
            f"in detail for turn {turn}. " * 4
        )
    return (
        f"Provide an exhaustive lecture on quantum chromodynamics, lattice gauge theory, "
        f"and renormalization for turn {turn}. " * 12
    )


def _filler_assistant(turn: int, size: str) -> str:
    if size == "small":
        return f"Here is a concise explanation of quantum physics for turn {turn}."
    if size == "medium":
        return (
            f"Here is a detailed explanation of quarks, entanglement, and wave-function "
            f"collapse for turn {turn}. " * 4
        )
    return (
        f"Here is a comprehensive explanation of advanced quantum physics topics for "
        f"turn {turn}. " * 12
    )


def generate_conversation(
    turns: int = 10,
    *,
    message_size: str = "medium",
    needles: tuple[RecallNeedle, ...] = NEEDLE_SUITE,
    late_needle_turn: int | None = None,
) -> list[dict[str, MessageRole | str]]:
    """Build a synthetic multi-turn conversation with injected recall needles."""
    conversation: list[dict[str, MessageRole | str]] = []
    needle_turns = {n.turn_index: n for n in needles}
    if late_needle_turn is not None:
        needle_turns[late_needle_turn] = RecallNeedle(
            key="late_transaction_id",
            value="txn_late_needle_zz99",
            turn_index=late_needle_turn,
            category="late_injection",
        )

    for turn in range(turns):
        needle = needle_turns.get(turn)
        if needle:
            conversation.append(
                {
                    "role": "user",
                    "content": f"Remember this {needle.category}: {needle.value}",
                }
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"Stored {needle.key}={needle.value}.",
                }
            )
        else:
            conversation.append(
                {"role": "user", "content": _filler_user(turn, message_size)}
            )
            conversation.append(
                {"role": "assistant", "content": _filler_assistant(turn, message_size)}
            )

    return conversation


def all_needle_values(
    needles: tuple[RecallNeedle, ...] = NEEDLE_SUITE,
    *,
    late_needle_turn: int | None = None,
) -> list[str]:
    """Return every needle value that should appear in a compiled prompt."""
    values = [n.value for n in needles]
    if late_needle_turn is not None:
        values.append("txn_late_needle_zz99")
    return values
