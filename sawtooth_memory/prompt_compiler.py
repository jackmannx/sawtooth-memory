"""
prompt_compiler.py — Shared prompt assembly for async and sync ContextManagers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ContextManagerConfig
from .state import MemoryState


@dataclass
class PromptCompileResult:
    """Compiled OpenAI-compatible messages and optional L3 retrieval metadata."""

    messages: list[dict[str, str]]
    l3_retrieval: list[dict[str, Any]] = field(default_factory=list)


def compile_prompt(
    state: MemoryState,
    config: ContextManagerConfig,
    *,
    l3_block: str = "",
    l3_retrieval: list[dict[str, Any]] | None = None,
) -> PromptCompileResult:
    """
    Compile all memory tiers into an OpenAI-compatible messages list.

    Structure of the injected system message:
        [SYSTEM_L0]
        [ARCHIVE_L2]          (omitted if empty)
        [ARCHIVE_L3]          (omitted if empty)
        [ENTITY_LEDGER_L1_5]  (omitted if empty)

    Followed by raw Working Memory (L1) messages.
    """
    system_parts: list[str] = []

    system_parts.append(f"[SYSTEM_L0]\n{state.l0_system.content}")

    if state.l2_archival.narrative.strip():
        system_parts.append(f"[ARCHIVE_L2]\n{state.l2_archival.narrative.strip()}")

    if l3_block.strip():
        system_parts.append(f"[ARCHIVE_L3]\n{l3_block.strip()}")

    if state.l1_5_entities.entities:
        system_parts.append(
            f"[ENTITY_LEDGER_L1_5]\n{state.l1_5_entities.to_json_str()}"
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]

    for msg in state.l1_working.messages:
        messages.append(msg.to_openai_dict())

    return PromptCompileResult(
        messages=messages,
        l3_retrieval=list(l3_retrieval or []),
    )


def resolve_l3_retrieval_query(
    state: MemoryState,
    retrieval_query: str | None,
) -> str | None:
    """Use explicit query or fall back to the last user message in L1."""
    if retrieval_query:
        return retrieval_query
    for msg in reversed(state.l1_working.messages):
        if msg.role == "user":
            return msg.content
    return None


def format_l3_retrieval_block(
    results: list[Any],
    *,
    token_budget: int,
    count_text: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Format semantic search hits into a prompt block within *token_budget*.

    *count_text* is a callable ``(text: str) -> int`` (typically TokenMonitor.count_text).
    """
    if not results:
        return "", []

    block_lines: list[str] = []
    retrieval_meta: list[dict[str, Any]] = []
    current_tokens = 0

    for i, res in enumerate(results, 1):
        line = f"{i}. {res.text}"
        line_tokens = count_text(line)
        if current_tokens + line_tokens > token_budget and current_tokens > 0:
            break

        block_lines.append(line)
        current_tokens += line_tokens
        retrieval_meta.append({
            "text": res.text,
            "similarity": res.similarity,
            "origin": "L3 Semantic Retrieval",
        })

    if not block_lines:
        return "", []

    return "\n".join(block_lines), retrieval_meta
