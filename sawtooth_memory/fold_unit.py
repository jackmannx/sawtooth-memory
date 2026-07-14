"""Zero-LLM structured trajectory folding for DTE mode."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .compression_core import messages_to_text
from .ner import NERPipeline, active_strategy_context
from .state import MemoryState, Message

_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class FoldUnit:
    cycle_id: str
    stub: str
    messages_text: str
    tokens_evicted: int
    entity_keys: tuple[str, ...]
    l3_chunks: int


def create_fold_unit(
    messages: list[Message],
    state: MemoryState,
    ner_pipeline: NERPipeline,
    *,
    cycle_id: str,
    l3_chunks: int = 0,
    enable_ner: bool = True,
    messages_text: str | None = None,
    mark_l3_recoverable: bool = False,
) -> FoldUnit:
    """Externalize an evicted trajectory into L1.5, L2 and DTE accounting.

    When *mark_l3_recoverable* is True, the stub records ``l3=<cycle_id>`` even
    if indexing has not completed yet (async finalize path).
    """
    text = messages_text if messages_text is not None else messages_to_text(messages)
    entity_keys: tuple[str, ...] = ()

    if enable_ner:
        extraction = ner_pipeline.extract_with_metadata(text)
        entity_keys = tuple(sorted(extraction.entities))
        if extraction.entities:
            token = active_strategy_context.set(extraction.strategies)
            try:
                state.l1_5_entities.upsert(extraction.entities)
            finally:
                active_strategy_context.reset(token)

    tokens_evicted = sum(message.token_count for message in messages)
    outcome = _extract_outcome_stub(messages)
    entities = ",".join(entity_keys[:8]) or "none"
    l3_ref = (
        cycle_id
        if l3_chunks > 0 or mark_l3_recoverable
        else "unavailable"
    )
    stub = (
        f"[FOLD n={len(messages)} tokens={tokens_evicted} "
        f"entities={entities} l3={l3_ref}] {outcome}"
    )
    state.l2_archival.append_narrative(stub)
    state.dte.narrative_debt_tokens += tokens_evicted
    state.dte.folds_since_narrative += 1
    state.dte.fold_cycles += 1

    return FoldUnit(
        cycle_id=cycle_id,
        stub=stub,
        messages_text=text,
        tokens_evicted=tokens_evicted,
        entity_keys=entity_keys,
        l3_chunks=l3_chunks,
    )


def fold_lines(narrative: str) -> list[str]:
    """Return structured fold lines eligible for later consolidation."""
    return [line for line in narrative.splitlines() if line.startswith("[FOLD ")]


def remove_fold_lines(narrative: str) -> str:
    """Remove fold units after a successful narrative consolidation."""
    return "\n".join(
        line for line in narrative.splitlines() if not line.startswith("[FOLD ")
    ).strip()


def _extract_outcome_stub(messages: list[Message], max_chars: int = 240) -> str:
    for message in reversed(messages):
        text = _SPACE.sub(" ", message.content).strip()
        if text:
            if len(text) > max_chars:
                text = f"{text[: max_chars - 1].rstrip()}…"
            return f"{message.role}: {text}"
    return "No textual outcome retained."
