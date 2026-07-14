"""Lightweight intent-aware prompt budgeting for DTE."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .config import ContextManagerConfig
from .state import EntityLedger

Intent = Literal["entity_lookup", "causal_why", "recap", "tool_retry", "general"]

_ENTITY_SHAPE = re.compile(
    r"\b(?:[A-Z]{2,}[-_][A-Z0-9_-]+|[a-z]+_id|uuid|id|code|path|url|token)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class PromptIntentPlan:
    intent: Intent
    include_l2: bool
    ledger_covers_query: bool
    l3_top_k: int
    l3_token_budget: int

    @property
    def prefers_narrative(self) -> bool:
        return self.intent in ("causal_why", "recap")


def plan_prompt(
    query: str | None,
    ledger: EntityLedger,
    config: ContextManagerConfig,
) -> PromptIntentPlan:
    """Classify query intent and allocate the smallest useful memory payload."""
    text = (query or "").strip()
    lowered = text.casefold()

    if any(term in lowered for term in ("why ", "reason", "because", "decision")):
        intent: Intent = "causal_why"
    elif any(
        term in lowered
        for term in ("recap", "summarize", "summary", "what happened", "so far")
    ):
        intent = "recap"
    elif any(
        term in lowered
        for term in ("retry", "rerun", "run again", "last command", "failed command")
    ):
        intent = "tool_retry"
    elif _ENTITY_SHAPE.search(text):
        intent = "entity_lookup"
    else:
        intent = "general"

    covered = ledger_covers(text, ledger)
    include_l2 = True
    if (
        config.omit_l2_when_ledger_covers
        and covered
        and intent in ("entity_lookup", "tool_retry")
    ):
        include_l2 = False

    top_k = config.l3_retrieval_top_k
    token_budget = config.l3_retrieval_max_tokens
    if intent in ("entity_lookup", "tool_retry"):
        top_k = max(1, min(top_k, 2))
        token_budget = max(64, token_budget // 2)
    elif intent in ("causal_why", "recap"):
        top_k = max(top_k, min(6, top_k * 2))
        token_budget = max(token_budget, min(1500, token_budget * 2))

    return PromptIntentPlan(
        intent=intent,
        include_l2=include_l2,
        ledger_covers_query=covered,
        l3_top_k=top_k,
        l3_token_budget=token_budget,
    )


def ledger_covers(query: str, ledger: EntityLedger) -> bool:
    query_terms = {term.casefold() for term in _WORD.findall(query)}
    if not query_terms:
        return False
    return bool(query_terms & ledger_term_set(ledger))


def ledger_term_set(ledger: EntityLedger) -> set[str]:
    """Tokenize ledger keys/values once for coverage checks."""
    known_terms: set[str] = set()
    for key, history in ledger.entities.items():
        known_terms.update(term.casefold() for term in _WORD.findall(key))
        for value in history:
            known_terms.update(term.casefold() for term in _WORD.findall(value))
    return known_terms
