"""
salience.py — Lightweight local entity salience extraction.

Catches unstructured but critical identifiers (tracking codes, ticket IDs,
reference numbers) that rigid regex patterns miss, using cue-word proximity,
structural shape, entropy, and rarity heuristics. Runs entirely in-process.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Set

# Cue words → inferred ledger key (longer phrases checked first)
_CUE_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("tracking code", "tracking_code"),
    ("tracking number", "tracking_code"),
    ("reference number", "reference_id"),
    ("transaction id", "transaction_id"),
    ("api key", "api_key"),
    ("case number", "case_id"),
    ("order number", "order_id"),
    ("ticket number", "ticket_id"),
    ("incident id", "incident_id"),
    ("customer ref", "customer_ref"),
    ("tracking", "tracking_code"),
    ("reference", "reference_id"),
    ("transaction", "transaction_id"),
    ("incident", "incident_id"),
    ("ticket", "ticket_id"),
    ("order", "order_id"),
    ("token", "token"),
    ("secret", "secret"),
    ("ref", "reference_id"),
    ("code", "code"),
    ("id", "identifier"),
)

# Broad structural patterns for identifier-like spans
_CANDIDATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.VERBOSE)
    for p in (
        # PREFIX-value: cust_ref: JSMITH-2024
        r"\b[a-zA-Z][a-zA-Z0-9_]*\s*:\s*([A-Za-z0-9][-A-Za-z0-9_]{2,})\b",
        # INC-4421, REF-ABC12
        r"\b[A-Z]{2,}[-_][A-Z0-9][-A-Z0-9_]+\b",
        # txn_998877_alpha_omega
        r"\b[a-z]{2,}_[a-z0-9_]{3,}\b",
        # Alpha-991, Case-2024
        r"\b[A-Z][a-z]+-\d+\b",
        # word-digits-word: alpha-99-omega
        r"\b[A-Za-z0-9]{2,}[-_][A-Za-z0-9]{2,}[-_][A-Za-z0-9]+\b",
        # Mixed alphanumeric codes (6-48 chars, must have both letters and digits)
        r"\b(?=[A-Za-z0-9_-]{6,48}\b)(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9][-A-Za-z0-9_]+\b",
    )
)

_COMMON_WORDS: frozenset[str] = frozenset(
    {
        "assistant",
        "compress",
        "compression",
        "context",
        "conversation",
        "database",
        "default",
        "error",
        "example",
        "following",
        "message",
        "please",
        "quantum",
        "summary",
        "system",
        "thanks",
        "there",
        "these",
        "think",
        "turn",
        "user",
        "would",
    }
)


@dataclass(frozen=True)
class SalienceConfig:
    """Tunable thresholds for the salience extractor."""

    threshold: float = 0.5
    max_entities: int = 20
    cue_window_chars: int = 40


def _shannon_entropy(token: str) -> float:
    """Normalised Shannon entropy in [0, 1] for a token."""
    if not token:
        return 0.0
    counts = Counter(token.lower())
    length = len(token)
    entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
    max_entropy = math.log2(min(length, 36))  # cap at alphanumeric alphabet
    return min(entropy / max(max_entropy, 1e-9), 1.0)


def _structural_id_likeness(token: str) -> float:
    """Score how much *token* looks like a copy-exact identifier."""
    score = 0.0
    has_alpha = any(c.isalpha() for c in token)
    has_digit = any(c.isdigit() for c in token)
    if has_alpha and has_digit:
        score += 0.35
    if "_" in token or "-" in token:
        score += 0.2
    if token.isupper() and len(token) >= 4:
        score += 0.15
    if re.match(r"^[a-z]{2,}_", token):
        score += 0.25
    if re.match(r"^[A-Z]{2,}[-_]", token):
        score += 0.25
    if ":" in token:
        score += 0.1
    return min(score, 1.0)


def _cue_proximity(text: str, span: str, window: int) -> tuple[float, str]:
    """Return (score, inferred_key) based on cue words near *span*."""
    lower_text = text.lower()
    span_lower = span.lower()
    pos = lower_text.find(span_lower)
    if pos < 0:
        return 0.0, "identifier"

    start = max(0, pos - window)
    end = min(len(lower_text), pos + len(span) + window)
    context = lower_text[start:end]

    best_score = 0.0
    best_key = "identifier"
    for cue, key in _CUE_KEY_MAP:
        if cue in context:
            # Longer cues are more specific — prefer them
            cue_score = min(0.3 + len(cue) * 0.02, 0.7)
            if cue_score > best_score:
                best_score = cue_score
                best_key = key
    return best_score, best_key


def _rarity_score(token: str, text: str) -> float:
    """Rare tokens in the chunk are more likely to be needles."""
    count = text.lower().count(token.lower())
    if count <= 1:
        return 0.3
    if count == 2:
        return 0.15
    return 0.0


def score_candidate(token: str, text: str, *, cue_window: int = 40) -> tuple[float, str]:
    """Return (salience_score, inferred_key) for a candidate token."""
    clean = token.strip().strip("`'\".,;:!?()[]{}")
    if not clean or len(clean) < 4:
        return 0.0, "identifier"

    lower = clean.lower()
    if lower in _COMMON_WORDS:
        return 0.0, "identifier"

    # Pure numbers without context are usually not identifiers
    if clean.isdigit():
        return 0.0, "identifier"

    cue_score, inferred_key = _cue_proximity(text, clean, cue_window)
    structural = _structural_id_likeness(clean)
    entropy = _shannon_entropy(clean)
    rarity = _rarity_score(clean, text)

    score = (
        0.30 * cue_score
        + 0.30 * structural
        + 0.20 * entropy
        + 0.20 * rarity
    )
    return min(score, 1.0), inferred_key


def _collect_candidates(text: str) -> Set[str]:
    """Gather identifier-like spans from broad structural patterns."""
    found: Set[str] = set()
    for pattern in _CANDIDATE_PATTERNS:
        for match in pattern.finditer(text):
            if match.lastindex:
                found.add(match.group(1))
            else:
                found.add(match.group(0))
    return found


def _disambiguate_key(base_key: str, used_keys: Set[str]) -> str:
    if base_key not in used_keys:
        return base_key
    idx = 2
    while f"{base_key}_{idx}" in used_keys:
        idx += 1
    return f"{base_key}_{idx}"


class SalienceEntityExtractor:
    """Heuristic extractor for unstructured critical entities."""

    def __init__(self, config: SalienceConfig | None = None) -> None:
        self._config = config or SalienceConfig()

    def extract(
        self,
        text: str,
        *,
        exclude_values: Iterable[str] | None = None,
    ) -> Dict[str, str]:
        """Return salient entities above the configured threshold."""
        excluded = {v.lower() for v in (exclude_values or ())}
        candidates = _collect_candidates(text)
        scored: list[tuple[float, str, str]] = []

        for candidate in candidates:
            if candidate.lower() in excluded:
                continue
            score, key = score_candidate(
                candidate, text, cue_window=self._config.cue_window_chars
            )
            if score >= self._config.threshold:
                scored.append((score, key, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        result: Dict[str, str] = {}
        used_keys: Set[str] = set()

        for _, key, value in scored[: self._config.max_entities]:
            final_key = _disambiguate_key(key, used_keys)
            used_keys.add(final_key)
            result[final_key] = value

        return result
