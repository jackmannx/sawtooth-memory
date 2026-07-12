"""
entity_guard.py — Protection manifest, secure merge, and post-merge verification.

Ensures locally discovered critical entities survive LLM compression even when
the compression model drops or paraphrases them.
"""

from __future__ import annotations

from typing import Any

from .ner import EntityStrategy, ExtractionResult


def build_protected_entities(
    extraction: ExtractionResult,
    pinned_entities: dict[str, str] | None = None,
) -> dict[str, str]:
    """Collect all locally protected entities before compression."""
    protected = dict(extraction.entities)
    if pinned_entities:
        for key, value in pinned_entities.items():
            protected[key] = value
    return protected


def build_strategy_map(
    extraction: ExtractionResult,
    llm_entities: dict[str, str],
    pinned_entities: dict[str, str] | None = None,
) -> dict[str, EntityStrategy]:
    """Build per-key strategy map for telemetry."""
    strategy_map: dict[str, EntityStrategy] = dict(extraction.strategies)
    if pinned_entities:
        for key in pinned_entities:
            strategy_map[key] = "pinned"
    for key in llm_entities:
        if key not in strategy_map:
            strategy_map[key] = "llm_synthesis"
    return strategy_map


def secure_merge_entities(
    llm_entities: dict[str, str],
    protected_entities: dict[str, str],
) -> dict[str, str]:
    """Merge LLM entities with protected local entities (local wins on conflict)."""
    return {**llm_entities, **protected_entities}


def verify_protected_entities(
    protected_entities: dict[str, str],
    combined_entities: dict[str, str],
    narrative: str,
) -> dict[str, str]:
    """
    Re-inject protected values the LLM dropped from both entities and narrative.

    Returns entities that must be force-added to the ledger.
    """
    entity_values = set(combined_entities.values())
    narrative_lower = narrative.lower()
    reinjected: dict[str, str] = {}

    for key, value in protected_entities.items():
        if value in entity_values:
            continue
        if value.lower() in narrative_lower:
            continue
        reinjected[key] = value

    return reinjected


def apply_entity_guard(
    extraction: ExtractionResult,
    llm_entities: dict[str, str],
    narrative: str,
    *,
    pinned_entities: dict[str, str] | None = None,
    enable_verifier: bool = True,
) -> tuple[dict[str, str], dict[str, EntityStrategy]]:
    """
    Full entity guard pipeline: protect → merge → verify.

    Returns (final_entities, strategy_map).
    """
    protected = build_protected_entities(extraction, pinned_entities)
    combined = secure_merge_entities(llm_entities, protected)

    if enable_verifier:
        reinjected = verify_protected_entities(protected, combined, narrative)
        combined.update(reinjected)

    strategy_map = build_strategy_map(extraction, llm_entities, pinned_entities)
    return combined, strategy_map


def format_protection_manifest(protected_entities: dict[str, str]) -> str:
    """Format protected entities for injection into the compression prompt."""
    if not protected_entities:
        return ""
    lines = [
        "PROTECTED VALUES (reproduce exactly in extracted_entities; do not paraphrase):"
    ]
    for key, value in protected_entities.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_compression_user_content(
    pruned_text: str,
    protected_entities: dict[str, str] | None = None,
) -> str:
    """Build the user message sent to the compression LLM."""
    parts = ["Compress the following context logs:"]
    manifest = format_protection_manifest(protected_entities or {})
    if manifest:
        parts.append("")
        parts.append(manifest)
    parts.append("")
    parts.append(pruned_text)
    return "\n\n".join(parts)
