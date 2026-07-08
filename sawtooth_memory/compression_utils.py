"""Shared compression prompt and output parsing utilities."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

COMPRESSION_SYSTEM_PROMPT = """\
You are a memory compression engine for an AI agent system.

Your job:
1. Read the conversational logs provided.
2. Write a dense, chronological NARRATIVE of what the agent decided, discovered, \
and accomplished. Be specific. Preserve causality (why things happened).
3. Extract all EXACT DETERMINISTIC VALUES into a flat key-value dictionary. \
This includes: UUIDs, database IDs, file paths, connection strings, precise \
numeric results, API endpoints, and any other value that must be reproduced \
exactly in future tool calls.

Rules:
- IGNORE errors/exceptions if they were subsequently resolved.
- Do NOT include verbose JSON payloads or base64 strings.
- Use snake_case keys in extracted_entities.
- Respond ONLY with valid JSON. No preamble, no markdown fences, no extra text.

Required output schema:
{
  "narrative_summary": "<dense chronological narrative as a single string>",
  "extracted_entities": {
    "<key>": "<exact_value>"
  }
}
"""

_MD_FENCE_RE = re.compile(r"```(?:json)?\s*", re.IGNORECASE)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown JSON fences that some models emit despite JSON mode."""
    return _MD_FENCE_RE.sub("", text).strip()


def parse_compression_json(raw: str) -> dict[str, Any]:
    """
    Best-effort JSON extraction with fallback layers:
      1. Strip markdown fences and try json.loads directly.
      2. Find the outermost {...} block and retry json.loads.
      3. Return the raw text as the narrative.
    """
    cleaned = strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"narrative_summary": raw, "extracted_entities": {}}


def normalize_compression_result(
    parsed: dict[str, Any],
    *,
    raw_fallback: str | None = None,
) -> dict[str, str | dict[str, str]]:
    """Coerce parsed JSON into the canonical compression return shape."""
    narrative = parsed.get("narrative_summary", "")
    if not narrative and raw_fallback is not None:
        narrative = raw_fallback

    entities = parsed.get("extracted_entities", {})
    if not isinstance(entities, dict):
        entities = {}
    entities = {str(k): str(v) for k, v in entities.items()}

    return {
        "narrative_summary": narrative,
        "extracted_entities": entities,
    }
