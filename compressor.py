"""
compressor.py — Ollama-backed async compression engine.

Sends a message chunk to a local Ollama model and returns a structured
dict with 'narrative_summary' and 'extracted_entities'.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from .config import OllamaConfig
from .exceptions import CompressionError, OllamaConnectionError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
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


class OllamaCompressor:
    """Async client for the local Ollama inference backend."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def compress(self, messages_text: str) -> dict:
        """
        Compress a raw message chunk via Ollama.

        Returns:
            {"narrative_summary": str, "extracted_entities": dict[str, str]}

        Raises:
            OllamaConnectionError: if Ollama is unreachable.
            CompressionError: on HTTP errors.
        """
        payload = {
            "model": self._config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Compress the following context logs:\n\n{messages_text}",
                },
            ],
        }

        client = await self._get_client()

        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self._config.base_url}. "
                "Is Ollama running? (`ollama serve`)"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise CompressionError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        raw_content = resp.json().get("message", {}).get("content", "")
        return self._parse_output(raw_content)

    def _parse_output(self, content: str) -> dict:
        """Parse the model's JSON output."""
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    return {"narrative_summary": content, "extracted_entities": {}}
            else:
                return {"narrative_summary": content, "extracted_entities": {}}

        narrative = result.get("narrative_summary", "")
        entities = result.get("extracted_entities", {})

        if not isinstance(entities, dict):
            entities = {}
        entities = {str(k): str(v) for k, v in entities.items()}

        return {"narrative_summary": narrative, "extracted_entities": entities}
