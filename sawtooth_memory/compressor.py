"""
compressor.py — Ollama-backed async compression engine.

Handles:
  1. Pre-processing: strips base64 blobs, stack traces, verbose JSON.
  2. Dual-extraction inference: sends pruned chunk to a local Ollama model.
  3. Output parsing: returns {"narrative_summary": ..., "extracted_entities": {...}}.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from sawtooth_memory.config import OllamaConfig
from sawtooth_memory.exceptions import CompressionError, OllamaConnectionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compression system prompt
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Pre-processing regexes
# ---------------------------------------------------------------------------

# Base64-like strings over 80 chars (avoids mangling normal text)
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")

# Python / JS stack traces
_STACKTRACE_RE = re.compile(
    r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)",
    re.DOTALL,
)

# Long runs of whitespace-separated hex (e.g. binary output)
_HEX_RE = re.compile(r"(?:[0-9a-fA-F]{2}\s){16,}")


def _prune(raw: str) -> str:
    """Strip noise that wastes compressor tokens without adding meaning."""
    text = _BASE64_RE.sub("[BASE64_REMOVED]", raw)
    text = _STACKTRACE_RE.sub("[STACKTRACE_REMOVED]", text)
    text = _HEX_RE.sub("[HEX_REMOVED]", text)
    return text


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class OllamaCompressor:
    """
    Async client for the local Ollama inference backend.

    Sends pruned message chunks to a small local model and returns a
    structured dict with 'narrative_summary' and 'extracted_entities'.
    """

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
            logger.debug("OllamaCompressor: HTTP client closed.")

    async def compress(self, messages_text: str) -> dict:
        """
        Prune and compress a raw message chunk.

        Returns:
            {
                "narrative_summary": str,
                "extracted_entities": dict[str, str],
            }

        Raises:
            OllamaConnectionError: if Ollama is unreachable.
            CompressionError: if the HTTP response indicates an error or
                              the request times out.
        """
        pruned = _prune(messages_text)
        logger.debug(
            f"OllamaCompressor: pruned {len(messages_text)} → {len(pruned)} chars"
        )

        payload = {
            "model": self._config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Compress the following context logs:\n\n{pruned}",
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
        except httpx.TimeoutException as exc:
            raise CompressionError(
                f"Ollama timed out after {self._config.timeout_seconds}s. "
                "Try a smaller model or increase timeout_seconds."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise CompressionError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        raw_content = resp.json().get("message", {}).get("content", "")
        return self._parse_output(raw_content)

    def _parse_output(self, content: str) -> dict:
        """
        Parse the model's JSON output. Applies light cleanup to handle
        common model quirks (markdown fences, leading text).
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(
                        "OllamaCompressor: could not parse model output as JSON; "
                        "storing raw text as narrative."
                    )
                    return {"narrative_summary": content, "extracted_entities": {}}
            else:
                return {"narrative_summary": content, "extracted_entities": {}}

        narrative = result.get("narrative_summary", "")
        entities = result.get("extracted_entities", {})

        if not isinstance(entities, dict):
            entities = {}
        entities = {str(k): str(v) for k, v in entities.items()}

        return {"narrative_summary": narrative, "extracted_entities": entities}


# ---------------------------------------------------------------------------
# Cloud compressor
# ---------------------------------------------------------------------------

import asyncio

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from sawtooth_memory.config import CloudConfig
from sawtooth_memory.providers import ProviderAdapter, get_adapter


def _is_rate_limit(exc: BaseException) -> bool:
    """Tenacity predicate — retry on HTTP 429 only."""
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 429
    )


class CloudCompressor:
    """
    Async compression engine backed by a cloud LLM provider
    (OpenAI, Anthropic, or Gemini).

    Implements the same ``compress(messages_text) → dict`` interface as
    ``OllamaCompressor`` so it can be dropped in as a replacement backend.

    Return value::

        {
            "narrative_summary":   str,
            "extracted_entities":  dict[str, str],
            "total_tokens":        int,
        }

    Retry policy
    ------------
    HTTP 429 (rate-limit) responses are retried up to 5 times with
    exponential back-off (2 s → 4 s → 8 s → 16 s → 32 s) via tenacity.

    Proxy support
    -------------
    Set ``CloudConfig.base_url`` to route traffic through Helicone,
    LiteLLM, Azure OpenAI, or any other OpenAI-compatible gateway.
    """

    def __init__(self, config: CloudConfig) -> None:
        self._config = config
        self._adapter: ProviderAdapter = get_adapter(config)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout_seconds,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("CloudCompressor: HTTP client closed.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def compress(self, messages_text: str) -> dict:
        """
        Prune, send to the cloud provider, and return a normalised result.

        Returns:
            {
                "narrative_summary":   str,
                "extracted_entities":  dict[str, str],
                "total_tokens":        int,
            }

        Raises:
            CompressionError: on HTTP errors, timeouts, or JSON parse failures.
        """
        pruned = _prune(messages_text)
        logger.debug(
            f"CloudCompressor: pruned {len(messages_text)} → {len(pruned)} chars"
        )

        content = f"Compress the following context logs:\n\n{pruned}"
        payload = self._adapter.build_payload(
            model=self._config.model,
            system_prompt=_SYSTEM_PROMPT,
            content=content,
        )
        headers = self._adapter.build_headers(
            self._config.api_key.get_secret_value()
        )

        response_data = await self._post_with_retry(
            url=self._adapter.endpoint,
            headers=headers,
            payload=payload,
        )

        parsed, total_tokens = self._adapter.parse_response(response_data)
        return self._normalise(parsed, total_tokens)

    # ------------------------------------------------------------------
    # HTTP execution with tenacity retry
    # ------------------------------------------------------------------

    async def _post_with_retry(
        self, url: str, headers: dict, payload: dict
    ) -> dict:
        """
        Execute an httpx POST with exponential back-off on HTTP 429.

        Tenacity cannot decorate async methods directly when they are
        instance methods (the ``self`` reference complicates pickling),
        so we use a local closure instead.
        """
        @retry(
            retry=retry_if_exception(_is_rate_limit),
            wait=wait_exponential(multiplier=1, min=2, max=32),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        async def _attempt() -> dict:
            client = await self._get_client()
            try:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise CompressionError(
                    f"CloudCompressor: cannot reach {url}. Check your network "
                    "or CloudConfig.base_url."
                ) from exc
            except httpx.TimeoutException as exc:
                raise CompressionError(
                    f"CloudCompressor: request timed out after "
                    f"{self._config.timeout_seconds}s."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429:
                    # Let tenacity handle the retry.
                    raise
                raise CompressionError(
                    f"CloudCompressor: provider returned HTTP {status}: "
                    f"{exc.response.text[:400]}"
                ) from exc
            return resp.json()

        return await _attempt()

    # ------------------------------------------------------------------
    # Output normalisation
    # ------------------------------------------------------------------

    def _normalise(self, parsed: dict, total_tokens: int) -> dict:
        """
        Coerce parsed JSON into the canonical return shape, mirroring
        the OllamaCompressor contract plus a ``total_tokens`` field.
        """
        narrative = parsed.get("narrative_summary", "")
        entities = parsed.get("extracted_entities", {})

        if not isinstance(entities, dict):
            entities = {}
        entities = {str(k): str(v) for k, v in entities.items()}

        return {
            "narrative_summary": narrative,
            "extracted_entities": entities,
            "total_tokens": total_tokens,
        }
