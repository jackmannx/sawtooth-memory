"""
providers/adapters.py — ProviderAdapter Protocol + concrete implementations.

Each adapter is a pure data-construction object: it builds the correct HTTP
headers and request payload for its provider and knows how to parse the
response back into a normalised structure.  No I/O happens here.

The compression system prompt is defined once here and shared across
adapters so every provider receives identical instructions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sawtooth_memory.compression_utils import parse_compression_json


@runtime_checkable
class ProviderAdapter(Protocol):
    """
    Interface that every cloud provider adapter must satisfy.

    Adopters of this Protocol are pure payload-construction objects;
    they perform no I/O.
    """

    @property
    def endpoint(self) -> str:
        """Returns the default API endpoint for the provider."""
        ...

    def build_headers(self, api_key: str) -> dict:
        """Constructs provider-specific authentication headers."""
        ...

    def build_payload(self, model: str, system_prompt: str, content: str) -> dict:
        """
        Constructs the exact JSON payload, strictly enforcing
        JSON / Structured Output mode per the provider's spec.
        """
        ...

    def parse_response(self, response_data: dict) -> tuple[dict, int]:
        """
        Extracts the structured JSON dict and total token count from the
        raw provider response object.

        Returns:
            (parsed_json_dict, total_tokens_used)
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class OpenAIAdapter:
    """
    Adapter for the OpenAI Chat Completions API.

    Enforces JSON output via ``"response_format": {"type": "json_object"}``.
    Compatible with OpenAI-compatible proxies (Helicone, LiteLLM, Azure).

    Endpoint: POST /v1/chat/completions
    """

    _DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url

    @property
    def endpoint(self) -> str:
        if self._base_url:
            # Strip trailing slash; callers append the path.
            return self._base_url.rstrip("/") + "/v1/chat/completions"
        return self._DEFAULT_ENDPOINT

    def build_headers(self, api_key: str) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def build_payload(self, model: str, system_prompt: str, content: str) -> dict:
        return {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }

    def parse_response(self, response_data: dict) -> tuple[dict, int]:
        choices: list[dict] = response_data.get("choices", [])
        raw_text: str = ""
        if choices:
            raw_text = (
                choices[0].get("message", {}).get("content", "")
            )
        usage: dict = response_data.get("usage", {})
        total_tokens: int = usage.get("total_tokens", 0)

        parsed = parse_compression_json(raw_text)
        return parsed, total_tokens


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

# JSON extraction tool schema — forces the model to return structured output
# matching our required schema rather than wrapping it in prose.
_ANTHROPIC_TOOL_SCHEMA = {
    "name": "store_compression_result",
    "description": (
        "Stores the result of memory compression. "
        "Call this tool with the narrative summary and extracted entities."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "narrative_summary": {
                "type": "string",
                "description": (
                    "Dense chronological narrative of what the agent "
                    "decided, discovered, and accomplished."
                ),
            },
            "extracted_entities": {
                "type": "object",
                "description": (
                    "Flat key-value mapping of exact deterministic values "
                    "(UUIDs, IDs, paths, connection strings, numeric results)."
                ),
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["narrative_summary", "extracted_entities"],
    },
}


class AnthropicAdapter:
    """
    Adapter for the Anthropic Messages API.

    Uses the ``tools`` parameter with ``tool_choice: {"type": "tool"}`` to
    guarantee that the model calls our ``store_compression_result`` tool,
    yielding a strictly-typed JSON schema rather than free-form text.

    This is the most reliable way to enforce structured output on Anthropic;
    native JSON mode (``betas``) exists but tool-use is stable and GA.

    Endpoint: POST /v1/messages
    """

    _DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages"
    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url

    @property
    def endpoint(self) -> str:
        if self._base_url:
            return self._base_url.rstrip("/") + "/v1/messages"
        return self._DEFAULT_ENDPOINT

    def build_headers(self, api_key: str) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self._ANTHROPIC_VERSION,
        }

    def build_payload(self, model: str, system_prompt: str, content: str) -> dict:
        return {
            "model": model,
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": content},
            ],
            "tools": [_ANTHROPIC_TOOL_SCHEMA],
            "tool_choice": {
                "type": "tool",
                "name": "store_compression_result",
            },
        }

    def parse_response(self, response_data: dict) -> tuple[dict, int]:
        # With tool_choice forced, the first content block is always tool_use.
        content_blocks: list[dict] = response_data.get("content", [])
        tool_input: dict = {}

        for block in content_blocks:
            if block.get("type") == "tool_use":
                raw_input = block.get("input", {})
                # input is already a parsed dict when using the tools API,
                # but apply JSON safety for proxy/edge-case responses.
                if isinstance(raw_input, str):
                    tool_input = parse_compression_json(raw_input)
                else:
                    tool_input = raw_input
                break

        # If somehow no tool_use block arrived, fall back to text extraction.
        if not tool_input:
            for block in content_blocks:
                if block.get("type") == "text":
                    tool_input = parse_compression_json(block.get("text", ""))
                    break

        usage: dict = response_data.get("usage", {})
        total_tokens: int = usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        )

        return tool_input, total_tokens


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------


class GeminiAdapter:
    """
    Adapter for the Google Gemini generateContent API.

    Enforces JSON output via
    ``"generationConfig": {"responseMimeType": "application/json"}``.

    Endpoint: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    Note: the model is embedded in the URL, not the payload body.
    """

    _DEFAULT_BASE = "https://generativelanguage.googleapis.com"
    _PATH_TEMPLATE = "/v1beta/models/{model}:generateContent"

    def __init__(self, model: str = "", base_url: str | None = None) -> None:
        # We need the model at endpoint construction time for Gemini.
        self._model = model
        self._base_url = base_url

    @property
    def endpoint(self) -> str:
        base = (self._base_url or self._DEFAULT_BASE).rstrip("/")
        return base + self._PATH_TEMPLATE.format(model=self._model)

    def build_headers(self, api_key: str) -> dict:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

    def build_payload(self, model: str, system_prompt: str, content: str) -> dict:
        # Keep model in sync if set via build_payload (e.g. from CloudCompressor).
        self._model = model
        return {
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": content}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
            },
        }

    def parse_response(self, response_data: dict) -> tuple[dict, int]:
        candidates: list[dict] = response_data.get("candidates", [])
        raw_text: str = ""

        if candidates:
            parts: list[dict] = (
                candidates[0].get("content", {}).get("parts", [])
            )
            if parts:
                raw_text = parts[0].get("text", "")

        usage_meta: dict = response_data.get("usageMetadata", {})
        total_tokens: int = usage_meta.get("totalTokenCount", 0)

        parsed = parse_compression_json(raw_text)
        return parsed, total_tokens
