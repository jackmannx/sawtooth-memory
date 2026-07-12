"""
sync_compressor.py — Synchronous compression backends for SyncContextManager.

Uses httpx.Client (blocking) with the same pruning and parsing as the async
compressors in compressor.py.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from sawtooth_memory.compression_utils import (
    COMPRESSION_SYSTEM_PROMPT,
    normalize_compression_result,
    parse_compression_json,
    prune_compression_input,
)
from sawtooth_memory.config import CloudConfig, OllamaConfig
from sawtooth_memory.entity_guard import build_compression_user_content
from sawtooth_memory.exceptions import CompressionError, OllamaConnectionError
from sawtooth_memory.providers import ProviderAdapter, get_adapter

logger = logging.getLogger(__name__)


def _is_rate_limit(exc: BaseException) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 429
    )


class SyncOllamaCompressor:
    """Blocking Ollama client for inline compression cycles."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client: httpx.Client | None = None

    @property
    def model(self) -> str:
        return self._config.model

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
            )
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()
            logger.debug("SyncOllamaCompressor: HTTP client closed.")

    def ping(self) -> None:
        client = self._get_client()
        resp = client.get("/api/tags", timeout=5.0)
        resp.raise_for_status()

    def compress(
        self,
        messages_text: str,
        *,
        protected_entities: dict[str, str] | None = None,
    ) -> dict:
        pruned = prune_compression_input(messages_text)
        logger.debug(
            "SyncOllamaCompressor: pruned %d → %d chars",
            len(messages_text),
            len(pruned),
        )

        user_content = build_compression_user_content(pruned, protected_entities)
        payload = {
            "model": self._config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        client = self._get_client()
        try:
            resp = client.post("/api/chat", json=payload)
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
        parsed = parse_compression_json(raw_content)
        return normalize_compression_result(parsed, raw_fallback=raw_content)


class SyncCloudCompressor:
    """Blocking cloud LLM client for inline compression cycles."""

    def __init__(self, config: CloudConfig) -> None:
        self._config = config
        self._adapter: ProviderAdapter = get_adapter(config)
        self._client: httpx.Client | None = None

    @property
    def provider(self) -> str:
        return self._config.provider.value

    @property
    def model(self) -> str:
        return self._config.model

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self._config.timeout_seconds)
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()
            logger.debug("SyncCloudCompressor: HTTP client closed.")

    def ping(self) -> None:
        if not self._config.api_key.get_secret_value().strip():
            raise ValueError(
                f"{self._config.provider.value} API key is empty or missing."
            )

    def compress(
        self,
        messages_text: str,
        *,
        protected_entities: dict[str, str] | None = None,
    ) -> dict:
        pruned = prune_compression_input(messages_text)
        logger.debug(
            "SyncCloudCompressor: pruned %d → %d chars",
            len(messages_text),
            len(pruned),
        )

        content = build_compression_user_content(pruned, protected_entities)
        payload = self._adapter.build_payload(
            model=self._config.model,
            system_prompt=COMPRESSION_SYSTEM_PROMPT,
            content=content,
        )
        headers = self._adapter.build_headers(
            self._config.api_key.get_secret_value()
        )

        response_data = self._post_with_retry(
            url=self._adapter.endpoint,
            headers=headers,
            payload=payload,
        )
        parsed, total_tokens = self._adapter.parse_response(response_data)
        result = normalize_compression_result(parsed)
        result["total_tokens"] = total_tokens
        return result

    def _post_with_retry(self, url: str, headers: dict, payload: dict) -> dict:
        @retry(
            retry=retry_if_exception(_is_rate_limit),
            wait=wait_exponential(multiplier=1, min=2, max=32),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _attempt() -> dict:
            client = self._get_client()
            try:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise CompressionError(
                    f"SyncCloudCompressor: cannot reach {url}. Check your network "
                    "or CloudConfig.base_url."
                ) from exc
            except httpx.TimeoutException as exc:
                raise CompressionError(
                    f"SyncCloudCompressor: request timed out after "
                    f"{self._config.timeout_seconds}s."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429:
                    raise
                raise CompressionError(
                    f"SyncCloudCompressor: provider returned HTTP {status}: "
                    f"{exc.response.text[:400]}"
                ) from exc
            return resp.json()

        return _attempt()
