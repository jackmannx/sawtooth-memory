"""
tests/test_cloud_compressor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for:
  - ProviderAdapter Protocol compliance
  - OpenAIAdapter, AnthropicAdapter, GeminiAdapter — endpoint, headers,
    payload construction, and response parsing (with and without markdown fences)
  - CloudCompressor.compress() — mocked httpx, tenacity retry on 429,
    and result normalisation
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from sawtooth_memory.compressor import CloudCompressor
from sawtooth_memory.config import CloudConfig, Provider
from sawtooth_memory.exceptions import CompressionError
from sawtooth_memory.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    ProviderAdapter,
)
from sawtooth_memory.providers.adapters import _safe_parse_json
from sawtooth_memory.providers.factory import get_adapter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NARRATIVE = "Agent connected to PostgreSQL and found a 14% revenue drop in Q3."
ENTITIES = {"conn_id": "conn_994a82", "target_table": "sales_q3"}
CLEAN_JSON = json.dumps(
    {"narrative_summary": NARRATIVE, "extracted_entities": ENTITIES}
)
FENCED_JSON = f"```json\n{CLEAN_JSON}\n```"


@pytest.fixture
def openai_config():
    return CloudConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        api_key="sk-openai-test",
    )


@pytest.fixture
def anthropic_config():
    return CloudConfig(
        provider=Provider.ANTHROPIC,
        model="claude-3-5-haiku-latest",
        api_key="sk-ant-test",
    )


@pytest.fixture
def gemini_config():
    return CloudConfig(
        provider=Provider.GEMINI,
        model="gemini-1.5-flash",
        api_key="gemini-test-key",
    )


# ---------------------------------------------------------------------------
# Helper: build a minimal mock httpx response
# ---------------------------------------------------------------------------

def _mock_response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
        resp.text = json.dumps(body)
    else:
        resp.raise_for_status = MagicMock()
    return resp


# ===========================================================================
# 1.  _safe_parse_json (fallback defence)
# ===========================================================================

class TestSafeParseJson:
    def test_clean_json(self):
        result = _safe_parse_json(CLEAN_JSON)
        assert result["narrative_summary"] == NARRATIVE

    def test_markdown_fenced_json(self):
        result = _safe_parse_json(FENCED_JSON)
        assert result["narrative_summary"] == NARRATIVE

    def test_json_with_preamble_falls_back_to_brace_search(self):
        raw = "Here is the result:\n" + CLEAN_JSON
        result = _safe_parse_json(raw)
        assert result["narrative_summary"] == NARRATIVE

    def test_total_garbage_returns_raw_as_narrative(self):
        raw = "This is definitely not JSON."
        result = _safe_parse_json(raw)
        assert result["narrative_summary"] == raw
        assert result["extracted_entities"] == {}


# ===========================================================================
# 2.  ProviderAdapter Protocol
# ===========================================================================

class TestProviderAdapterProtocol:
    def test_openai_satisfies_protocol(self):
        assert isinstance(OpenAIAdapter(), ProviderAdapter)

    def test_anthropic_satisfies_protocol(self):
        assert isinstance(AnthropicAdapter(), ProviderAdapter)

    def test_gemini_satisfies_protocol(self):
        assert isinstance(GeminiAdapter(model="gemini-1.5-flash"), ProviderAdapter)


# ===========================================================================
# 3.  OpenAI adapter
# ===========================================================================

class TestOpenAIAdapter:
    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = OpenAIAdapter()

    # --- endpoint -----------------------------------------------------------

    def test_default_endpoint(self):
        assert self.adapter.endpoint == "https://api.openai.com/v1/chat/completions"

    def test_custom_base_url(self):
        a = OpenAIAdapter(base_url="https://oai.helicone.ai")
        assert a.endpoint == "https://oai.helicone.ai/v1/chat/completions"

    def test_custom_base_url_trailing_slash_stripped(self):
        a = OpenAIAdapter(base_url="https://oai.helicone.ai/")
        assert a.endpoint == "https://oai.helicone.ai/v1/chat/completions"

    # --- headers ------------------------------------------------------------

    def test_build_headers(self):
        headers = self.adapter.build_headers("sk-test")
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    # --- payload ------------------------------------------------------------

    def test_build_payload_enforces_json_mode(self):
        payload = self.adapter.build_payload("gpt-4o-mini", "SYS", "USER_CONTENT")
        assert payload["response_format"] == {"type": "json_object"}

    def test_build_payload_message_roles(self):
        payload = self.adapter.build_payload("gpt-4o-mini", "SYS", "USER_CONTENT")
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["system", "user"]

    def test_build_payload_model_field(self):
        payload = self.adapter.build_payload("gpt-4o-mini", "SYS", "U")
        assert payload["model"] == "gpt-4o-mini"

    # --- parse_response ------------------------------------------------------

    def _openai_response(self, content: str, total_tokens: int = 150) -> dict:
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": total_tokens},
        }

    def test_parse_clean_json(self):
        parsed, tokens = self.adapter.parse_response(self._openai_response(CLEAN_JSON, 150))
        assert parsed["narrative_summary"] == NARRATIVE
        assert parsed["extracted_entities"]["conn_id"] == "conn_994a82"
        assert tokens == 150

    def test_parse_fenced_json(self):
        parsed, tokens = self.adapter.parse_response(self._openai_response(FENCED_JSON))
        assert parsed["narrative_summary"] == NARRATIVE

    def test_parse_missing_usage_defaults_zero(self):
        response = {"choices": [{"message": {"content": CLEAN_JSON}}]}
        _, tokens = self.adapter.parse_response(response)
        assert tokens == 0

    def test_parse_empty_choices_returns_defaults(self):
        parsed, tokens = self.adapter.parse_response({"choices": [], "usage": {}})
        assert parsed["narrative_summary"] == ""
        assert tokens == 0


# ===========================================================================
# 4.  Anthropic adapter
# ===========================================================================

class TestAnthropicAdapter:
    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = AnthropicAdapter()

    # --- endpoint -----------------------------------------------------------

    def test_default_endpoint(self):
        assert self.adapter.endpoint == "https://api.anthropic.com/v1/messages"

    def test_custom_base_url(self):
        a = AnthropicAdapter(base_url="https://proxy.example.com")
        assert a.endpoint == "https://proxy.example.com/v1/messages"

    # --- headers ------------------------------------------------------------

    def test_build_headers(self):
        headers = self.adapter.build_headers("sk-ant-test")
        assert headers["x-api-key"] == "sk-ant-test"
        assert "anthropic-version" in headers
        assert headers["Content-Type"] == "application/json"

    # --- payload ------------------------------------------------------------

    def test_build_payload_uses_tools(self):
        payload = self.adapter.build_payload("claude-3-5-haiku-latest", "SYS", "U")
        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["name"] == "store_compression_result"

    def test_build_payload_tool_choice_forced(self):
        payload = self.adapter.build_payload("claude-3-5-haiku-latest", "SYS", "U")
        assert payload["tool_choice"]["type"] == "tool"
        assert payload["tool_choice"]["name"] == "store_compression_result"

    def test_build_payload_system_is_top_level(self):
        payload = self.adapter.build_payload("claude-3-5-haiku-latest", "SYS", "U")
        assert payload["system"] == "SYS"

    def test_build_payload_max_tokens_present(self):
        payload = self.adapter.build_payload("claude-3-5-haiku-latest", "SYS", "U")
        assert "max_tokens" in payload

    # --- parse_response  (tool_use block) ------------------------------------

    def _anthropic_response(
        self,
        tool_input: dict | str,
        input_tokens: int = 80,
        output_tokens: int = 40,
    ) -> dict:
        """Build a response that mimics a forced tool_use call."""
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "store_compression_result",
                    "input": tool_input,
                }
            ],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    def test_parse_tool_use_dict_input(self):
        response = self._anthropic_response(
            {"narrative_summary": NARRATIVE, "extracted_entities": ENTITIES}
        )
        parsed, tokens = self.adapter.parse_response(response)
        assert parsed["narrative_summary"] == NARRATIVE
        assert parsed["extracted_entities"]["conn_id"] == "conn_994a82"
        assert tokens == 120  # 80 + 40

    def test_parse_tool_use_string_input_strips_fences(self):
        """Some proxy layers serialise tool input as a JSON string."""
        response = self._anthropic_response(FENCED_JSON)
        parsed, tokens = self.adapter.parse_response(response)
        assert parsed["narrative_summary"] == NARRATIVE

    def test_parse_total_tokens_sum(self):
        response = self._anthropic_response(
            {"narrative_summary": "x", "extracted_entities": {}},
            input_tokens=100,
            output_tokens=55,
        )
        _, tokens = self.adapter.parse_response(response)
        assert tokens == 155

    def test_parse_fallback_to_text_block_when_no_tool_use(self):
        """If the model returned a text block instead of tool_use (edge-case), extract from text."""
        response = {
            "content": [{"type": "text", "text": CLEAN_JSON}],
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }
        parsed, _ = self.adapter.parse_response(response)
        assert parsed["narrative_summary"] == NARRATIVE

    def test_parse_empty_content_returns_defaults(self):
        response = {"content": [], "usage": {"input_tokens": 0, "output_tokens": 0}}
        parsed, tokens = self.adapter.parse_response(response)
        assert tokens == 0


# ===========================================================================
# 5.  Gemini adapter
# ===========================================================================

class TestGeminiAdapter:
    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = GeminiAdapter(model="gemini-1.5-flash")

    # --- endpoint -----------------------------------------------------------

    def test_default_endpoint_contains_model(self):
        assert "gemini-1.5-flash" in self.adapter.endpoint
        assert self.adapter.endpoint.endswith(":generateContent")

    def test_custom_base_url(self):
        a = GeminiAdapter(model="gemini-1.5-flash", base_url="https://my-proxy.io")
        assert a.endpoint.startswith("https://my-proxy.io")

    # --- headers ------------------------------------------------------------

    def test_build_headers(self):
        headers = self.adapter.build_headers("gemini-key-123")
        assert headers["x-goog-api-key"] == "gemini-key-123"
        assert headers["Content-Type"] == "application/json"

    # --- payload ------------------------------------------------------------

    def test_build_payload_json_mime_type(self):
        payload = self.adapter.build_payload("gemini-1.5-flash", "SYS", "U")
        assert payload["generationConfig"]["responseMimeType"] == "application/json"

    def test_build_payload_system_instruction(self):
        payload = self.adapter.build_payload("gemini-1.5-flash", "SYS", "U")
        assert payload["system_instruction"]["parts"][0]["text"] == "SYS"

    def test_build_payload_user_content(self):
        payload = self.adapter.build_payload("gemini-1.5-flash", "SYS", "HELLO")
        assert payload["contents"][0]["parts"][0]["text"] == "HELLO"

    # --- parse_response ------------------------------------------------------

    def _gemini_response(self, text: str, total_tokens: int = 200) -> dict:
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": text}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 150,
                "candidatesTokenCount": 50,
                "totalTokenCount": total_tokens,
            },
        }

    def test_parse_clean_json(self):
        parsed, tokens = self.adapter.parse_response(self._gemini_response(CLEAN_JSON, 200))
        assert parsed["narrative_summary"] == NARRATIVE
        assert parsed["extracted_entities"]["target_table"] == "sales_q3"
        assert tokens == 200

    def test_parse_fenced_json(self):
        parsed, _ = self.adapter.parse_response(self._gemini_response(FENCED_JSON))
        assert parsed["narrative_summary"] == NARRATIVE

    def test_parse_missing_candidates_returns_defaults(self):
        parsed, tokens = self.adapter.parse_response({"candidates": [], "usageMetadata": {}})
        assert parsed["narrative_summary"] == ""
        assert tokens == 0

    def test_parse_missing_usage_metadata(self):
        _, tokens = self.adapter.parse_response({"candidates": []})
        assert tokens == 0


# ===========================================================================
# 6.  Factory function
# ===========================================================================

class TestGetAdapter:
    def test_returns_openai_adapter(self, openai_config):
        adapter = get_adapter(openai_config)
        assert isinstance(adapter, OpenAIAdapter)

    def test_returns_anthropic_adapter(self, anthropic_config):
        adapter = get_adapter(anthropic_config)
        assert isinstance(adapter, AnthropicAdapter)

    def test_returns_gemini_adapter(self, gemini_config):
        adapter = get_adapter(gemini_config)
        assert isinstance(adapter, GeminiAdapter)


# ===========================================================================
# 7.  CloudCompressor.compress() — end-to-end with mocked httpx
# ===========================================================================

class TestCloudCompressorOpenAI:
    @pytest.fixture
    def compressor(self, openai_config):
        return CloudCompressor(openai_config)

    @pytest.mark.asyncio
    async def test_compress_success(self, compressor):
        body = {
            "choices": [{"message": {"content": CLEAN_JSON}}],
            "usage": {"total_tokens": 120},
        }
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_mock_response(body))
            mock_gc.return_value = mock_client

            result = await compressor.compress("USER: hello\nASSISTANT: hi")

        assert result["narrative_summary"] == NARRATIVE
        assert result["extracted_entities"]["conn_id"] == "conn_994a82"
        assert result["total_tokens"] == 120

    @pytest.mark.asyncio
    async def test_compress_connection_error_raises(self, compressor):
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_gc.return_value = mock_client

            with pytest.raises(CompressionError, match="cannot reach"):
                await compressor.compress("some text")

    @pytest.mark.asyncio
    async def test_compress_timeout_raises(self, compressor):
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_gc.return_value = mock_client

            with pytest.raises(CompressionError, match="timed out"):
                await compressor.compress("some text")

    @pytest.mark.asyncio
    async def test_non_429_http_error_raises_without_retry(self, compressor):
        err_resp = _mock_response({"error": "bad request"}, status_code=400)
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=err_resp)
            mock_gc.return_value = mock_client

            with pytest.raises(CompressionError):
                await compressor.compress("text")

    @pytest.mark.asyncio
    async def test_entities_cast_to_strings(self, compressor):
        body_json = json.dumps({
            "narrative_summary": "done",
            "extracted_entities": {"count": 42, "flag": True},
        })
        body = {
            "choices": [{"message": {"content": body_json}}],
            "usage": {"total_tokens": 50},
        }
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_mock_response(body))
            mock_gc.return_value = mock_client

            result = await compressor.compress("text")

        assert result["extracted_entities"]["count"] == "42"
        assert result["extracted_entities"]["flag"] == "True"


class TestCloudCompressorAnthropic:
    @pytest.fixture
    def compressor(self, anthropic_config):
        return CloudCompressor(anthropic_config)

    @pytest.mark.asyncio
    async def test_compress_success_tool_use(self, compressor):
        body = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "store_compression_result",
                    "input": {
                        "narrative_summary": NARRATIVE,
                        "extracted_entities": ENTITIES,
                    },
                }
            ],
            "usage": {"input_tokens": 90, "output_tokens": 45},
        }
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_mock_response(body))
            mock_gc.return_value = mock_client

            result = await compressor.compress("ASSISTANT: queried db")

        assert result["narrative_summary"] == NARRATIVE
        assert result["extracted_entities"]["target_table"] == "sales_q3"
        assert result["total_tokens"] == 135  # 90 + 45

    @pytest.mark.asyncio
    async def test_compress_payload_includes_tools(self, compressor):
        """Verify that the actual HTTP payload posted to Anthropic contains tools."""
        body = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "store_compression_result",
                    "input": {"narrative_summary": "x", "extracted_entities": {}},
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_mock_response(body))
            mock_gc.return_value = mock_client

            await compressor.compress("text")

        _, kwargs = mock_client.post.call_args
        posted_payload = kwargs.get("json") or mock_client.post.call_args[0][1]
        assert "tools" in posted_payload


class TestCloudCompressorGemini:
    @pytest.fixture
    def compressor(self, gemini_config):
        return CloudCompressor(gemini_config)

    @pytest.mark.asyncio
    async def test_compress_success(self, compressor):
        body = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": CLEAN_JSON}],
                    }
                }
            ],
            "usageMetadata": {"totalTokenCount": 180},
        }
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_mock_response(body))
            mock_gc.return_value = mock_client

            result = await compressor.compress("TOOL: result data")

        assert result["narrative_summary"] == NARRATIVE
        assert result["extracted_entities"]["conn_id"] == "conn_994a82"
        assert result["total_tokens"] == 180

    @pytest.mark.asyncio
    async def test_compress_payload_json_mime(self, compressor):
        """Verify generationConfig is present in the posted payload."""
        body = {
            "candidates": [
                {"content": {"parts": [{"text": CLEAN_JSON}]}}
            ],
            "usageMetadata": {"totalTokenCount": 10},
        }
        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_mock_response(body))
            mock_gc.return_value = mock_client

            await compressor.compress("text")

        call_kwargs = mock_client.post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("generationConfig", {}).get("responseMimeType") == "application/json"


# ===========================================================================
# 8.  CloudCompressor — tenacity 429 retry
# ===========================================================================

class TestRateLimitRetry:
    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self, openai_config):
        """First call returns 429, second call succeeds."""
        compressor = CloudCompressor(openai_config)

        rate_limit_resp = MagicMock(spec=httpx.Response)
        rate_limit_resp.status_code = 429
        rate_limit_resp.text = "rate limited"
        rate_limit_error = httpx.HTTPStatusError(
            message="429",
            request=MagicMock(),
            response=rate_limit_resp,
        )
        rate_limit_resp.raise_for_status.side_effect = rate_limit_error

        success_body = {
            "choices": [{"message": {"content": CLEAN_JSON}}],
            "usage": {"total_tokens": 80},
        }
        success_resp = _mock_response(success_body)

        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return rate_limit_resp
            return success_resp

        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=_side_effect)
            mock_gc.return_value = mock_client

            # Patch tenacity wait to be instant in tests
            with patch(
                "tenacity.wait_exponential",
                return_value=lambda retry_state: 0,
            ):
                result = await compressor.compress("text")

        assert call_count == 2
        assert result["narrative_summary"] == NARRATIVE

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self, openai_config):
        """All 5 attempts return 429; should raise HTTPStatusError after exhaustion."""
        compressor = CloudCompressor(openai_config)

        rate_limit_resp = MagicMock(spec=httpx.Response)
        rate_limit_resp.status_code = 429
        rate_limit_resp.text = "rate limited"
        rate_limit_error = httpx.HTTPStatusError(
            message="429",
            request=MagicMock(),
            response=rate_limit_resp,
        )
        rate_limit_resp.raise_for_status.side_effect = rate_limit_error

        with patch.object(compressor, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=rate_limit_resp)
            mock_gc.return_value = mock_client

            with patch(
                "tenacity.wait_exponential",
                return_value=lambda retry_state: 0,
            ):
                with pytest.raises(httpx.HTTPStatusError):
                    await compressor.compress("text")


# ===========================================================================
# 9.  CloudConfig validation
# ===========================================================================

class TestCloudConfig:
    def test_requires_provider(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CloudConfig(model="gpt-4o-mini", api_key="sk-x")  # type: ignore[call-arg]

    def test_api_key_is_secret(self, openai_config):
        # SecretStr should not expose the value via str()
        assert "sk-openai-test" not in str(openai_config.api_key)
        assert openai_config.api_key.get_secret_value() == "sk-openai-test"

    def test_base_url_optional(self, openai_config):
        assert openai_config.base_url is None

    def test_base_url_accepted(self):
        cfg = CloudConfig(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            api_key="sk-x",
            base_url="https://oai.helicone.ai",
        )
        assert cfg.base_url == "https://oai.helicone.ai"
