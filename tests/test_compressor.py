"""tests/test_compressor.py — Unit tests for OllamaCompressor."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sawtooth_memory.compressor import OllamaCompressor, _prune
from sawtooth_memory.exceptions import CompressionError, OllamaConnectionError


@pytest.fixture
def compressor(ollama_config):
    return OllamaCompressor(ollama_config)


class TestPrune:
    def test_strips_base64(self):
        base64_blob = "A" * 100
        result = _prune(f"Some text {base64_blob} more text")
        assert "[BASE64_REMOVED]" in result
        assert "A" * 100 not in result

    def test_strips_stacktrace(self):
        trace = (
            "Traceback (most recent call last):\n"
            "  File 'foo.py', line 1\n"
            "    raise ValueError\n"
            "ValueError: oops\n\n"
        )
        result = _prune(trace)
        assert "[STACKTRACE_REMOVED]" in result

    def test_preserves_normal_text(self):
        text = "The agent found a 14% revenue drop in Q3."
        result = _prune(text)
        assert result == text

    def test_short_strings_not_stripped(self):
        text = "abc123XYZ"
        result = _prune(text)
        assert result == text


class TestParseOutput:
    def test_valid_json(self, compressor):
        raw = json.dumps({
            "narrative_summary": "Agent connected to DB.",
            "extracted_entities": {"conn_id": "abc123"},
        })
        result = compressor._parse_output(raw)
        assert result["narrative_summary"] == "Agent connected to DB."
        assert result["extracted_entities"]["conn_id"] == "abc123"

    def test_json_with_markdown_fence(self, compressor):
        raw = "```json\n{\"narrative_summary\": \"done\", \"extracted_entities\": {}}\n```"
        result = compressor._parse_output(raw)
        assert result["narrative_summary"] == "done"

    def test_missing_entities_defaults_empty(self, compressor):
        raw = json.dumps({"narrative_summary": "summary only"})
        result = compressor._parse_output(raw)
        assert result["extracted_entities"] == {}

    def test_malformed_json_returns_raw_as_narrative(self, compressor):
        raw = "This is not JSON at all."
        result = compressor._parse_output(raw)
        assert result["narrative_summary"] == raw
        assert result["extracted_entities"] == {}

    def test_entities_cast_to_str(self, compressor):
        raw = json.dumps({
            "narrative_summary": "x",
            "extracted_entities": {"count": 42, "flag": True},
        })
        result = compressor._parse_output(raw)
        assert result["extracted_entities"]["count"] == "42"
        assert result["extracted_entities"]["flag"] == "True"


class TestCompress:
    @pytest.mark.asyncio
    async def test_compress_success(self, compressor):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "narrative_summary": "Agent ran a query.",
                    "extracted_entities": {"table": "sales_q3"},
                })
            }
        }

        with patch.object(compressor, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await compressor.compress("USER: run a query\nASSISTANT: done")

        assert result["narrative_summary"] == "Agent ran a query."
        assert result["extracted_entities"]["table"] == "sales_q3"

    @pytest.mark.asyncio
    async def test_compress_connection_error(self, compressor):
        import httpx

        with patch.object(compressor, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("connection refused")
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(OllamaConnectionError):
                await compressor.compress("some text")
