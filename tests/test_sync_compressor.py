"""Tests for synchronous compression backends."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from sawtooth_memory.config import OllamaConfig
from sawtooth_memory.exceptions import OllamaConnectionError
from sawtooth_memory.sync_compressor import SyncOllamaCompressor


def test_sync_ollama_compress_parses_json():
    config = OllamaConfig(base_url="http://localhost:11434", model="test-model")
    compressor = SyncOllamaCompressor(config)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {
            "content": (
                '{"narrative_summary": "Done.", "extracted_entities": {"id": "abc"}}'
            )
        }
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.post.return_value = mock_response

    with patch.object(compressor, "_get_client", return_value=mock_client):
        result = compressor.compress("USER: hello")

    assert result["narrative_summary"] == "Done."
    assert result["extracted_entities"] == {"id": "abc"}
    compressor.close()


def test_sync_ollama_compress_connect_error():
    config = OllamaConfig(base_url="http://localhost:11434", model="test-model")
    compressor = SyncOllamaCompressor(config)

    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.post.side_effect = httpx.ConnectError("connection refused")

    with patch.object(compressor, "_get_client", return_value=mock_client):
        with pytest.raises(OllamaConnectionError):
            compressor.compress("USER: hello")

    compressor.close()


def test_sync_ollama_prunes_base64():
    config = OllamaConfig()
    compressor = SyncOllamaCompressor(config)

    captured_payload = {}

    def capture_post(url, json=None, **kwargs):
        captured_payload["json"] = json
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": '{"narrative_summary": "ok", "extracted_entities": {}}'
            }
        }
        mock_response.raise_for_status = MagicMock()
        return mock_response

    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.post = capture_post

    blob = "A" * 100
    with patch.object(compressor, "_get_client", return_value=mock_client):
        compressor.compress(f"USER: {blob}")

    user_content = captured_payload["json"]["messages"][1]["content"]
    assert "[BASE64_REMOVED]" in user_content
    compressor.close()
