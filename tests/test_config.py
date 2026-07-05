"""tests/test_config.py — ContextManagerConfig validation and auto-routing."""

from __future__ import annotations

import pytest

from sawtooth_memory.config import (
    CloudConfig,
    ContextManagerConfig,
    OllamaConfig,
    Provider,
    infer_cloud_provider,
    resolve_cloud_api_key,
)


class TestInferCloudProvider:
    def test_openai_models(self):
        assert infer_cloud_provider("gpt-4o-mini") == Provider.OPENAI
        assert infer_cloud_provider("o1-preview") == Provider.OPENAI

    def test_anthropic_models(self):
        assert infer_cloud_provider("claude-3-5-haiku-latest") == Provider.ANTHROPIC

    def test_gemini_models(self):
        assert infer_cloud_provider("gemini-1.5-flash") == Provider.GEMINI

    def test_local_models_return_none(self):
        assert infer_cloud_provider("phi4-mini") is None
        assert infer_cloud_provider("llama3") is None


class TestBackgroundModelAutoRouting:
    def test_local_model_routes_to_ollama(self):
        config = ContextManagerConfig(background_model="phi4-mini")
        assert config.ollama is not None
        assert config.ollama.model == "phi4-mini"
        assert config.cloud is None

    def test_cloud_model_routes_with_env_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = ContextManagerConfig(background_model="gpt-4o-mini")
        assert config.cloud is not None
        assert config.cloud.provider == Provider.OPENAI
        assert config.cloud.model == "gpt-4o-mini"
        assert config.cloud.api_key.get_secret_value() == "sk-test-key"
        assert config.ollama is None

    def test_cloud_model_without_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            ContextManagerConfig(background_model="gpt-4o-mini")

    def test_explicit_cloud_config_overrides_model_name(self):
        config = ContextManagerConfig(
            background_model="gpt-4o-mini",
            cloud=CloudConfig(
                provider=Provider.OPENAI,
                model="gpt-4o",
                api_key="sk-explicit",
            ),
        )
        assert config.cloud.model == "gpt-4o-mini"

    def test_explicit_ollama_config_overrides_model_name(self):
        config = ContextManagerConfig(
            background_model="llama3",
            ollama=OllamaConfig(model="phi4-mini"),
        )
        assert config.ollama.model == "llama3"


class TestResolveCloudApiKey:
    def test_reads_provider_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert resolve_cloud_api_key(Provider.ANTHROPIC) == "sk-ant-test"

    def test_gemini_falls_back_to_secondary_env_var(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
        assert resolve_cloud_api_key(Provider.GEMINI) == "gem-test"


class TestL3ConfigValidation:
    def test_l3_requires_semantic_storage_when_enabled(self):
        from tests.l3_helpers import InMemorySemanticStorage

        with pytest.raises(ValueError, match="requires storage_adapter"):
            ContextManagerConfig(enable_l3_semantic_storage=True)

        with pytest.raises(ValueError, match="SemanticStorageAdapter"):
            ContextManagerConfig(
                enable_l3_semantic_storage=True,
                storage_adapter=object(),
            )

        storage = InMemorySemanticStorage(embedding_dimension=128)
        with pytest.raises(ValueError, match="embedding_dimension"):
            ContextManagerConfig(
                enable_l3_semantic_storage=True,
                storage_adapter=storage,
                embedding_dimension=64,
            )
