"""Tests for SyncContextManager — sync-native API without AnyIO portal."""

from unittest.mock import MagicMock, patch

import pytest

from sawtooth_memory import ContextManagerConfig, SyncContextManager


@pytest.fixture
def sync_config():
    return ContextManagerConfig.for_sync_script(soft_limit_tokens=1000)


def test_sync_manager_out_of_context(sync_config):
    manager = SyncContextManager(system_prompt="Test", config=sync_config)

    with pytest.raises(RuntimeError, match="must be used within a 'with' context"):
        manager.add_message("user", "Hello")

    with pytest.raises(RuntimeError, match="must be used within a 'with' context"):
        manager.build_prompt()


def test_sync_manager_does_not_start_anyio_portal(sync_config):
    with patch("anyio.from_thread.start_blocking_portal") as mock_portal:
        with patch(
            "sawtooth_memory.sync_manager.SyncOllamaCompressor.ping",
            return_value=None,
        ):
            with SyncContextManager("Test Agent", config=sync_config) as memory:
                memory.add_message("user", "Hello world!")
                stats = memory.get_stats()
                assert stats["l1_message_count"] == 1
                assert stats["compression"]["mode"] == "inline_sync"

        mock_portal.assert_not_called()


def test_sync_manager_core_pipeline(sync_config):
    with patch(
        "sawtooth_memory.sync_manager.SyncOllamaCompressor.ping",
        return_value=None,
    ):
        with SyncContextManager(
            system_prompt="You are a helpful assistant.",
            config=sync_config,
        ) as memory:
            memory.add_message("user", "Hello world!")
            memory.add_message("assistant", "Greetings! How can I help?")

            stats = memory.get_stats()
            assert stats["l1_message_count"] == 2
            assert stats["l0_tokens"] > 0

            prompt = memory.build_prompt()
            assert len(prompt) == 3
            assert prompt[0]["role"] == "system"
            assert "You are a helpful assistant." in prompt[0]["content"]
            assert prompt[1]["content"] == "Hello world!"


def test_sync_manager_pin_entity(sync_config):
    with patch(
        "sawtooth_memory.sync_manager.SyncOllamaCompressor.ping",
        return_value=None,
    ):
        with SyncContextManager("Test", config=sync_config) as memory:
            memory.pin_entity("tracking_code", "ALPHA-991")
            assert memory.state.l1_5_entities.entities["tracking_code"][-1] == "ALPHA-991"


def test_sync_manager_health_check(sync_config):
    with patch(
        "sawtooth_memory.sync_manager.SyncOllamaCompressor.ping",
        return_value=None,
    ):
        with SyncContextManager("Test Agent", config=sync_config) as memory:
            health = memory.health_check()
            assert health["status"] == "healthy"
            assert health["checks"]["configuration"] == "OK"
            assert health["checks"]["runtime"] == "sync_inline"
            assert health["checks"]["compression_mode"] == "blocking"


def test_sync_manager_inline_compression_on_soft_limit():
    config = ContextManagerConfig.for_sync_script(
        soft_limit_tokens=10,
        hard_limit_tokens=500,
        chunk_size=1,
        compression_mode="always_llm",
    )

    mock_compressor = MagicMock()
    mock_compressor.compress.return_value = {
        "narrative_summary": "Compressed turn.",
        "extracted_entities": {},
    }
    mock_compressor.ping.return_value = None
    mock_compressor.close.return_value = None

    with patch(
        "sawtooth_memory.sync_manager.SyncOllamaCompressor",
        return_value=mock_compressor,
    ):
        with SyncContextManager("Sys.", config=config) as memory:
            memory.add_message("user", "This is a long enough message to trigger compression.")
            assert memory.get_stats()["compression"]["cycles"] >= 1
            assert "Compressed turn." in memory.state.l2_archival.narrative

            memory.add_message(
                "user", "A second long message must trigger another compression cycle."
            )
            assert memory.get_stats()["compression"]["cycles"] >= 2


def test_for_sync_script_factory():
    config = ContextManagerConfig.for_sync_script(soft_limit_tokens=1500)
    assert config.soft_limit_tokens == 1500
    assert config.enable_l3_semantic_storage is False
