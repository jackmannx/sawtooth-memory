"""tests/test_middleware.py — Integration tests for ContextManager."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from sawtooth_memory.config import ContextManagerConfig
from sawtooth_memory.middleware import ContextManager


@pytest.fixture
def no_fallback_config():
    return ContextManagerConfig(
        soft_limit_tokens=30,
        hard_limit_tokens=60,
        chunk_size=2,
        fallback_truncate=False,
    )


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager_protocol(self, config):
        async with ContextManager("You are helpful.", config) as cm:
            assert cm._worker._running is True
        assert cm._worker._running is False

    @pytest.mark.asyncio
    async def test_manual_start_stop(self, config):
        cm = ContextManager("You are helpful.", config)
        await cm.start()
        assert cm._worker._running
        await cm.stop()
        assert not cm._worker._running

    @pytest.mark.asyncio
    async def test_journal_path_from_config(self, tmp_path):
        journal_file = tmp_path / "custom_journal.jsonl"
        config = ContextManagerConfig(journal_path=str(journal_file))
        cm = ContextManager("Sys.", config)
        assert cm._journal_path == journal_file
        assert cm._journal is not None
        assert cm._journal.path == journal_file

    @pytest.mark.asyncio
    async def test_journal_path_kwarg_overrides_config(self, tmp_path):
        config_path = tmp_path / "config_journal.jsonl"
        kwarg_path = tmp_path / "kwarg_journal.jsonl"
        config = ContextManagerConfig(journal_path=str(config_path))
        cm = ContextManager("Sys.", config, journal_path=kwarg_path)
        assert cm._journal_path == kwarg_path
        assert cm._journal.path == kwarg_path


class TestAddMessage:
    @pytest.mark.asyncio
    async def test_message_added_to_l1(self, config):
        async with ContextManager("Sys prompt.", config) as cm:
            await cm.add_message("user", "Hello")
            assert len(cm.state.l1_working.messages) == 1
            assert cm.state.l1_working.messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_token_count_increments(self, config):
        async with ContextManager("Sys prompt.", config) as cm:
            await cm.add_message("user", "Hello world")
            assert cm.state.l1_working.token_count > 0

    @pytest.mark.asyncio
    async def test_soft_limit_triggers_compression(self, config):
        with patch(
            "sawtooth_memory.middleware.CompressionWorker.enqueue"
        ) as mock_enqueue:
            async with ContextManager("Sys.", config) as cm:
                for i in range(30):
                    await cm.add_message(
                        "user", f"Message {i} with enough words to consume tokens"
                    )
                assert mock_enqueue.called

    @pytest.mark.asyncio
    async def test_messages_sliced_on_compression(self, config):
        with patch("sawtooth_memory.middleware.CompressionWorker.enqueue"):
            async with ContextManager("Sys.", config) as cm:
                for i in range(20):
                    await cm.add_message("user", f"Message number {i} goes here")

                assert len(cm.state.l1_working.messages) < 20


class TestBuildPrompt:
    @pytest.mark.asyncio
    async def test_system_block_always_present(self, config):
        async with ContextManager("You are a robot.", config) as cm:
            prompt = await cm.build_prompt()
            assert prompt[0]["role"] == "system"
            assert "[SYSTEM_L0]" in prompt[0]["content"]
            assert "You are a robot." in prompt[0]["content"]

    @pytest.mark.asyncio
    async def test_working_memory_follows_system(self, config):
        async with ContextManager("Sys.", config) as cm:
            await cm.add_message("user", "Hi")
            await cm.add_message("assistant", "Hello!")
            prompt = await cm.build_prompt()
            assert prompt[1]["role"] == "user"
            assert prompt[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_archive_injected_when_present(self, config):
        async with ContextManager("Sys.", config) as cm:
            cm.state.l2_archival.narrative = "Agent did a thing."
            prompt = await cm.build_prompt()
            assert "[ARCHIVE_L2]" in prompt[0]["content"]
            assert "Agent did a thing." in prompt[0]["content"]

    @pytest.mark.asyncio
    async def test_entity_ledger_injected_when_present(self, config):
        async with ContextManager("Sys.", config) as cm:
            cm.state.l1_5_entities.entities["conn_id"] = ["abc-123"]
            prompt = await cm.build_prompt()
            assert "[ENTITY_LEDGER_L1_5]" in prompt[0]["content"]
            assert "conn_id" in prompt[0]["content"]

    @pytest.mark.asyncio
    async def test_empty_archive_not_injected(self, config):
        async with ContextManager("Sys.", config) as cm:
            prompt = await cm.build_prompt()
            assert "[ARCHIVE_L2]" not in prompt[0]["content"]

    @pytest.mark.asyncio
    async def test_empty_entities_not_injected(self, config):
        async with ContextManager("Sys.", config) as cm:
            prompt = await cm.build_prompt()
            assert "[ENTITY_LEDGER_L1_5]" not in prompt[0]["content"]

    @pytest.mark.asyncio
    async def test_tier_ordering(self, config):
        async with ContextManager("Sys.", config) as cm:
            cm.state.l2_archival.narrative = "L2 Narrative."
            cm.state.l1_5_entities.entities["conn_id"] = ["abc-123"]
            
            # Mock L3 retrieval
            cm._config.enable_l3_prompt_retrieval = True
            cm._last_l3_retrieval = []
            
            async def mock_retrieve(query):
                return "1. L3 Chunk.", [{"text": "L3 Chunk.", "similarity": 1.0, "origin": "L3 Semantic Retrieval"}]
            cm._retrieve_l3_chunks = mock_retrieve
            cm._l3_indexer = True # Just to pass the truthy check
            
            prompt = await cm.build_prompt(retrieval_query="test")
            content = prompt[0]["content"]
            
            l2_idx = content.index("[ARCHIVE_L2]")
            l3_idx = content.index("[ARCHIVE_L3]")
            l1_5_idx = content.index("[ENTITY_LEDGER_L1_5]")
            
            assert l2_idx < l3_idx < l1_5_idx


class TestGetStats:
    @pytest.mark.asyncio
    async def test_stats_keys_present(self, config):
        async with ContextManager("Sys.", config) as cm:
            stats = cm.get_stats()
            assert "l0_tokens" in stats
            assert "l1_tokens" in stats
            assert "l1_message_count" in stats
            assert "l1_5_entity_count" in stats
            assert "l2_tokens" in stats
            assert "worker" in stats

    @pytest.mark.asyncio
    async def test_stats_l0_tokens_nonzero(self, config):
        async with ContextManager("You are an agent.", config) as cm:
            assert cm.get_stats()["l0_tokens"] > 0


class TestRepr:
    @pytest.mark.asyncio
    async def test_repr_contains_key_info(self, config):
        async with ContextManager("Sys.", config) as cm:
            r = repr(cm)
            assert "ContextManager" in r
            assert "l1=" in r


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_passes_valid_config(self, config):
        with patch(
            "sawtooth_memory.middleware.OllamaCompressor.ping",
            new_callable=AsyncMock,
        ):
            async with ContextManager("Sys.", config) as cm:
                report = await cm.health_check()
                assert report["status"] == "healthy"
                assert report["checks"]["configuration"] == "OK"
                assert report["checks"]["worker_status"] == "RUNNING"
                assert report["checks"]["backend"] == "ollama"

    @pytest.mark.asyncio
    async def test_health_check_reports_degraded_when_ollama_unreachable(self, config):
        with patch(
            "sawtooth_memory.middleware.OllamaCompressor.ping",
            new_callable=AsyncMock,
            side_effect=ConnectionError("refused"),
        ):
            async with ContextManager("Sys.", config) as cm:
                report = await cm.health_check()
                assert report["status"] == "degraded"
                assert "UNREACHABLE" in report["checks"]["backend_reachable"]

    @pytest.mark.asyncio
    async def test_health_check_raises_on_invalid_limits(self):
        # Setup: Soft limit is dangerously higher than Hard limit
        bad_config = ContextManagerConfig(
            soft_limit_tokens=500, hard_limit_tokens=200, chunk_size=3
        )
        cm = ContextManager("Sys.", bad_config)

        with pytest.raises(ValueError, match="strictly less than hard_limit_tokens"):
            await cm.health_check()


class TestHardTruncate:
    @pytest.mark.asyncio
    async def test_force_truncate_releases_compression_debounce_lock(self, config):
        """Hard truncation must release the debounce lock left by a queued cycle."""
        async with ContextManager("Sys.", config, enable_events=False) as cm:
            for _ in range(config.chunk_size):
                await cm.add_message("user", "seed message with enough words for tokens")

            cm._monitor._is_compression_queued = True
            assert cm._monitor._is_compression_queued is True

            await cm._force_truncate()
            assert cm._monitor._is_compression_queued is False

    @pytest.mark.asyncio
    async def test_hard_limit_allows_fresh_compression_after_slow_worker(self):
        """After hard truncation, new messages can queue another compression cycle."""
        slow_compressor = AsyncMock()

        async def slow_compress(_text: str) -> dict:
            await asyncio.sleep(5)
            return {"narrative_summary": "done", "extracted_entities": {}}

        slow_compressor.compress = slow_compress
        slow_compressor.close = AsyncMock()

        config = ContextManagerConfig(
            soft_limit_tokens=20,
            hard_limit_tokens=40,
            chunk_size=2,
            fallback_truncate=True,
            enable_deterministic_ner=False,
            compression_mode="always_llm",
        )

        with patch(
            "sawtooth_memory.middleware.OllamaCompressor",
            return_value=slow_compressor,
        ), patch(
            "sawtooth_memory.middleware.CompressionWorker.enqueue"
        ) as mock_enqueue:
            cm = ContextManager("Sys.", config, enable_events=False)
            await cm.start()
            try:
                for i in range(8):
                    await cm.add_message(
                        "user", f"Message {i} with enough words to use tokens"
                    )
                initial_enqueue_count = mock_enqueue.call_count
                assert initial_enqueue_count >= 1

                for i in range(8, 12):
                    await cm.add_message(
                        "user", f"Hard push message {i} with many words here"
                    )

                for i in range(12, 20):
                    await cm.add_message(
                        "user", f"Recovery message {i} with enough words again"
                    )

                assert mock_enqueue.call_count > initial_enqueue_count
            finally:
                await cm.stop()
