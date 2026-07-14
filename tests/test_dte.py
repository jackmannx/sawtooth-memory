"""Dual-Target Externalization behavior and cost-control tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sawtooth_memory import ContextManagerConfig, SyncContextManager
from sawtooth_memory.intent_planner import plan_prompt
from sawtooth_memory.middleware import ContextManager
from sawtooth_memory.state import EntityLedger


@pytest.mark.asyncio
async def test_dte_soft_limit_folds_without_compressor_call():
    config = ContextManagerConfig(
        soft_limit_tokens=10,
        hard_limit_tokens=500,
        chunk_size=1,
        compression_mode="dte",
        enable_deterministic_ner=True,
    )
    compressor = MagicMock()
    compressor.compress = AsyncMock()
    compressor.close = AsyncMock()

    with patch(
        "sawtooth_memory.middleware.OllamaCompressor", return_value=compressor
    ):
        async with ContextManager("Sys.", config, enable_events=False) as memory:
            await memory.add_message(
                "user", "Use incident INC-4421 and tracking code ALPHA-991 now."
            )

            compressor.compress.assert_not_awaited()
            assert "[FOLD " in memory.state.l2_archival.narrative
            assert memory.state.dte.narrative_debt_tokens > 0
            values = [
                value
                for history in memory.state.l1_5_entities.entities.values()
                for value in history
            ]
            assert "INC-4421" in values


@pytest.mark.asyncio
async def test_tool_observation_is_crushed_and_reversible():
    config = ContextManagerConfig(
        soft_limit_tokens=10000,
        hard_limit_tokens=20000,
        obs_crush_min_tokens=20,
    )
    content = json.dumps([{"id": index, "status": "ok"} for index in range(100)])

    async with ContextManager("Sys.", config, enable_events=False) as memory:
        await memory.add_message("tool", content)
        stored = memory.state.l1_working.messages[0].content
        cache_id = stored.split("id=", 1)[1].split(" ", 1)[0]

        assert stored.startswith("[OBSERVATION_CRUSHED")
        assert memory.retrieve_observation(cache_id) == content
        assert memory.get_stats()["dte"]["observation_tokens_saved"] > 0


def test_intent_planner_omits_l2_when_ledger_covers_entity_query():
    ledger = EntityLedger()
    ledger.upsert({"incident_id": "INC-4421"})
    config = ContextManagerConfig()

    plan = plan_prompt("What is the incident_id for INC-4421?", ledger, config)

    assert plan.intent == "entity_lookup"
    assert plan.ledger_covers_query is True
    assert plan.include_l2 is False
    assert plan.l3_top_k <= 2


def test_sync_dte_consolidates_on_pull_with_spend_budget():
    config = ContextManagerConfig.for_sync_script(
        soft_limit_tokens=10,
        hard_limit_tokens=500,
        chunk_size=1,
        compression_mode="dte",
        narrative_debt_trigger_tokens=1,
        background_spend_ratio=1.0,
        enable_novelty_filter=False,
        compression_guideline="Summarize.",
        enable_sync_consolidation=True,
    )
    compressor = MagicMock()
    compressor.compress.return_value = {
        "narrative_summary": "The user selected the safe migration path.",
        "extracted_entities": {},
    }
    compressor.close.return_value = None

    with patch(
        "sawtooth_memory.sync_manager.SyncOllamaCompressor",
        return_value=compressor,
    ):
        with SyncContextManager("Sys.", config) as memory:
            memory.add_message(
                "user", "We selected the safe migration path because rollback matters."
            )
            assert compressor.compress.call_count == 0

            memory.build_prompt(retrieval_query="Recap what happened")
            memory.build_prompt(retrieval_query="Recap what happened")

            assert compressor.compress.call_count == 1
            assert "[FOLD " not in memory.state.l2_archival.narrative
            assert "safe migration path" in memory.state.l2_archival.narrative
            assert memory.state.dte.narrative_debt_tokens == 0
