"""Dual-Target Externalization behavior and cost-control tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sawtooth_memory import ContextManagerConfig, SyncContextManager
from sawtooth_memory.dte_runtime import account_main_prompt_tokens, prompt_turn_key
from sawtooth_memory.intent_planner import plan_prompt
from sawtooth_memory.middleware import ContextManager
from sawtooth_memory.novelty import residualize
from sawtooth_memory.observation_crush import crush_observation
from sawtooth_memory.state import (
    EntityLedger,
    MemoryState,
    Message,
    SystemPrompt,
    WorkingMemory,
)


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
        with SyncContextManager(
            "You are a careful migration analyst that preserves causal decisions.",
            config,
        ) as memory:
            memory.add_message(
                "user", "We selected the safe migration path because rollback matters."
            )
            assert compressor.compress.call_count == 0

            # Second turn grows turn-scoped prompt spend enough to cover consolidation.
            memory.add_message(
                "assistant",
                "Understood. I will keep the rollback rationale in working memory.",
            )
            memory.build_prompt(retrieval_query="Recap what happened")
            memory.build_prompt(retrieval_query="Recap what happened")

            assert compressor.compress.call_count == 1
            assert "[FOLD " not in memory.state.l2_archival.narrative
            assert "safe migration path" in memory.state.l2_archival.narrative
            assert memory.state.dte.narrative_debt_tokens == 0


def test_prompt_token_accounting_is_turn_scoped():
    state = MemoryState(
        l0_system=SystemPrompt(content="sys", token_count=1),
        l1_working=WorkingMemory(),
    )
    msg = Message(role="user", content="hello")
    msg.token_count = 2
    state.l1_working.append(msg)
    dte = state.dte

    key1 = prompt_turn_key(state)
    key = account_main_prompt_tokens(dte, 100, previous_key=None, current_key=key1)
    key = account_main_prompt_tokens(dte, 100, previous_key=key, current_key=key1)
    assert dte.main_prompt_tokens == 100

    msg2 = Message(role="assistant", content="world")
    msg2.token_count = 2
    state.l1_working.append(msg2)
    key2 = prompt_turn_key(state)
    account_main_prompt_tokens(dte, 50, previous_key=key, current_key=key2)
    assert dte.main_prompt_tokens == 150


def test_novelty_strips_ledger_values_in_one_pass():
    ledger = EntityLedger()
    ledger.upsert({"incident_id": "INC-4421", "code": "ALPHA-991"})
    source = "[FOLD n=1] Resolved INC-4421 with ALPHA-991 and a remaining note."

    result = residualize(
        source,
        ledger,
        existing_narrative="Resolved earlier.",
        count_text=lambda text: max(1, len(text.split())),
    )

    assert "INC-4421" not in result.residual
    assert "ALPHA-991" not in result.residual
    assert "remaining note" in result.residual.casefold()


def test_observation_crush_skips_hash_when_compact_not_smaller():
    content = json.dumps({"ok": True, "n": 1})
    counts: list[int] = []

    def count_text(text: str) -> int:
        counts.append(len(text))
        return max(1, len(text) // 4)

    result = crush_observation(content, count_text=count_text, min_tokens=1)
    # Tiny JSON sample is not smaller than original → passthrough, one tokenize.
    assert result.cache_id is None
    assert result.content == content
    assert len(counts) == 1


@pytest.mark.asyncio
async def test_repeated_build_prompt_does_not_inflate_spend():
    config = ContextManagerConfig(
        soft_limit_tokens=10000,
        hard_limit_tokens=20000,
        compression_mode="dte",
    )
    async with ContextManager("Sys.", config, enable_events=False) as memory:
        await memory.add_message("user", "hello there")
        await memory.build_prompt()
        first = memory.state.dte.main_prompt_tokens
        await memory.build_prompt()
        second = memory.state.dte.main_prompt_tokens
        assert first > 0
        assert second == first


@pytest.mark.asyncio
async def test_dte_hard_truncate_creates_fold_stub():
    config = ContextManagerConfig(
        soft_limit_tokens=5,
        hard_limit_tokens=8,
        chunk_size=1,
        compression_mode="dte",
        fallback_truncate=True,
        max_unsummarized_turns=100,
    )
    compressor = MagicMock()
    compressor.compress = AsyncMock()
    compressor.close = AsyncMock()

    with patch(
        "sawtooth_memory.middleware.OllamaCompressor", return_value=compressor
    ):
        async with ContextManager("Sys.", config, enable_events=False) as memory:
            # Soft limit will fold first messages; then fill past hard limit.
            await memory.add_message("user", "alpha " * 20)
            await memory.add_message("user", "beta " * 20)
            await memory.add_message("user", "gamma " * 40)
            assert "[FOLD " in memory.state.l2_archival.narrative
            compressor.compress.assert_not_awaited()


def test_working_memory_slice_oldest_decrements_tokens():
    wm = WorkingMemory()
    for text in ("one", "two words", "three more words"):
        msg = Message(role="user", content=text)
        msg.token_count = len(text.split())
        wm.append(msg)
    total = wm.token_count
    chunk = wm.slice_oldest(2)
    assert len(chunk) == 2
    assert wm.token_count == total - sum(m.token_count for m in chunk)
