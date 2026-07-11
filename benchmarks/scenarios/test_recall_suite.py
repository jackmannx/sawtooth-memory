"""Recall accuracy benchmarks across the needle suite."""

from __future__ import annotations

import pytest

from benchmarks.common.config import MockedContextManager, recall_benchmark_config
from benchmarks.common.fixtures import (
    NEEDLE_SUITE,
    all_needle_values,
    generate_conversation,
)


async def _recall_score(
    turns: int,
    *,
    message_size: str = "medium",
    late_needle_turn: int | None = None,
) -> dict[str, object]:
    config = recall_benchmark_config()
    needles = all_needle_values(late_needle_turn=late_needle_turn)
    conversation = generate_conversation(
        turns=turns,
        message_size=message_size,
        late_needle_turn=late_needle_turn,
    )

    session = MockedContextManager(
        "You are a physics expert.",
        config,
        mock_delay_ms=0.0,
        enable_events=False,
    )
    async with session as cm:
        for msg in conversation:
            await cm.add_message(msg["role"], msg["content"])  # type: ignore[arg-type]

        final_prompt = await cm.build_prompt()
        prompt_string = "\n".join(m["content"] for m in final_prompt)
        ledger_string = cm.state.l1_5_entities.to_json_str()
        archive_string = cm.state.l2_archival.narrative
        searchable = f"{prompt_string}\n{ledger_string}\n{archive_string}"

    retained = [needle for needle in needles if needle in searchable]
    missing = [needle for needle in needles if needle not in searchable]

    return {
        "turns": turns,
        "message_size": message_size,
        "needles_total": len(needles),
        "needles_retained": len(retained),
        "needles_missing": missing,
        "recall_pct": (len(retained) / len(needles)) * 100.0 if needles else 100.0,
    }


@pytest.mark.integration_benchmark
@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [10, 25])
async def test_standard_needle_recall(turns: int) -> None:
    results = await _recall_score(turns, message_size="small")
    assert results["recall_pct"] == 100.0
    assert results["needles_missing"] == []


@pytest.mark.integration_benchmark
@pytest.mark.asyncio
async def test_late_needle_recall() -> None:
    results = await _recall_score(30, message_size="small", late_needle_turn=25)
    assert "txn_late_needle_zz99" not in results["needles_missing"]
    assert results["recall_pct"] == 100.0


@pytest.mark.integration_benchmark
@pytest.mark.asyncio
async def test_entity_ledger_covers_all_needle_categories() -> None:
    """Every needle category in the suite should be represented."""
    results = await _recall_score(10, message_size="small")
    assert results["needles_total"] == len(NEEDLE_SUITE)
    assert results["recall_pct"] == 100.0
