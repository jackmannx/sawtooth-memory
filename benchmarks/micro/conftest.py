"""Microbenchmark fixtures."""

from __future__ import annotations

import pytest

from benchmarks.common.config import benchmark_config
from sawtooth_memory.monitor import TokenMonitor
from sawtooth_memory.ner import NERPipeline
from sawtooth_memory.state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    Message,
    SystemPrompt,
    WorkingMemory,
)


@pytest.fixture
def token_monitor() -> TokenMonitor:
    return TokenMonitor(model="gpt-4o", soft_limit=250, hard_limit=600)


@pytest.fixture
def ner_pipeline() -> NERPipeline:
    return NERPipeline.from_config(
        enable=True,
        custom_patterns={
            "aws_arn": r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]+:\d{12}:[a-zA-Z0-9\-\_/]+",
        },
    )


@pytest.fixture
def entity_ledger() -> EntityLedger:
    return EntityLedger()


@pytest.fixture
def populated_state() -> MemoryState:
    state = MemoryState(
        l0_system=SystemPrompt(content="You are a physics expert.", token_count=5),
        l1_working=WorkingMemory(),
        l1_5_entities=EntityLedger(),
        l2_archival=ArchivalMemory(),
    )
    state.l2_archival.narrative = "User discussed quantum mechanics across many turns."
    state.l1_5_entities.upsert(
        {
            "transaction_id": "txn_998877_alpha_omega",
            "config_path": "/etc/nginx/sites-enabled/api.conf",
            "runbook_uri": "https://internal.corp/runbooks/inc-4421",
        }
    )
    for i in range(20):
        msg = Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i} " * 20)
        msg.token_count = 30
        state.l1_working.append(msg)
    state.l1_working.token_count = sum(m.token_count for m in state.l1_working.messages)
    return state


@pytest.fixture
def bench_config():
    return benchmark_config(enable_events=False)
