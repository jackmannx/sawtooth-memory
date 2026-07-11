"""EntityLedger microbenchmarks."""

from __future__ import annotations

import pytest

from sawtooth_memory.state import EntityLedger


@pytest.mark.benchmark(group="entity_ledger")
class TestEntityLedgerBenchmark:
    def test_upsert_single(self, benchmark, entity_ledger: EntityLedger) -> None:
        benchmark(
            entity_ledger.upsert,
            {"transaction_id": "txn_998877_alpha_omega"},
        )

    def test_upsert_batch(self, benchmark, entity_ledger: EntityLedger) -> None:
        entities = {f"key_{i}": f"value_{i}" for i in range(50)}
        benchmark(entity_ledger.upsert, entities)

    def test_to_json_str(self, benchmark, entity_ledger: EntityLedger) -> None:
        for i in range(100):
            entity_ledger.upsert({f"entity_{i}": f"val_{i}"})
            if i % 10 == 0:
                entity_ledger.upsert({f"entity_{i}": f"val_{i}_updated"})

        benchmark(entity_ledger.to_json_str)
