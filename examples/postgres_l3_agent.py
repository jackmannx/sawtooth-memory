"""
postgres_l3_agent.py — Durable Postgres storage with L3 semantic retrieval.

Prerequisites:
  1. PostgreSQL with pgvector:  CREATE EXTENSION IF NOT EXISTS vector;
  2. pip install "sawtooth-memory[postgres]"
  3. Export SAWTOOTH_PG_DSN=postgresql://user:pass@localhost:5432/sawtooth

Uses the local hash embedding backend so you can try L3 without OpenAI keys.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

from sawtooth_memory import (
    ContextManager,
    ContextManagerConfig,
    PostgresStorageAdapter,
)


async def main() -> None:
    dsn = os.environ.get("SAWTOOTH_PG_DSN")
    if not dsn:
        raise SystemExit(
            "Set SAWTOOTH_PG_DSN to a Postgres DSN with the pgvector extension."
        )

    postgres = PostgresStorageAdapter(dsn=dsn, embedding_dimension=64)
    config = ContextManagerConfig(
        storage_adapter=postgres,
        session_id="demo_l3_session",
        soft_limit_tokens=80,
        hard_limit_tokens=300,
        chunk_size=2,
        compression_mode="always_llm",
        enable_l3_semantic_storage=True,
        enable_l3_prompt_retrieval=True,
        embedding_backend="hash",
        embedding_dimension=64,
        l3_retrieval_top_k=3,
        l3_retrieval_max_tokens=400,
        enable_deterministic_ner=True,
        enable_salience_extractor=True,
        enable_ingest_entity_scan=True,
        fallback_truncate=True,
    )

    fake = AsyncMock()
    fake.compress = AsyncMock(
        return_value={
            "narrative_summary": (
                "User disputed charge on card ending 4412 related to "
                "order ORD-7781 and reference REF-9900."
            ),
            "extracted_entities": {
                "order_id": "ORD-7781",
                "reference": "REF-9900",
            },
        }
    )
    fake.close = AsyncMock()
    fake.ping = AsyncMock()

    with patch("sawtooth_memory.middleware.OllamaCompressor", return_value=fake):
        async with ContextManager(
            "You are a billing dispute agent.", config=config
        ) as cm:
            await cm.add_message(
                "user",
                "I dispute order ORD-7781. Reference REF-9900. Card ending 4412.",
            )
            await cm.add_message(
                "assistant",
                "Logged ORD-7781 / REF-9900. Investigating the charge now.",
            )
            await cm.add_message("user", "Also check related shipment SHIP-221.")
            await cm.add_message("assistant", "Checking SHIP-221.")
            await asyncio.sleep(0.4)

            matches = await cm.search_semantic_archive("order dispute REF-9900", top_k=3)
            prompt = await cm.build_prompt(retrieval_query="order dispute REF-9900")
            count = await cm.l3_chunk_count()
            trace = cm.explain_prompt()

            print(f"L3 chunks indexed: {count}")
            print(f"Explicit search hits: {len(matches)}")
            for hit in matches:
                print(f"  sim={hit.similarity:.3f} text={hit.text[:100]!r}")

            system = prompt[0]["content"]
            print("\n[ARCHIVE_L3] in prompt:", "[ARCHIVE_L3]" in system)
            print("Explain L3 block keys:", list(trace.get("l3_semantic", {}).keys()))

        await postgres.close()


if __name__ == "__main__":
    asyncio.run(main())
