"""
cloud_compressor.py — Explicit CloudConfig for background/sync compression.

Uses the public Provider + CloudConfig surface instead of background_model
auto-routing. Set OPENAI_API_KEY (or pass api_key=) before running.
"""

import asyncio
import os

from sawtooth_memory import (
    CloudConfig,
    ContextManager,
    ContextManagerConfig,
    Provider,
)


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    config = ContextManagerConfig(
        cloud=CloudConfig(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            api_key=api_key,
        ),
        soft_limit_tokens=600,
        hard_limit_tokens=1500,
        enable_deterministic_ner=True,
        enable_salience_extractor=True,
        enable_ingest_entity_scan=True,
        compression_mode="dte",
    )

    async with ContextManager(
        "You are a precise operations assistant.", config=config
    ) as cm:
        await cm.add_message(
            "user",
            "Transaction txn_998877_alpha failed at gateway gw-east-3.",
        )
        await cm.add_message(
            "assistant",
            "Captured txn_998877_alpha on gw-east-3.",
        )
        await cm.pin_entity("primary_txn", "txn_998877_alpha")

        prompt = await cm.build_prompt()
        health = await cm.health_check()
        trace = cm.explain_prompt()

        print("Health:", health["status"], health["checks"].get("compressor"))
        print("Entities in explain trace:", len(trace.get("l1_5_entities", [])))
        print("Compiled messages:", len(prompt))
        for msg in prompt:
            preview = msg["content"][:160] + ("..." if len(msg["content"]) > 160 else "")
            print(f"  [{msg['role']}] {preview}")


if __name__ == "__main__":
    asyncio.run(main())
