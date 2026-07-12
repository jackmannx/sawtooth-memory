import asyncio

from sawtooth_memory import ContextManager, ContextManagerConfig
from sawtooth_memory.config import OllamaConfig


async def main():
    # Requires a running Ollama instance with the configured model pulled.
    config = ContextManagerConfig(
        ollama=OllamaConfig(model="phi4-mini"),
        soft_limit_tokens=500,
        hard_limit_tokens=1500,
        fallback_truncate=True,
        enable_deterministic_ner=True,
        enable_salience_extractor=True,
        enable_ingest_entity_scan=True,
    )

    system_prompt = "You are a highly capable AI assistant."

    async with ContextManager(system_prompt=system_prompt, config=config) as cm:
        print("--- Ingesting Messages ---")
        await cm.add_message("user", "Hello! My database connection ID is db_prod_994.")
        await cm.add_message("assistant", "I have noted your connection ID.")
        await cm.add_message("user", "Actually, switch that to db_staging_112.")
        await cm.add_message("user", "Also escalate ticket INC-4421 to on-call.")

        # Entity Guard: regex + salience extract IDs at ingest and compression time.
        # pin_entity() is available for explicit pinning of critical values.
        # build_prompt() stitches L0, L2, L1.5, and L1 into an OpenAI-style list.
        prompt = await cm.build_prompt()

        print("\n--- Compiled Prompt ---")
        for msg in prompt:
            role = msg["role"].upper()
            content = msg["content"]
            preview = content[:200] + ("..." if len(content) > 200 else "")
            print(f"[{role}]: {preview}")

        print("\n--- Current Memory State ---")
        print(f"L1 messages: {len(cm.state.l1_working.messages)}")
        print(f"L1.5 entities: {cm.state.l1_5_entities.entities}")
        print(f"L2 narrative length: {len(cm.state.l2_archival.narrative)} chars")


if __name__ == "__main__":
    asyncio.run(main())
