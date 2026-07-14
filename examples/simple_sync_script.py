"""
simple_sync_script.py — Minimal linear script using SyncContextManager.

No asyncio, no background threads, no AnyIO portal required.

Requires a running Ollama instance with the configured model (default: phi4-mini).
"""

from sawtooth_memory import ContextManagerConfig, SyncContextManager


def main() -> None:
    config = ContextManagerConfig.for_sync_script(soft_limit_tokens=1500)

    with SyncContextManager("You are a helpful assistant.", config=config) as memory:
        memory.add_message("user", "Ticket INC-4421 needs escalation.")
        memory.add_message("assistant", "I'll look up ticket INC-4421 now.")

        prompt = memory.build_prompt()
        stats = memory.get_stats()

        print("=== Compiled prompt ===")
        for msg in prompt:
            print(f"[{msg['role']}] {msg['content'][:200]}...")
            print()

        print("=== Stats ===")
        print(f"L1 messages: {stats['l1_message_count']}")
        print(f"L1.5 entities: {stats['l1_5_entity_count']}")
        print(f"Compression mode: {stats['compression']['mode']}")


if __name__ == "__main__":
    main()
