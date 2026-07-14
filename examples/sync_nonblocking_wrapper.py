"""
sync_nonblocking_wrapper.py — SawtoothSyncWrapper for sync hosts that need
non-blocking background compression (Flask/Django/CLI with worker semantics).

Unlike SyncContextManager (inline blocking folds), this drives the full async
ContextManager on an AnyIO portal thread.

Requires a reachable Ollama (or set background_model / cloud config).
"""

from sawtooth_memory import (
    CompressionCycleCompleteEvent,
    ContextManagerConfig,
    SawtoothSyncWrapper,
    get_event_bus,
)


def main() -> None:
    config = ContextManagerConfig(
        soft_limit_tokens=800,
        hard_limit_tokens=2000,
        enable_deterministic_ner=True,
        enable_salience_extractor=True,
        enable_ingest_entity_scan=True,
    )

    async def on_cycle(event: CompressionCycleCompleteEvent) -> None:
        print(
            f"[event] compression complete: "
            f"{event.messages_compressed} msgs, "
            f"{event.l1_tokens_evicted} L1 tokens evicted"
        )

    bus = get_event_bus()
    bus.subscribe("compression.cycle_complete", on_cycle)

    with SawtoothSyncWrapper(
        "You are a support agent.",
        config=config,
        enable_events=True,
    ) as memory:
        memory.add_message("user", "Escalate ticket INC-4421 to on-call.")
        memory.pin_entity("oncall_queue", "SRE-P1")
        memory.add_message("assistant", "Pinned SRE-P1 and noted INC-4421.")

        prompt = memory.build_prompt()
        stats = memory.get_stats()

        print("=== Compiled prompt preview ===")
        for msg in prompt:
            preview = msg["content"][:180] + ("..." if len(msg["content"]) > 180 else "")
            print(f"[{msg['role']}] {preview}")

        print("\n=== State snapshot ===")
        print(f"L1 messages: {len(memory.state.l1_working.messages)}")
        print(f"L1.5 entities: {memory.state.l1_5_entities.entities}")
        print(f"Worker queue depth: {stats['worker']['queue_depth']}")
        print(f"Health: {memory.health_check()['status']}")


if __name__ == "__main__":
    main()
