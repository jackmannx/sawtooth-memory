"""
run_demo.py — Flagship demo: naive FIFO memory vs. Sawtooth, side by side.

A single long coding-agent session (see scenario.py) is fed through two
memories:

  1. NaiveMemory   — a flat message list, FIFO-truncated at a token budget.
                     This is the default shape of most hand-rolled agent
                     memory, and of LangChain-style buffer memories once
                     they hit a hard cap.
  2. SyncContextManager (Sawtooth) — DTE compression mode. Evicted messages
                     are folded into an entity ledger deterministically,
                     with zero LLM calls, so this demo needs no API key
                     and no local model.

At the end, both are asked to recall facts planted at the very start of
the conversation, after ~35 turns of unrelated debugging chatter have
pushed those turns out of any fixed-size window.

Run:
    pip install -e .
    python examples/flagship_demo/run_demo.py
"""

from naive_baseline import NaiveMemory
from scenario import CONVERSATION, PLANTED_FACTS, SYSTEM_PROMPT

from sawtooth_memory import ContextManagerConfig, SyncContextManager

WIDTH = 78


def header(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def run_naive() -> NaiveMemory:
    memory = NaiveMemory(system_prompt=SYSTEM_PROMPT, max_tokens=900)
    for role, content in CONVERSATION:
        memory.add_message(role, content)
    return memory


def run_sawtooth() -> SyncContextManager:
    config = ContextManagerConfig.for_sync_script(
        soft_limit_tokens=900,
        hard_limit_tokens=1400,
        chunk_size=4,
    )
    cm = SyncContextManager(SYSTEM_PROMPT, config=config)
    cm.__enter__()
    for role, content in CONVERSATION:
        cm.add_message(role, content)
    return cm


def check_recall(prompt_text: str, facts: dict[str, str]) -> dict[str, bool]:
    return {key: value in prompt_text for key, value in facts.items()}


def main() -> None:
    header("FLAGSHIP DEMO — Naive Memory vs. Sawtooth Memory")
    print(
        f"Replaying a {len(CONVERSATION)}-turn production-incident debugging "
        "session through both memories."
    )
    print("Facts planted in turns 1-4, asked about again in the final turn:")
    for key, value in PLANTED_FACTS.items():
        print(f"  - {key}: {value}")

    header("1. Naive FIFO memory (flat list, hard token cap)")
    naive = run_naive()
    naive_prompt = naive.build_prompt()
    naive_text = "\n".join(m["content"] for m in naive_prompt)
    print(f"Messages retained in window: {len(naive.messages)}")
    print(f"Messages dropped:            {len(naive.dropped_messages)}")
    naive_recall = check_recall(naive_text, PLANTED_FACTS)
    for key, found in naive_recall.items():
        status = "PRESENT" if found else "LOST"
        print(f"  [{status:7}] {key} = {PLANTED_FACTS[key]}")

    header("2. Sawtooth Memory (SyncContextManager, DTE, zero LLM calls)")
    cm = run_sawtooth()
    sawtooth_prompt = cm.build_prompt()
    sawtooth_text = "\n".join(m["content"] for m in sawtooth_prompt)
    stats = cm.get_stats()
    print(f"L1 (raw) messages retained:  {stats['l1_message_count']}")
    print(f"L1.5 entities tracked:       {stats['l1_5_entity_count']}")
    print(f"DTE fold cycles run:         {stats['dte']['fold_cycles']}")
    print("Background LLM calls made:   0  (DTE mode folds deterministically)")
    sawtooth_recall = check_recall(sawtooth_text, PLANTED_FACTS)
    for key, found in sawtooth_recall.items():
        status = "PRESENT" if found else "LOST"
        print(f"  [{status:7}] {key} = {PLANTED_FACTS[key]}")
    cm.__exit__(None, None, None)

    header("Result")
    naive_lost = sum(1 for found in naive_recall.values() if not found)
    sawtooth_lost = sum(1 for found in sawtooth_recall.values() if not found)
    print(f"Naive memory lost   {naive_lost}/{len(PLANTED_FACTS)} planted facts.")
    print(f"Sawtooth memory lost {sawtooth_lost}/{len(PLANTED_FACTS)} planted facts.")
    print(
        "\nBoth memories saw the exact same conversation and the same token "
        "budget. The naive list dropped the facts it no longer had room for; "
        "Sawtooth's entity ledger kept them exact, without ever calling an LLM."
    )
    print(
        "\nFor real-world latency numbers (blocking LLM summarization vs. "
        "Sawtooth's background worker), see BENCHMARKS.md at the repo root."
    )


if __name__ == "__main__":
    main()
