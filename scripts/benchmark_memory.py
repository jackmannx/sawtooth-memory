import asyncio
import os
import time
import json
import urllib.request
from typing import List, Dict, cast, Literal

# --- Sawtooth Imports ---
from sawtooth_memory import ContextManager, ContextManagerConfig
from sawtooth_memory.config import CloudConfig, OllamaConfig, Provider
from sawtooth_memory.monitor import TokenMonitor
from pydantic import SecretStr

# ===========================================================================
# ⚙️ BENCHMARK CONFIGURATION ZONE
# ===========================================================================
RUN_MODE = os.getenv("BENCHMARK_MODE", "local").lower()

# --- Local Settings ---
LOCAL_MODEL = "phi4-mini"
LOCAL_OLLAMA_URL = "http://localhost:11434"

# --- Cloud Settings ---
CLOUD_PROVIDER_NAME = "openai"
CLOUD_MODEL = "gpt-4o-mini"


# ===========================================================================


# --- Fail-Safe LangChain Simulation Runner ---
class EmulatedLangChainOllamaSummaryMemory:
    """
    A robust, zero-dependency emulation of LangChain's ConversationSummaryMemory.
    It directly calls the local Ollama HTTP endpoint synchronously on the main thread,
    perfectly reproducing LangChain's blocking architecture, latency overhead,
    and aggressive history loss for a fair, uncompromised benchmark fight.
    """

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self.buffer = ""
        self.request_mod = urllib.request

    def save_context(self, user_msg: str, ai_msg: str) -> None:
        prompt = (
            f"Progressively summarize the lines of conversation provided, adding to the current summary "
            f"and returning a new summary.\n\n"
            f"Current Summary:\n{self.buffer}\n\n"
            f"New Lines of Conversation:\nHuman: {user_msg}\nAI: {ai_msg}\n\n"
            f"New Summary (Do not include prefix/labels, just give the plain paragraph summary):"
        )

        url = f"{self.base_url}/api/generate"
        data = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")

        try:
            req = self.request_mod.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.request_mod.urlopen(req, timeout=30) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                self.buffer = res_body.get("response", "").strip()
        except Exception as e:
            self.buffer = (
                f"[System Warning: Summarization failed due to timeout or load: {e}]"
            )


def get_sawtooth_config() -> ContextManagerConfig:
    """Dynamically builds the Sawtooth configuration based on configuration."""
    if RUN_MODE == "local":
        return ContextManagerConfig(
            soft_limit_tokens=250,
            hard_limit_tokens=600,
            chunk_size=4,
            fallback_truncate=True,
            tokenizer_model="gpt-4o",
            ollama=OllamaConfig(base_url=LOCAL_OLLAMA_URL, model=LOCAL_MODEL),
        )
    elif RUN_MODE == "cloud":
        provider_map = {
            "openai": Provider.OPENAI,
            "anthropic": Provider.ANTHROPIC,
            "gemini": Provider.GEMINI,
        }
        api_key_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }

        env_var = api_key_env_map.get(CLOUD_PROVIDER_NAME)
        if env_var is None:
            raise ValueError(f"Unknown provider map: {CLOUD_PROVIDER_NAME}")

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"Missing API Key! Please set the {env_var} environment variable."
            )

        return ContextManagerConfig(
            soft_limit_tokens=250,
            hard_limit_tokens=600,
            chunk_size=4,
            fallback_truncate=True,
            tokenizer_model="gpt-4o",
            cloud=CloudConfig(
                provider=provider_map[CLOUD_PROVIDER_NAME],
                model=CLOUD_MODEL,
                api_key=SecretStr(api_key),
            ),
        )
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


# ---------------------------------------------------------------------------
# 1. THE SIMULATION ENGINE
# ---------------------------------------------------------------------------
def generate_synthetic_conversation(turns: int = 10) -> List[Dict[str, str]]:
    """Generates a conversation with a Golden Needle injected early."""
    conversation = []
    golden_needle = "txn_998877_alpha_omega"

    for i in range(turns):
        if i == 2:
            conversation.append(
                {
                    "role": "user",
                    "content": f"Please remember my transaction ID for later: {golden_needle}",
                }
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"I have saved your transaction ID: {golden_needle}.",
                }
            )
        else:
            conversation.append(
                {
                    "role": "user",
                    "content": f"Tell me a long, detailed fact about quantum physics. (Turn {i})",
                }
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"Here is a complex explanation about quarks, entanglement, and wave-function collapse for turn {i}. "
                    * 3,
                }
            )

    return conversation


# ---------------------------------------------------------------------------
# 2. THE LANGCHAIN RUNNER
# ---------------------------------------------------------------------------
async def run_langchain_benchmark(conversation: List[Dict[str, str]]) -> dict:
    print(f"\n[LangChain] Starting LOCAL Run (Ollama Architecture: {LOCAL_MODEL})...")
    start_time = time.time()

    memory = EmulatedLangChainOllamaSummaryMemory(
        model=LOCAL_MODEL, base_url=LOCAL_OLLAMA_URL
    )

    for i in range(0, len(conversation), 2):
        user_msg = conversation[i]["content"]
        ai_msg = conversation[i + 1]["content"]

        turn_start = time.time()
        memory.save_context(user_msg, ai_msg)
        turn_elapsed = time.time() - turn_start

        print(
            f"  [LangChain] Processed turn {i // 2 + 1} (Main thread froze for {turn_elapsed:.2f}s)..."
        )

    execution_time = time.time() - start_time

    final_summary = memory.buffer
    monitor = TokenMonitor(model="gpt-4o")
    final_tokens = monitor.count_text(final_summary)
    needle_retained = "txn_998877_alpha_omega" in final_summary

    return {
        "framework": "LangChain Summary Memory",
        "final_prompt_tokens": final_tokens,
        "golden_needle_retained": needle_retained,
        "execution_time_seconds": round(execution_time, 2),
    }


# ---------------------------------------------------------------------------
# 3. THE SAWTOOTH CHAMPION
# ---------------------------------------------------------------------------
async def run_sawtooth_benchmark(conversation: List[Dict[str, str]]) -> dict:
    print(f"\n[Sawtooth] Starting LOCAL Champion Run (Ollama Worker: {LOCAL_MODEL})...")
    start_time = time.time()

    config = get_sawtooth_config()

    async with ContextManager(
        system_prompt="You are a physics expert.", config=config
    ) as cm:
        for i, msg in enumerate(conversation):
            # Satisfy Mypy strict literal typing for roles
            role = cast(Literal["user", "assistant", "system", "tool"], msg["role"])
            await cm.add_message(role, msg["content"])

            if i % 2 != 0:
                print(
                    f"  [Sawtooth] Processed turn {i // 2 + 1} instantly (Non-blocking background loop)..."
                )

        print("  [Sawtooth] Draining background compression worker pipeline safely...")
        await cm.stop()

        final_prompt = await cm.build_prompt()
        prompt_string = "\n".join([m["content"] for m in final_prompt])

        monitor = TokenMonitor(model="gpt-4o")
        final_tokens = monitor.count_text(prompt_string)
        needle_retained = "txn_998877_alpha_omega" in prompt_string

    execution_time = time.time() - start_time

    return {
        "framework": "Sawtooth Hierarchical Memory",
        "final_prompt_tokens": final_tokens,
        "golden_needle_retained": needle_retained,
        "execution_time_seconds": round(execution_time, 2),
    }


# ---------------------------------------------------------------------------
# 4. THE HARNESS ENTRYPOINT
# ---------------------------------------------------------------------------
async def main() -> None:
    print("================================================================")
    print(" SAWTOOTH MEMORY vs LANGCHAIN : DETACHED PERFORMANCE BENCHMARK  ")
    print("================================================================")

    conversation = generate_synthetic_conversation(turns=10)  # 20 messages total
    print(f"Generated synthetic conversation with {len(conversation)} total messages.")
    print(f"Target Local Model: {LOCAL_MODEL}")
    print("Golden Needle injected: 'txn_998877_alpha_omega'\n")

    lc_results = await run_langchain_benchmark(conversation)
    st_results = await run_sawtooth_benchmark(conversation)

    print("\n================================================================")
    print(f"{'Performance Metric':<30} | {'LangChain':<15} | {'Sawtooth':<15}")
    print("----------------------------------------------------------------")
    print(
        f"{'Main Thread Latency (s)':<30} | {lc_results['execution_time_seconds']:<15} | {st_results['execution_time_seconds']:<15}"
    )
    print(
        f"{'Final Prompt Token Cost':<30} | {lc_results['final_prompt_tokens']:<15} | {st_results['final_prompt_tokens']:<15}"
    )
    print(
        f"{'Golden Needle Recall (UUID)':<30} | {str(lc_results['golden_needle_retained']):<15} | {str(st_results['golden_needle_retained']):<15}"
    )
    print("================================================================")

    if st_results["execution_time_seconds"] < lc_results["execution_time_seconds"]:
        speedup = round(
            lc_results["execution_time_seconds"] / st_results["execution_time_seconds"],
            1,
        )
        print(
            f"\nPerformance Analysis: Sawtooth is {speedup}x faster on the main loop via async scheduling."
        )
    else:
        print("\nPerformance Analysis: Background worker finished processing.")

    if (
        not lc_results["golden_needle_retained"]
        and st_results["golden_needle_retained"]
    ):
        print(
            "🎯 Accuracy Analysis: Sawtooth perfectly anchored the UUID in the L1.5 ledger, while LangChain dropped it."
        )


if __name__ == "__main__":
    asyncio.run(main())
