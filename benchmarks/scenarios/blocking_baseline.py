"""Blocking baseline emulation for comparative latency benchmarks."""

from __future__ import annotations

import json
import time
import urllib.request


class BlockingSummaryMemory:
    """
    Zero-dependency emulation of sequential summary memory.

    Mirrors LangChain ConversationSummaryMemory: each save_context call
    blocks the main thread until the compressor returns.
    """

    def __init__(self, model: str, base_url: str, *, simulate_ms: float = 0.0) -> None:
        self.model = model
        self.base_url = base_url
        self.simulate_ms = simulate_ms
        self.buffer = ""

    def save_context(self, user_msg: str, ai_msg: str) -> float:
        """Block until summarization completes. Returns elapsed seconds."""
        start = time.perf_counter()

        if self.simulate_ms > 0:
            time.sleep(self.simulate_ms / 1000.0)
            self.buffer = f"Summary through turn including: {user_msg[:80]}..."
            return time.perf_counter() - start

        prompt = (
            "Progressively summarize the lines of conversation provided, adding to "
            "the current summary and returning a new summary.\n\n"
            f"Current Summary:\n{self.buffer}\n\n"
            f"New Lines of Conversation:\nHuman: {user_msg}\nAI: {ai_msg}\n\n"
            "New Summary:"
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
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                self.buffer = res_body.get("response", "").strip()
        except Exception as exc:
            self.buffer = f"[Summarization failed: {exc}]"

        return time.perf_counter() - start

    def final_text(self) -> str:
        return self.buffer
