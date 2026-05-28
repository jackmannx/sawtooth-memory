"""
monitor.py — Token counting and threshold detection.

Uses tiktoken to count tokens locally (no API call required) before
deciding whether to trigger background compression.
"""

from __future__ import annotations

import logging

import tiktoken

from .state import MemoryState, Message

logger = logging.getLogger(__name__)

_MESSAGE_OVERHEAD = 4


class TokenMonitor:
    """
    Counts tokens using a local tiktoken encoder and detects when
    Working Memory (L1) has crossed the soft compression threshold.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        soft_limit: int = 3000,
        hard_limit: int = 6000,
    ) -> None:
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

        self._enc = None
        try:
            self._enc = tiktoken.encoding_for_model(model)
            logger.debug(f"TokenMonitor: using tiktoken encoding for model '{model}'")
        except KeyError:
            logger.warning(
                f"TokenMonitor: model '{model}' not found in tiktoken, "
                "attempting cl100k_base fallback."
            )
            try:
                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
        except Exception as exc:
            logger.warning(
                f"TokenMonitor: tiktoken encoding unavailable ({exc}). "
                "Falling back to word-count approximation (~1.3 words/token)."
            )

    # ------------------------------------------------------------------
    # Core counting
    # ------------------------------------------------------------------

    def count_text(self, text: str) -> int:
        """Return the token count of a raw string."""
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, int(len(text.split()) * 1.3)) if text.strip() else 0

    def count_message(self, message: Message) -> int:
        """Return the token count of a Message including role overhead."""
        return self.count_text(message.content) + _MESSAGE_OVERHEAD

    # ------------------------------------------------------------------
    # Threshold checks
    # ------------------------------------------------------------------

    def exceeds_soft_limit(self, state: MemoryState) -> bool:
        """True when L1 Working Memory has passed the soft compression trigger."""
        return state.l1_working.token_count >= self.soft_limit

    def exceeds_hard_limit(self, state: MemoryState) -> bool:
        """True when L1 Working Memory has passed the hard safety cap."""
        return state.l1_working.token_count >= self.hard_limit

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def recount_working_memory(self, state: MemoryState) -> None:
        """
        Recompute token counts for all messages in L1 and update the total.
        Call this after any bulk mutation to ensure counts stay accurate.
        """
        total = 0
        for msg in state.l1_working.messages:
            msg.token_count = self.count_message(msg)
            total += msg.token_count
        state.l1_working.token_count = total
        logger.debug(f"TokenMonitor: recounted L1 → {total} tokens")
