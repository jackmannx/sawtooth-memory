"""
monitor.py — Token counting and threshold detection.

Uses tiktoken to count tokens locally (no API call required) before
deciding whether to trigger background compression.

Phase 2 enhancement: Optionally emits SoftLimitReachedEvent and HardLimitReachedEvent
when the respective thresholds are crossed for the first time. Adds Debouncing and
Turn-based Batching to prevent background queue flooding.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import tiktoken

from .events.bus import EventBus
from .events.types import (
    HardLimitReachedEvent,
    SawtoothEvent,
    SoftLimitReachedEvent,
)
from .state import MemoryState, Message

logger = logging.getLogger(__name__)

_MESSAGE_OVERHEAD = 4


class TokenMonitor:
    """
    Counts tokens using a local tiktoken encoder and detects when
    Working Memory (L1) has crossed the soft compression threshold.

    Includes Batching & Debouncing logic to ensure compression is only
    triggered when necessary and never floods the background worker.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        soft_limit: int = 3000,
        hard_limit: int = 6000,
        max_unsummarized_turns: int | None = None,  # NEW: Batching config
        event_bus: EventBus | None = None,
    ) -> None:
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.max_unsummarized_turns = max_unsummarized_turns
        self.event_bus = event_bus

        # Internal flags to avoid repeated events while already over the limit
        self._soft_exceeded = False
        self._hard_exceeded = False

        # NEW: Debounce lock to prevent queuing multiple compression tasks
        self._is_compression_queued = False

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

        # NEW: Subscribe to event bus to reset debounce locks automatically
        if self.event_bus is not None:
            self.event_bus.subscribe(
                "compression.cycle_complete", self._on_compression_done
            )  # type: ignore[arg-type]
            self.event_bus.subscribe(
                "compression.cycle_failed", self._on_compression_done
            )  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Batching & Debouncing Logic (NEW)
    # ------------------------------------------------------------------

    async def _on_compression_done(self, event: SawtoothEvent) -> None:
        """Reset the debounce lock and soft limit flag after compression completes."""
        self._is_compression_queued = False
        self._soft_exceeded = (
            False  # Reset so the telemetry event can fire again next cycle
        )
        logger.debug("TokenMonitor: Compression finished. Lock released.")

    def should_trigger_compression(self, state: MemoryState) -> bool:
        """
        Master evaluation function for middleware.
        Checks both token limits and turn limits, respecting the debounce lock.
        """
        # If the worker is already busy compressing, don't spam the queue
        if self._is_compression_queued:
            return False

        needs_compression = False

        # Rule 1: Check Token Limit (This inherently fires the SoftLimitReachedEvent if crossed)
        if self.exceeds_soft_limit(state):
            needs_compression = True

        # Rule 2: Check Turn Limit (Batching)
        if self.max_unsummarized_turns is not None:
            if len(state.l1_working.messages) >= self.max_unsummarized_turns:
                needs_compression = True

        # If either condition is met, lock the debouncer and return True
        if needs_compression:
            self._is_compression_queued = True
            return True

        return False

    # ------------------------------------------------------------------
    # Core counting (UNCHANGED)
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
    # Threshold checks (with optional event emission) (UNCHANGED)
    # ------------------------------------------------------------------

    def exceeds_soft_limit(self, state: MemoryState) -> bool:
        """
        True when L1 Working Memory has passed the soft compression trigger.
        If event bus is available and this is the first time crossing,
        emits SoftLimitReachedEvent (fire-and-forget).
        """
        exceeded = state.l1_working.token_count >= self.soft_limit
        if exceeded and not self._soft_exceeded and self.event_bus is not None:
            self._emit_soft_limit_reached(state.l1_working.token_count)
        self._soft_exceeded = exceeded
        return exceeded

    def exceeds_hard_limit(self, state: MemoryState) -> bool:
        """
        True when L1 Working Memory has passed the hard safety cap.
        If event bus is available and this is the first time crossing,
        emits HardLimitReachedEvent (fire-and-forget).
        """
        exceeded = state.l1_working.token_count >= self.hard_limit
        if exceeded and not self._hard_exceeded and self.event_bus is not None:
            self._emit_hard_limit_reached(state.l1_working.token_count)
        self._hard_exceeded = exceeded
        return exceeded

    # ------------------------------------------------------------------
    # Event emission helpers (UNCHANGED)
    # ------------------------------------------------------------------

    def _emit_soft_limit_reached(self, current_tokens: int) -> None:
        """Fire-and-forget emission of SoftLimitReachedEvent."""
        if self.event_bus is None:
            return
        event = SoftLimitReachedEvent(
            current_tokens=current_tokens,
            soft_limit=self.soft_limit,
            hard_limit=self.hard_limit,
        )
        asyncio.create_task(self.event_bus.emit(event))
        logger.info(f"Soft limit reached: {current_tokens}/{self.soft_limit} tokens")

    def _emit_hard_limit_reached(self, current_tokens: int) -> None:
        """Fire-and-forget emission of HardLimitReachedEvent."""
        if self.event_bus is None:
            return
        event = HardLimitReachedEvent(
            current_tokens=current_tokens,
            soft_limit=self.soft_limit,
            hard_limit=self.hard_limit,
        )
        asyncio.create_task(self.event_bus.emit(event))
        logger.warning(f"Hard limit reached: {current_tokens}/{self.hard_limit} tokens")

    # ------------------------------------------------------------------
    # State helpers (UNCHANGED)
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
