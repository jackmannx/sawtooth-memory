"""Built-in event handlers for logging and journaling."""

import logging
from typing import Awaitable, Callable

from sawtooth_memory.journal import AsyncCompressionJournal

from .types import CompressionCycleCompleteEvent

logger = logging.getLogger(__name__)


def make_journal_handler(
    journal: AsyncCompressionJournal,
) -> Callable[[CompressionCycleCompleteEvent], Awaitable[None]]:
    """Factory to create a journal handler bound to a specific agent's journal instance."""

    async def journal_handler(event: CompressionCycleCompleteEvent) -> None:
        await journal.write(
            cycle_id=event.cycle_id or event.event_id,
            l1_tokens_evicted=event.l1_tokens_evicted,
            l1_5_entities_retained=event.l1_5_entities_retained,
            l2_summary_generated=event.l2_summary_generated,
            timestamp=event.timestamp,
            extra={
                "messages_compressed": event.messages_compressed,
                "total_duration_ms": event.total_duration_ms,
                "final_l1_tokens": event.final_l1_tokens,
            },
        )

    return journal_handler


async def console_logger(event):
    """Log all events to console (for debugging)."""
    logger.info(f"[EVENT] {event.event_type}: {event}")
