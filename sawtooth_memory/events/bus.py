"""Simple async event bus with optional handler batching."""

import asyncio
import logging
from typing import Awaitable, Callable, Dict, List, Optional, Set, TypeVar

from .types import SawtoothEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[SawtoothEvent], Awaitable[None]]

T = TypeVar("T")


class EventBus:
    """
    Minimal async event bus protected against Python 3.11+ Garbage Collection.
    """

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []

        # GC Shield: Strong references for fire-and-forget background tasks
        self._background_tasks: Set[asyncio.Task] = set()

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove handler for an event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def emit(self, event: SawtoothEvent, fire_and_forget: bool = True) -> None:
        """
        Emit an event from an async context.
        """
        handlers = self._global_handlers.copy()
        if event.event_type in self._handlers:
            handlers.extend(self._handlers[event.event_type])

        if not handlers:
            return

        if fire_and_forget:
            for handler in handlers:
                task = asyncio.create_task(self._safe_call(handler, event))
                # Protect the task from garbage collection
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
        else:
            tasks = [self._safe_call(handler, event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)

    def emit_nowait(self, event: SawtoothEvent) -> None:
        """
        Schedule an emit from synchronous code, protected from Python 3.11 GC.
        """
        task = asyncio.create_task(self.emit(event, fire_and_forget=True))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def drain(self) -> None:
        """Wait for all in-flight background tasks to complete."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    async def _safe_call(self, handler: EventHandler, event: SawtoothEvent) -> None:
        """Call handler and log errors without crashing."""
        try:
            await handler(event)
        except Exception as e:
            logger.exception(f"Error in event handler {handler.__name__}: {e}")


# Global bus singleton (lightweight)
_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_event_bus() -> None:
    """Reset the global bus (useful for testing)."""
    global _bus
    _bus = None
