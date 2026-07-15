"""
Admin Event Bus
===============
Tiny in-process pub/sub used by the admin dashboard's SSE stream.

Design constraints (all deliberate):
- **Never blocks and never raises into the call path.** ``publish`` is a
  synchronous, non-blocking fan-out; when a subscriber's queue is full the
  oldest event is dropped so a slow dashboard can neither block a call nor
  grow memory without bound.
- **In-process only.** No persistence, no Redis — if nobody is subscribed,
  publishing is a no-op.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Per-subscriber buffer. Small on purpose: the dashboard is a live view, not
# an event log — a consumer that falls this far behind loses the oldest
# events rather than stalling the publisher.
DEFAULT_QUEUE_SIZE = 256


class EventBus:
    """Bounded fan-out pub/sub for admin/dashboard events."""

    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE):
        self._maxsize = maxsize
        self._subscribers: List[asyncio.Queue] = []

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber and return its (bounded) event queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue. Unknown queues are ignored."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def publish(self, event: str, call_id: str,
                data: Optional[Dict[str, Any]] = None) -> None:
        """Fan an event out to every subscriber. Non-blocking, never raises.

        Adds a wall-clock ``ts``. ``call_id`` should be the session's
        transcript id, or ``"-"`` when the event happens outside a call.
        A full subscriber queue drops its OLDEST event to make room (the
        dashboard is a live view; stalling the publisher is never acceptable).
        """
        if not self._subscribers:
            return
        item = {
            "event": event,
            "call_id": call_id or "-",
            "ts": time.time(),
            "data": dict(data) if data else {},
        }
        for q in list(self._subscribers):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()  # drop-oldest
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(item)
                except Exception:
                    pass
            except Exception as e:  # defensive: publish must never raise
                logger.debug(f"EventBus publish error: {e}")
