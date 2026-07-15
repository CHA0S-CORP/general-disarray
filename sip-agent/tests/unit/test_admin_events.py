"""Unit tests for the admin EventBus (in-process pub/sub behind /admin/events)."""
import asyncio

import pytest

from admin_events import EventBus

pytestmark = pytest.mark.unit


async def test_publish_with_no_subscribers_is_noop():
    bus = EventBus()
    # Must not raise, must not allocate anything.
    bus.publish("call.started", "c-1", {"direction": "inbound"})
    assert bus.subscriber_count == 0


async def test_subscribe_publish_receive():
    bus = EventBus()
    q = bus.subscribe()
    bus.publish("user_turn", "c-42", {"text": "hello"})

    item = q.get_nowait()
    assert item["event"] == "user_turn"
    assert item["call_id"] == "c-42"
    assert item["data"] == {"text": "hello"}
    assert isinstance(item["ts"], float) and item["ts"] > 0


async def test_empty_call_id_becomes_dash():
    bus = EventBus()
    q = bus.subscribe()
    bus.publish("tool_call", "", {"tool": "CALC", "success": True})
    assert q.get_nowait()["call_id"] == "-"


async def test_unsubscribe_stops_delivery():
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    assert bus.subscriber_count == 0
    bus.publish("barge_in", "c-1", {})
    assert q.empty()
    # Unsubscribing an unknown/already-removed queue is harmless.
    bus.unsubscribe(q)
    bus.unsubscribe(asyncio.Queue())


async def test_fanout_to_multiple_subscribers():
    bus = EventBus()
    q1, q2 = bus.subscribe(), bus.subscribe()
    bus.publish("call.ended", "c-9", {})
    assert q1.get_nowait()["event"] == "call.ended"
    assert q2.get_nowait()["event"] == "call.ended"


async def test_bounded_queue_drops_oldest_without_blocking():
    bus = EventBus(maxsize=4)
    q = bus.subscribe()
    for i in range(10):
        # put_nowait semantics: this must never block or raise.
        bus.publish("user_turn", "c-1", {"n": i})
    assert q.qsize() == 4
    received = [q.get_nowait()["data"]["n"] for _ in range(4)]
    # Oldest were dropped; the newest 4 survive in order.
    assert received == [6, 7, 8, 9]


async def test_slow_subscriber_does_not_affect_others():
    bus = EventBus(maxsize=2)
    slow, fast = bus.subscribe(), bus.subscribe()
    bus.publish("a", "c", {"n": 0})
    bus.publish("b", "c", {"n": 1})
    bus.publish("c", "c", {"n": 2})  # overflows `slow` too (same maxsize)
    # Drain fast first: it also has maxsize 2, so it kept the newest 2.
    assert [fast.get_nowait()["data"]["n"] for _ in range(2)] == [1, 2]
    assert [slow.get_nowait()["data"]["n"] for _ in range(2)] == [1, 2]
