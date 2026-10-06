"""Regression tests for realtime_client fixes: late commit acks map to their
own turn, reconnect backoff only resets after a healthy connection, a receive
loop failure closes the old socket, and no fake 0ms STT latency is recorded."""
import asyncio
import json
import time

import pytest

import realtime_client as rc
from realtime_client import RealtimeWebSocketClient

pytestmark = pytest.mark.unit


class _FakeWS:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def close(self):
        self.closed = True


def _client(config):
    c = RealtimeWebSocketClient(config)
    c._ws = _FakeWS()
    c._connected = True
    return c


async def test_late_commit_ack_maps_to_its_own_turn(config):
    c = _client(config)
    # Turn 1 times out before the server even acks the commit.
    assert await c.commit_and_wait(0.01) == ""

    # Turn 2 commits; THEN turn 1's ack + transcript finally arrive.
    await c.commit_audio_buffer()
    fut = c._pending_transcript
    await c._handle_message({"type": "input_audio_buffer.committed", "item_id": "A"})
    await c._handle_message({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "A", "transcript": "turn one words"})
    assert not fut.done(), "turn 1's late transcript leaked into turn 2"

    await c._handle_message({"type": "input_audio_buffer.committed", "item_id": "B"})
    await c._handle_message({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "B", "transcript": "turn two words"})
    assert fut.result() == "turn two words"


async def test_no_fake_zero_latency_metric(config, monkeypatch):
    recorded = []
    monkeypatch.setattr(rc.Metrics, "record_stt_latency",
                        lambda *a, **k: recorded.append(a))
    c = _client(config)
    await c._handle_message({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "X", "transcript": "hi"})
    assert recorded == []


class _AsyncioProxy:
    """asyncio with sleep() recorded instead of slept (module-local patch)."""

    def __init__(self):
        self.sleeps = []

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, delay):
        self.sleeps.append(delay)


async def test_backoff_grows_across_accept_then_drop_cycles(config, monkeypatch):
    proxy = _AsyncioProxy()
    monkeypatch.setattr(rc, "asyncio", proxy)
    c = RealtimeWebSocketClient(config)

    async def accept():
        c._connected = True
        c._connected_at = time.monotonic()

    c._connect = accept

    for _ in range(3):  # server accepts, then drops immediately
        c._connected = False
        await c._handle_connection_failure()
        await c._reconnect_task
    base = rc.RECONNECT_BASE_DELAY_SECONDS
    assert proxy.sleeps == [base, base * 2, base * 4]

    # A connection that stayed healthy resets the backoff.
    c._connected_at = time.monotonic() - rc.HEALTHY_CONNECTION_SECONDS - 1
    c._connected = False
    await c._handle_connection_failure()
    await c._reconnect_task
    assert proxy.sleeps[-1] == base


async def test_receive_loop_error_closes_old_socket(config):
    class _BrokenWS(_FakeWS):
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("handler bug")

    c = RealtimeWebSocketClient(config)
    ws = _BrokenWS()
    c._ws = ws
    c._connected = True
    failures = []

    async def on_failure():
        failures.append(True)

    c._handle_connection_failure = on_failure
    await c._receive_loop()
    assert ws.closed is True
    assert c._ws is None
    assert failures == [True]
