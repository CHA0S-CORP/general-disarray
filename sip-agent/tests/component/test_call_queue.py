"""Component tests for the Redis-backed CallQueue, using fakeredis (in-memory).

We bypass CallQueue.connect() and inject a FakeRedis client, then drive the real
enqueue/recover/worker logic.
"""
import asyncio

import fakeredis.aioredis
import pytest
import pytest_asyncio

from call_queue import CallQueue, QueuedCallStatus

pytestmark = pytest.mark.component


@pytest_asyncio.fixture
async def queue():
    q = CallQueue(redis_url="redis://localhost:6379/0", max_concurrent=2)
    q.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield q
    await q.redis.flushall()
    await q.redis.aclose()


def _request(extension="1001", message="hello"):
    from api import OutboundCallRequest
    return OutboundCallRequest(message=message, extension=extension)


async def test_enqueue_assigns_positions(queue):
    c1 = await queue.enqueue("a", _request())
    c2 = await queue.enqueue("b", _request())
    assert c1.position == 1
    assert c2.position == 2
    status = await queue.get_queue_status()
    assert status["queued"] == 2
    assert status["max_concurrent"] == 2


async def test_get_call_roundtrip(queue):
    await queue.enqueue("a", _request(extension="2002"))
    call = await queue.get_call("a")
    assert call is not None
    assert call.status == QueuedCallStatus.QUEUED
    assert '"extension":"2002"' in call.request_json.replace(" ", "")


async def test_get_missing_call(queue):
    assert await queue.get_call("does-not-exist") is None


async def test_recover_processing_calls(queue):
    # Simulate a call that was mid-processing at shutdown.
    await queue.enqueue("a", _request())
    await queue.redis.blpop(queue.QUEUE_KEY, timeout=1)   # pop it
    await queue.redis.sadd(queue.PROCESSING_KEY, "a")     # mark processing

    await queue._recover_processing_calls()

    status = await queue.get_queue_status()
    assert status["queued"] == 1          # requeued
    assert status["processing"] == 0      # cleared
    assert (await queue.get_call("a")).status == QueuedCallStatus.QUEUED


async def test_worker_processes_enqueued_call(queue):
    processed = []

    class FakeHandler:
        async def _execute_call(self, call_id, request):
            processed.append((call_id, request.extension))

    await queue.start(FakeHandler())
    try:
        await queue.enqueue("job1", _request(extension="3003"))
        # Worker pops via blpop(timeout=1); wait for completion.
        for _ in range(60):
            call = await queue.get_call("job1")
            if call and call.status == QueuedCallStatus.COMPLETED:
                break
            await asyncio.sleep(0.1)
        assert processed == [("job1", "3003")]
        assert (await queue.get_call("job1")).status == QueuedCallStatus.COMPLETED
    finally:
        await queue.stop()
