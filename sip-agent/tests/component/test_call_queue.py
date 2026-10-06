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


# --- shutdown / atomicity regressions -----------------------------------------

async def test_cancelled_call_stays_in_processing_for_requeue(queue):
    """A graceful stop mid-call must not strand the call as 'processing'
    outside the processing set (CancelledError skips `except Exception`)."""
    started = asyncio.Event()

    class SlowHandler:
        async def _execute_call(self, call_id, request):
            started.set()
            await asyncio.sleep(3600)

    await queue.start(SlowHandler())
    await queue.enqueue("job1", _request())
    await asyncio.wait_for(started.wait(), timeout=5)
    await queue.stop()

    assert "job1" in await queue.redis.smembers(queue.PROCESSING_KEY)
    # The next start() requeues it.
    await queue._recover_processing_calls()
    assert await queue.redis.lrange(queue.QUEUE_KEY, 0, -1) == ["job1"]
    assert (await queue.get_call("job1")).status == QueuedCallStatus.QUEUED


async def test_worker_moves_call_atomically_into_inflight(queue):
    """BLMOVE hands a popped id to the in-flight list in one step; recovery
    requeues anything found there (crash between pop and SADD)."""
    await queue.enqueue("a", _request())
    moved = await queue.redis.blmove(queue.QUEUE_KEY, queue.INFLIGHT_KEY, 1, "LEFT", "RIGHT")
    assert moved == "a"
    # Simulated crash before SADD: not in the processing set, only in-flight.
    await queue._recover_processing_calls()
    assert await queue.redis.lrange(queue.QUEUE_KEY, 0, -1) == ["a"]
    assert await queue.redis.llen(queue.INFLIGHT_KEY) == 0


async def test_finished_call_clears_inflight(queue):
    class FakeHandler:
        async def _execute_call(self, call_id, request):
            return None

    await queue.start(FakeHandler())
    try:
        await queue.enqueue("j", _request())
        for _ in range(60):
            call = await queue.get_call("j")
            if call and call.status == QueuedCallStatus.COMPLETED:
                break
            await asyncio.sleep(0.1)
        assert await queue.redis.llen(queue.INFLIGHT_KEY) == 0
        assert await queue.redis.scard(queue.PROCESSING_KEY) == 0
    finally:
        await queue.stop()


async def test_enqueue_rejects_active_duplicate_atomically(queue):
    from call_queue import DuplicateCallError
    results = await asyncio.gather(
        *(queue.enqueue("dup", _request()) for _ in range(5)),
        return_exceptions=True)
    ok = [r for r in results if not isinstance(r, Exception)]
    dups = [r for r in results if isinstance(r, DuplicateCallError)]
    assert len(ok) == 1 and len(dups) == 4
    assert await queue.redis.lrange(queue.QUEUE_KEY, 0, -1) == ["dup"]


async def test_enqueue_allows_reuse_of_finished_id(queue):
    await queue.enqueue("r", _request())
    call = await queue.get_call("r")
    call.status = QueuedCallStatus.COMPLETED
    import json
    await queue.redis.set(f"{queue.CALL_PREFIX}r", json.dumps(call.to_dict()))
    await queue.redis.delete(queue.QUEUE_KEY)
    again = await queue.enqueue("r", _request())
    assert again.position == 1
