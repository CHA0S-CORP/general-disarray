"""Unit tests for the current-session ContextVar (call_session).

The variable is the concurrency backbone of 04b: tasks acting on behalf of
one call bind their session at the top of the task body, and children spawned
afterwards inherit it. The inverse — setting it AFTER create_task — must NOT
propagate, because asyncio tasks snapshot their context at creation time.
"""
import asyncio

import pytest

from call_session import (CallSession, get_current_session,
                          set_current_session)

pytestmark = pytest.mark.unit


def _session(tid: str = "t1") -> CallSession:
    return CallSession(call_info=object(), direction="inbound",
                       transcript_id=tid)


async def test_default_is_none():
    assert get_current_session() is None


async def test_set_and_get_in_same_task():
    s = _session()
    token = set_current_session(s)
    try:
        assert get_current_session() is s
    finally:
        # Reset so the test task doesn't leak the binding to later tests.
        import call_session
        call_session.current_session.reset(token)


async def test_child_task_inherits_binding_set_before_spawn():
    s = _session()

    async def child():
        return get_current_session()

    async def parent():
        set_current_session(s)
        return await asyncio.create_task(child())

    # parent runs as its own task so its set() can't leak into the test task.
    assert await asyncio.create_task(parent()) is s


async def test_set_after_create_task_does_not_propagate():
    """Tasks snapshot their context at create_task: a set() in the parent
    AFTER the spawn must be invisible to the child."""
    s = _session()
    started = asyncio.Event()
    release = asyncio.Event()

    async def child():
        started.set()
        await release.wait()
        return get_current_session()

    async def parent():
        task = asyncio.create_task(child())
        await started.wait()
        set_current_session(s)  # too late: child already snapshotted
        release.set()
        return await task

    assert await asyncio.create_task(parent()) is None


async def test_sibling_tasks_see_only_their_own_binding():
    s1, s2 = _session("a"), _session("b")
    seen = {}

    async def worker(name, session):
        set_current_session(session)
        await asyncio.sleep(0.01)  # interleave with the sibling
        seen[name] = get_current_session()

    await asyncio.gather(asyncio.create_task(worker("a", s1)),
                         asyncio.create_task(worker("b", s2)))
    assert seen == {"a": s1, "b": s2}
