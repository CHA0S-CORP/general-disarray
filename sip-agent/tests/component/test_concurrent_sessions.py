"""Component tests for concurrent call sessions (04b).

Two live sessions must stay fully isolated: histories, tool_state, playback
tags, CALLBACK caller defaults (via the current-session contextvar), barge-in
and teardown. The shipped MAX_CONCURRENT_CALLS=1 behavior (replace/evict) is
covered too.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from call_session import TurnLedger, set_current_session

pytestmark = pytest.mark.component


def _call(uri, call_id=None, is_active=True):
    return SimpleNamespace(is_active=is_active, remote_uri=uri,
                           media_ready=False,
                           call_id=call_id or f"sip-{uri}")


@pytest.fixture
def real_assistant(comp_config):
    # The real orchestrator; SIPHandler falls back to its mock without pjsua2.
    from main import SIPAIAssistant
    return SIPAIAssistant(comp_config)


def _patch_audio(assistant):
    """Stub TTS + RTP so turns run without live services; returns the list of
    (call_info, audio) pairs actually 'played'.

    The stubs yield to the event loop (sleep(0)) — awaits that complete
    inline would make 'concurrent' turns run strictly serially, hiding any
    cross-session bug that needs a turn to suspend mid-turn while the
    sibling's turn runs."""
    sent = []

    async def synthesize(text):
        await asyncio.sleep(0)
        return b"\x01\x02" * 80

    async def send_audio(call_info, audio, tag=None):
        await asyncio.sleep(0)
        sent.append((call_info, audio))

    assistant.audio_pipeline.synthesize = synthesize
    assistant.sip_handler.send_audio = send_audio
    return sent


def _fake_stream_factory(reply_prefix="echo"):
    """stream_response stand-in: one sentence + final, derived from the last
    user message so each session's reply is distinguishable."""

    def stream_response(history, call_context=None):
        last_user = next((m["content"] for m in reversed(history)
                          if m.get("role") == "user"), "")
        text = f"{reply_prefix}: {last_user}"

        async def gen():
            yield {"type": "sentence", "text": text}
            yield {"type": "final", "text": text}

        return gen()

    return stream_response


# --- Isolation across two live sessions --------------------------------------

async def test_bound_tasks_resolve_their_own_session(real_assistant):
    """With two sessions live, a task bound via the contextvar sees ITS
    session through assistant.session/current_call; unbound callers see None
    (ambiguous) — the loud-failure mode for any missed call site."""
    a = real_assistant
    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")

    # Unbound (REST-style) access is ambiguous with 2 sessions.
    assert a.session is None
    assert a.current_call is None
    assert a.conversation_history == []

    async def check(sess):
        set_current_session(sess)
        assert a.session is sess
        assert a.current_call is sess.call_info
        # Playback-tag allocation and tool_state land on the bound session.
        ledger = TurnLedger()
        tag = a._allocate_tag(ledger, "hello")
        assert tag == 1
        a.session.tool_state["who"] = sess.transcript_id
        await asyncio.sleep(0.01)  # interleave with the sibling
        assert a.session is sess

    await asyncio.gather(asyncio.create_task(check(s1)),
                         asyncio.create_task(check(s2)))
    assert s1.next_playback_tag == 2 and s2.next_playback_tag == 2
    assert s1.tool_state["who"] == s1.transcript_id
    assert s2.tool_state["who"] == s2.transcript_id
    await a._teardown_session()


async def test_stale_bound_session_resolves_none_not_other_call(real_assistant):
    """A task bound to a torn-down session must get None — never the other
    live session."""
    a = real_assistant
    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")
    await a._teardown_session(s1)

    async def stale_task():
        set_current_session(s1)
        return a.session, a.current_call

    session, call = await asyncio.create_task(stale_task())
    assert session is None and call is None
    # An unbound caller now sees the sole surviving session.
    assert a.session is s2
    await a._teardown_session()


async def test_interleaved_turns_have_isolated_histories(real_assistant):
    a = real_assistant
    _patch_audio(a)
    a.llm_engine.stream_response = _fake_stream_factory()

    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")

    events = []

    async def turn(sess, text):
        set_current_session(sess)
        events.append(("start", sess.transcript_id))
        await a._handle_transcription(sess, text)
        events.append(("end", sess.transcript_id))

    await asyncio.gather(
        asyncio.create_task(turn(s1, "hi from alice")),
        asyncio.create_task(turn(s2, "hi from bob")),
    )

    # The turns must GENUINELY interleave (bob's turn starts before alice's
    # finishes) — otherwise serial execution makes isolation trivially true
    # and cross-session bugs on the speaking path go undetected.
    assert (events.index(("start", s2.transcript_id))
            < events.index(("end", s1.transcript_id)))

    assert [m["content"] for m in s1.conversation_history] == [
        "hi from alice", "echo: hi from alice"]
    assert [m["content"] for m in s2.conversation_history] == [
        "hi from bob", "echo: hi from bob"]
    # Transcripts are per-session too.
    t1 = a.transcripts.get(s1.transcript_id)
    t2 = a.transcripts.get(s2.transcript_id)
    assert [t["content"] for t in t1["turns"]] == [
        "hi from alice", "echo: hi from alice"]
    assert [t["content"] for t in t2["turns"]] == [
        "hi from bob", "echo: hi from bob"]
    await a._teardown_session()


async def test_callback_defaults_to_each_sessions_caller(real_assistant):
    """The CALLBACK caller-default special case resolves the calling session
    through the contextvar — each session's callback targets ITS caller."""
    from llm_engine import ToolCall
    a = real_assistant
    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")

    async def run_callback(sess):
        set_current_session(sess)
        return await a.tool_manager.execute_tool(ToolCall(
            name="CALLBACK",
            params={"delay": "60", "message": "your callback"}, raw=""))

    r1 = await asyncio.create_task(run_callback(s1))
    r2 = await asyncio.create_task(run_callback(s2))
    assert r1.status.value == "success" and r2.status.value == "success"

    targets = sorted(t.target_uri for t in
                     a.tool_manager.scheduled_tasks.values()
                     if t.task_type == "callback")
    assert targets == ["sip:alice@host", "sip:bob@host"]
    await a._teardown_session()


async def test_barge_in_on_one_session_leaves_other_turn_running(real_assistant):
    a = real_assistant
    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")

    async def long_turn():
        await asyncio.sleep(30)

    s1.turn_task = asyncio.create_task(long_turn())
    s2.turn_task = asyncio.create_task(long_turn())
    await asyncio.sleep(0)

    await a._handle_barge_in(s1)

    assert s1.turn_task is None            # cancelled and cleared
    assert s2.turn_task is not None and not s2.turn_task.done()
    s2.turn_task.cancel()
    await a._teardown_session()


async def test_teardown_of_one_session_leaves_other_functional(real_assistant):
    a = real_assistant
    _patch_audio(a)
    a.llm_engine.stream_response = _fake_stream_factory()

    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")

    await a._teardown_session(s1)
    assert list(a.sessions.values()) == [s2]

    async def turn(sess, text):
        set_current_session(sess)
        await a._handle_transcription(sess, text)

    await asyncio.create_task(turn(s2, "still there?"))
    assert [m["content"] for m in s2.conversation_history] == [
        "still there?", "echo: still there?"]
    await a._teardown_session()


# --- Inbound admission (capacity / duplicates / replacement) -----------------

def _assistant_with_cap(config_factory, speaches_url, vllm_url, cap):
    cfg = config_factory(speaches_api_url=speaches_url,
                         llm_base_url=f"{vllm_url}/v1",
                         llm_model="mock-model",
                         max_concurrent_calls=str(cap))
    from main import SIPAIAssistant
    return SIPAIAssistant(cfg)


async def test_cap2_second_inbound_call_adds_session(config_factory,
                                                     speaches_url, vllm_url):
    a = _assistant_with_cap(config_factory, speaches_url, vllm_url, 2)
    a.running = True
    _patch_audio(a)
    try:
        await asyncio.create_task(a._on_call_received(_call("sip:1@h", "c-1")))
        await asyncio.create_task(a._on_call_received(_call("sip:2@h", "c-2")))
        assert set(a.sessions) == {"c-1", "c-2"}
        # Both audio loops are alive.
        for s in a.sessions.values():
            assert s.audio_loop_task is not None and not s.audio_loop_task.done()

        # A duplicate INVITE callback for a live call id is suppressed.
        await asyncio.create_task(a._on_call_received(_call("sip:1@h", "c-1")))
        assert set(a.sessions) == {"c-1", "c-2"}

        # A third call at capacity replaces the oldest session (this path is
        # only reachable with the SIP busy gate off; the gate itself is unit-
        # tested in test_sip_busy.py).
        await asyncio.create_task(a._on_call_received(_call("sip:3@h", "c-3")))
        assert set(a.sessions) == {"c-2", "c-3"}
    finally:
        a.running = False
        await a._teardown_session()


async def test_cap1_second_inbound_call_replaces_first(real_assistant):
    """Shipping default (MAX_CONCURRENT_CALLS=1): exactly today's replace-the-
    session behavior for calls that get past the SIP busy gate."""
    a = real_assistant
    a.running = True
    _patch_audio(a)
    try:
        await asyncio.create_task(a._on_call_received(_call("sip:1@h", "c-1")))
        first = a.sessions["c-1"]
        await asyncio.create_task(a._on_call_received(_call("sip:2@h", "c-2")))
        assert set(a.sessions) == {"c-2"}
        assert first.audio_loop_task.done() or first.audio_loop_task.cancelled()
    finally:
        a.running = False
        await a._teardown_session()


async def test_capacity_eviction_prefers_dead_session_over_live(
        config_factory, speaches_url, vllm_url):
    """A hung-up call leaves PJSIP immediately but its session lingers in the
    registry until its audio loop notices; a new INVITE admitted in that
    window must evict the DEAD session — never a live call."""
    a = _assistant_with_cap(config_factory, speaches_url, vllm_url, 2)
    a.running = True
    _patch_audio(a)
    try:
        await asyncio.create_task(a._on_call_received(_call("sip:1@h", "c-1")))
        await asyncio.create_task(a._on_call_received(_call("sip:2@h", "c-2")))
        live = a.sessions["c-1"]
        dead = a.sessions["c-2"]

        # Freeze the lag window: c-2's caller hangs up (is_active False) but
        # its audio loop never gets to detach the session.
        dead.audio_loop_task.cancel()
        await asyncio.sleep(0)
        dead.call_info.is_active = False

        await asyncio.create_task(a._on_call_received(_call("sip:3@h", "c-3")))

        assert set(a.sessions) == {"c-1", "c-3"}
        assert a.sessions["c-1"] is live
        assert live.call_info.is_active  # the live call was never torn down
    finally:
        a.running = False
        await a._teardown_session()


async def test_ended_call_frees_its_registry_slot(real_assistant):
    """A naturally-ended call's audio loop detaches its own session, so dead
    sessions can't accumulate (or block /speak defaulting) at cap > 1."""
    a = real_assistant
    a.running = True
    _patch_audio(a)
    try:
        call = _call("sip:1@h", "c-1", is_active=False)  # ends immediately
        await asyncio.create_task(a._on_call_received(call))
        # The audio loop notices the dead call and detaches its own session
        # (it may already have finished by the time we get control back).
        for _ in range(100):
            if not a.sessions:
                break
            await asyncio.sleep(0.05)
        assert a.sessions == {}
    finally:
        a.running = False
        await a._teardown_session()


# --- Timers with concurrent calls ---------------------------------------------

async def test_timer_fires_into_the_session_that_set_it(real_assistant):
    """A timer set during call A announces into call A even with a second
    call live: the ScheduledTask carries its originating session, because
    the scheduler task's own context is unbound (with 2 sessions the
    assistant.current_call compat property resolves to None)."""
    from llm_engine import ToolCall
    a = real_assistant
    spoken = []

    async def stream_response(call_info, text):
        spoken.append((call_info, text))

    a._stream_response = stream_response
    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")

    async def set_timer(sess):
        set_current_session(sess)
        return await a.tool_manager.execute_tool(ToolCall(
            name="SET_TIMER",
            params={"duration": "60", "message": "your pasta is done"},
            raw=""))

    r = await asyncio.create_task(set_timer(s1))
    assert r.status.value == "success"
    task = next(t for t in a.tool_manager.scheduled_tasks.values()
                if t.task_type == "timer")
    assert task.session is s1

    # Fire it exactly the way the scheduler does: from a fresh task whose
    # current-session contextvar is unbound.
    await asyncio.create_task(a.tool_manager._execute_scheduled_task(task))

    assert spoken == [(s1.call_info, "your pasta is done")]
    assert s2.conversation_history == []
    await a._teardown_session()


async def test_timer_expires_silently_when_its_call_has_ended(real_assistant):
    """When the call that set a timer hangs up before it fires, the
    announcement is dropped — never spoken into the OTHER caller's call,
    even though that call is now the sole registered session."""
    from llm_engine import ToolCall
    a = real_assistant
    spoken = []

    async def stream_response(call_info, text):
        spoken.append((call_info, text))

    a._stream_response = stream_response
    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")

    async def set_timer(sess):
        set_current_session(sess)
        return await a.tool_manager.execute_tool(ToolCall(
            name="SET_TIMER",
            params={"duration": "60", "message": "alice's secret reminder"},
            raw=""))

    await asyncio.create_task(set_timer(s1))
    task = next(t for t in a.tool_manager.scheduled_tasks.values()
                if t.task_type == "timer")

    # Alice hangs up; bob is now the sole registered session (the state in
    # which the old sole-session fallback leaked alice's timer to bob).
    await a._teardown_session(s1)

    await asyncio.create_task(a.tool_manager._execute_scheduled_task(task))

    assert spoken == []
    await a._teardown_session()


async def test_unbound_timer_still_falls_back_to_sole_call(real_assistant):
    """A timer scheduled outside any call (REST /tools path: no bound
    session) keeps the legacy behavior — spoken into the sole active call."""
    a = real_assistant
    spoken = []

    async def stream_response(call_info, text):
        spoken.append((call_info, text))

    a._stream_response = stream_response
    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")

    task_id = await a.tool_manager.schedule_task(
        task_type="timer", delay_seconds=60, message="rest timer")
    task = a.tool_manager.scheduled_tasks[task_id]
    assert task.session is None

    await asyncio.create_task(a.tool_manager._execute_scheduled_task(task))
    assert spoken == [(s1.call_info, "rest timer")]
    await a._teardown_session()


# --- Shutdown drain -----------------------------------------------------------

async def test_drain_speaks_goodbye_to_all_sessions(real_assistant):
    a = real_assistant
    sent = _patch_audio(a)
    hung = []

    async def hangup_call(call_info):
        hung.append(call_info)
        call_info.is_active = False

    a.sip_handler.hangup_call = hangup_call

    s1 = a._begin_session(_call("sip:alice@host"), "inbound", "sip:alice@host")
    s2 = a._begin_session(_call("sip:bob@host"), "inbound", "sip:bob@host")

    start = time.monotonic()
    await a.drain_active_call(turn_timeout=1.0)
    elapsed = time.monotonic() - start

    # Both callers heard a goodbye and were hung up; the two drains ran
    # concurrently. Each drain includes a fixed 2s playback grace, so
    # concurrent ≈ 2s while a sequential regression (a for-loop instead of
    # the gather) ≈ 4s — the bound must sit between them to discriminate.
    played_to = {id(ci) for ci, _ in sent}
    assert {id(s1.call_info), id(s2.call_info)} <= played_to
    assert {id(c) for c in hung} == {id(s1.call_info), id(s2.call_info)}
    assert elapsed < 3.5
    await a._teardown_session()
