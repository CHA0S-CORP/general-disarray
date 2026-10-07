"""Component regression tests for call-lifecycle fixes in main.py.

Covers: cancellation propagation through _cancel_turn / teardown (the
_call_lock deadlock), eviction hanging up the evicted SIP leg, dead-session
sweep before duplicate suppression (pjsua slot reuse), shutdown-drain
admission refusal, outbound-call failure signalling + backlog skip +
cancellation, virtual-number release on setup failure, speculative cancel-
merge disarming on tool execution / spoken audio, LOG_LEVEL fallback and
rate-limited audio-read error logging.
"""
import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from call_session import set_current_session

pytestmark = pytest.mark.component


def _call(uri, call_id=None, is_active=True, media_ready=False, **extra):
    return SimpleNamespace(is_active=is_active, remote_uri=uri,
                           media_ready=media_ready,
                           call_id=call_id or f"sip-{uri}", **extra)


def _assistant(config_factory, speaches_url, vllm_url, **overrides):
    cfg = config_factory(speaches_api_url=speaches_url,
                         llm_base_url=f"{vllm_url}/v1",
                         llm_model="mock-model", **overrides)
    from main import SIPAIAssistant
    a = SIPAIAssistant(cfg)
    a.running = True

    async def synthesize(text):
        await asyncio.sleep(0)
        return b"\x01\x02" * 80

    async def send_audio(call_info, audio, tag=None):
        await asyncio.sleep(0)

    a.audio_pipeline.synthesize = synthesize
    a.sip_handler.send_audio = send_audio
    return a


@pytest.fixture
def a(config_factory, speaches_url, vllm_url):
    assistant = _assistant(config_factory, speaches_url, vllm_url)
    yield assistant
    assistant.running = False


def _record_hangups(a):
    hung = []

    async def hangup_call(call_info):
        hung.append(call_info)

    a.sip_handler.hangup_call = hangup_call
    return hung


def _slow_cancel_turn(cleanup_started: asyncio.Event):
    """A turn whose cancellation takes a while to unwind (stream.aclose(),
    TTS teardown...): the window in which the deadlock used to bite."""
    async def turn():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await asyncio.sleep(5)
            raise
    return turn()


# --- 1. Cancellation propagation / _call_lock deadlock ------------------------

async def test_cancel_turn_does_not_swallow_callers_cancellation(a):
    s = a._begin_session(_call("sip:1@h", "c-1"), "inbound", "sip:1@h")
    cleanup = asyncio.Event()
    s.turn_task = asyncio.create_task(_slow_cancel_turn(cleanup))
    await asyncio.sleep(0)
    reached = []

    async def caller():
        await a._cancel_turn(s)
        reached.append(True)  # must NOT run once the caller is cancelled

    task = asyncio.create_task(caller())
    await asyncio.wait_for(cleanup.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert task.cancelled()
    assert reached == []
    await a._teardown_session()


async def test_cancel_turn_absorbs_only_the_turns_cancellation(a):
    s = a._begin_session(_call("sip:1@h", "c-1"), "inbound", "sip:1@h")
    s.turn_task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    await asyncio.wait_for(a._cancel_turn(s), 2)  # returns normally
    assert s.turn_task is None
    await a._teardown_session()


async def test_teardown_under_call_lock_during_loop_tail_does_not_deadlock(a):
    """The audio loop is at its tail, waiting for its turn to unwind, when a
    new call's admission (holding _call_lock) tears the session down. The
    loop used to swallow that cancellation and then block on _call_lock
    forever — every later call answered but silent."""
    call = _call("sip:1@h", "c-1")
    s = a._begin_session(call, "inbound", "sip:1@h")
    cleanup = asyncio.Event()
    s.turn_task = asyncio.create_task(_slow_cancel_turn(cleanup))
    s.audio_loop_task = asyncio.create_task(a._audio_processing_loop(s))
    await asyncio.sleep(0.05)

    call.is_active = False  # loop exits -> tail -> _cancel_turn
    await asyncio.wait_for(cleanup.wait(), 2)

    async def admit_new_call():
        async with a._call_lock:
            await a._teardown_session(s)

    await asyncio.wait_for(admit_new_call(), 3)
    assert s.audio_loop_task.done()
    assert a.sessions == {}
    assert not a._call_lock.locked()


# --- 2/6. Eviction hangs up the live leg; dead sessions swept first ----------

async def test_evicted_live_call_is_hung_up(a):
    hung = _record_hangups(a)
    await asyncio.create_task(a._on_call_received(_call("sip:1@h", "c-1")))
    first = a.sessions["c-1"].call_info
    await asyncio.create_task(a._on_call_received(_call("sip:2@h", "c-2")))
    try:
        assert set(a.sessions) == {"c-2"}
        assert hung == [first]
    finally:
        a.running = False
        await a._teardown_session()


async def test_dead_session_on_reused_slot_does_not_block_new_call(a):
    """pjsua reuses call slots (call_id is the slot index): a lingering dead
    session for slot 0 must not make the next call on slot 0 look like a
    duplicate INVITE (it would be answered and left silent)."""
    hung = _record_hangups(a)
    await asyncio.create_task(a._on_call_received(_call("sip:1@h", "0")))
    old = a.sessions["0"]
    # Freeze the lag window: the caller hung up but the loop hasn't noticed.
    old.audio_loop_task.cancel()
    await asyncio.sleep(0)
    old.call_info.is_active = False

    new_call = _call("sip:2@h", "0")
    await asyncio.create_task(a._on_call_received(new_call))
    try:
        assert a.sessions["0"].call_info is new_call
        assert a.sessions["0"].audio_loop_task is not None
        assert hung == []  # the dead leg needs no hangup
    finally:
        a.running = False
        await a._teardown_session()


async def test_duplicate_invite_for_live_call_still_suppressed(a):
    await asyncio.create_task(a._on_call_received(_call("sip:1@h", "c-1")))
    first = a.sessions["c-1"]
    await asyncio.create_task(a._on_call_received(_call("sip:1@h", "c-1")))
    try:
        assert a.sessions["c-1"] is first
    finally:
        a.running = False
        await a._teardown_session()


# --- 12. Draining refuses new calls --------------------------------------------

async def test_no_new_calls_admitted_while_draining(a):
    hung = _record_hangups(a)
    await a.drain_active_call()  # idle: just sets the flag
    call = _call("sip:1@h", "c-1")
    await asyncio.create_task(a._on_call_received(call))
    assert a.sessions == {}
    assert hung == [call]


# --- 8. Virtual number released when setup fails ------------------------------

async def test_virtual_number_released_when_begin_session_raises(a):
    entry = SimpleNamespace(id="vn-1", number="7000", wants=lambda s: False,
                            greeting=None, purpose=None)
    released = []
    a.virtual_numbers = SimpleNamespace(
        claim=lambda number: entry if number == "7000" else None,
        release=lambda entry_id: released.append(entry_id))

    def boom(*args, **kwargs):
        raise RuntimeError("session setup failed")

    a._begin_session = boom
    await asyncio.create_task(a._on_call_received(
        _call("sip:1@h", "c-1", local_uri="sip:7000@host")))
    assert released == ["vn-1"]
    assert a.sessions == {}


# --- 3/4/7. Outbound calls ------------------------------------------------------

async def test_outbound_dial_failure_raises(a):
    from main import OutboundCallFailed

    async def make_call(uri):
        return None

    a.sip_handler.make_call = make_call
    with pytest.raises(OutboundCallFailed):
        await a.make_outbound_call("1001", "hi")


async def test_outbound_busy_stops_ringing_and_raises(a):
    from main import OutboundCallFailed
    a.config.callback_ring_timeout_s = 30
    call = _call("sip:1001@h", "o-1", is_active=False, ended=True,
                 last_status_code=486)

    async def make_call(uri):
        return call

    a.sip_handler.make_call = make_call
    start = time.monotonic()
    with pytest.raises(OutboundCallFailed, match="486"):
        await a.make_outbound_call("1001", "hi")
    assert time.monotonic() - start < 2  # did not wait out the ring timeout


async def test_outbound_ring_timeout_raises_and_hangs_up(a):
    from main import OutboundCallFailed
    a.config.callback_ring_timeout_s = 1
    hung = _record_hangups(a)
    call = _call("sip:1001@h", "o-1", is_active=False, ended=False)

    async def make_call(uri):
        return call

    a.sip_handler.make_call = make_call
    with pytest.raises(OutboundCallFailed):
        await a.make_outbound_call("1001", "hi")
    assert hung == [call]


def _answered_outbound(a, tmp_path, backlog=3200):
    rec = tmp_path / "rec.wav"
    rec.write_bytes(b"\x00" * (44 + backlog))
    call = _call("sip:1001@h", "o-1", is_active=True, record_file=str(rec),
                 record_file_pos=0)

    async def make_call(uri):
        return call

    a.sip_handler.make_call = make_call
    return call, rec


async def test_outbound_listener_skips_recording_backlog(a, tmp_path):
    call, rec = _answered_outbound(a, tmp_path)
    _record_hangups(a)
    seen = []

    async def fake_loop(session):
        seen.append(session.call_info.record_file_pos)

    a._audio_processing_loop = fake_loop
    await a.make_outbound_call("1001", "hi")
    assert seen == [rec.stat().st_size]


async def test_outbound_begin_session_failure_not_masked(a, tmp_path, caplog):
    """_begin_session raising must not turn into an UnboundLocalError from
    the finally block (which referenced a never-assigned `session`)."""
    call, _ = _answered_outbound(a, tmp_path)
    hung = _record_hangups(a)

    def boom(*args, **kwargs):
        raise RuntimeError("begin failed")

    a._begin_session = boom
    with caplog.at_level(logging.ERROR):
        await a.make_outbound_call("1001", "hi")
    assert "begin failed" in caplog.text
    assert "UnboundLocalError" not in caplog.text
    assert hung == [call]


async def test_outbound_call_task_cancellation_propagates(a, tmp_path):
    _answered_outbound(a, tmp_path)
    _record_hangups(a)
    loop_started = asyncio.Event()

    async def fake_loop(session):
        loop_started.set()
        await asyncio.Event().wait()

    a._audio_processing_loop = fake_loop
    task = asyncio.create_task(a.make_outbound_call("1001", "hi"))
    await asyncio.wait_for(loop_started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert task.cancelled()
    assert a.sessions == {}


# --- 5. Speculative cancel-merge disarming --------------------------------------

async def test_tool_execution_disarms_speculative_merge(a):
    from llm_engine import ToolCall
    s = a._begin_session(_call("sip:1@h", "c-1"), "inbound", "sip:1@h")

    async def turn():
        set_current_session(s)
        s.speculative_turn_text = "what time is it"
        await a.tool_manager.execute_tool(
            ToolCall(name="NO_SUCH_TOOL", params={}, raw=""))
        return s.speculative_turn_text

    assert await asyncio.create_task(turn()) is None
    await a._teardown_session()


async def test_spoken_audio_disarms_but_earcon_does_not(a):
    s = a._begin_session(_call("sip:1@h", "c-1"), "inbound", "sip:1@h")

    async def turn():
        set_current_session(s)
        s.speculative_turn_text = "hello"
        await a._play_audio(a._chime_pcm)  # earcon: still nothing heard
        after_chime = s.speculative_turn_text
        await a._speak("One moment.")      # phrase ack / error phrase
        return after_chime, s.speculative_turn_text

    after_chime, after_speak = await asyncio.create_task(turn())
    assert after_chime == "hello"
    assert after_speak is None
    await a._teardown_session()


# --- 9. LOG_LEVEL fallback ------------------------------------------------------

def test_invalid_log_level_falls_back_to_info():
    from main import _resolve_log_level
    assert _resolve_log_level("debug") == logging.DEBUG
    assert _resolve_log_level("WARNING") == logging.WARNING
    assert _resolve_log_level("verbose") == logging.INFO
    assert _resolve_log_level(None) == logging.INFO


# --- 10. Audio read errors surface at WARNING, rate-limited ---------------------

async def test_audio_read_errors_logged_at_warning_rate_limited(a, caplog):
    async def broken_receive(call_info, timeout=0.1):
        await asyncio.sleep(0.005)
        raise OSError("recording vanished")

    a.sip_handler.receive_audio = broken_receive
    s = a._begin_session(_call("sip:1@h", "c-1", media_ready=True),
                         "inbound", "sip:1@h")
    with caplog.at_level(logging.WARNING, logger="main"):
        s.audio_loop_task = asyncio.create_task(a._audio_processing_loop(s))
        await asyncio.sleep(0.3)
        a.running = False
        await a._teardown_session()
    warnings = [r for r in caplog.records
                if "Audio read error" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
