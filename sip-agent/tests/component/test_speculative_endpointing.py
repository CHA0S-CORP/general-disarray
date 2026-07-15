"""Component tests: speculative endpointing (ENDPOINT_MODE=speculative).

The audio loop ends utterances at the short threshold and then:
- holds transcript fragments that don't read as a finished thought, merging
  them with follow-up speech into ONE dispatched turn;
- dispatches an incomplete held fragment anyway once ENDPOINT_MAX_SILENCE_MS
  passes with no follow-up (never hold forever);
- cancel-merges: a turn dispatched from a complete-looking fragment is
  cancelled if the caller resumes speaking BEFORE the assistant is audibly
  speaking, and the two transcripts re-dispatch as one utterance — with no
  assistant history from the cancelled attempt.
"""
import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.component

CHUNK = b"\x00" * 640  # 320 samples @16kHz = 20ms of PCM
CHUNK_MS = 20

FRAG_INCOMPLETE = "my number is five five five"
FRAG_FOLLOWUP = "one two three."
MERGED = f"{FRAG_INCOMPLETE} {FRAG_FOLLOWUP}"


def _call(uri="sip:1001@host"):
    return SimpleNamespace(is_active=True, remote_uri=uri, media_ready=True)


class SttScript:
    """Scripted STT: the test pushes transcripts; the fake pipeline pops one
    per audio chunk (None when empty — silence)."""

    def __init__(self):
        self.results = deque()

    def push(self, text):
        self.results.append(text)

    def pop(self):
        return self.results.popleft() if self.results else None


class FakePlaylistPlayer:
    def __init__(self):
        self.audio_pending = False
        self.completed = []
        self.current = None
        self.fraction = 0.0
        self.cleared = False

    def has_audio(self):
        return self.audio_pending

    def snapshot(self):
        return (list(self.completed), self.current, self.fraction)

    def clear(self):
        self.cleared = True
        self.audio_pending = False


async def _wait_for(predicate, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        assert asyncio.get_event_loop().time() < deadline, "condition never met"
        await asyncio.sleep(0.01)


def _make_assistant(config_factory, speaches_url, vllm_url, monkeypatch,
                    **overrides):
    from main import SIPAIAssistant
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
        endpoint_mode="speculative",
        **overrides,
    )
    a = SIPAIAssistant(cfg)
    a.running = True

    stt = SttScript()
    has_speech = {"value": False}

    # Paced at CHUNK's real duration (20ms of PCM per 20ms of wall clock).
    # The audio loop drains as fast as audio is available, and the barge-in /
    # cancel-merge debounces are measured in audio duration — so a mock that
    # emits 20ms chunks every 5ms would hand the loop 4x real-time audio and
    # trip a 400ms debounce in 100ms of wall clock.
    async def fake_receive(call_info, timeout=0.1):
        await asyncio.sleep(CHUNK_MS / 1000)
        return CHUNK

    # process_audio/has_speech now take the session's SessionAudioState as
    # their first argument (per-call pipeline state).
    async def fake_process_audio(state, chunk):
        return stt.pop()

    player = FakePlaylistPlayer()
    monkeypatch.setattr(a.sip_handler, "receive_audio", fake_receive)
    monkeypatch.setattr(a.sip_handler, "get_playlist_player",
                        lambda call_info: player)
    monkeypatch.setattr(a.audio_pipeline, "process_audio", fake_process_audio)
    monkeypatch.setattr(a.audio_pipeline, "has_speech",
                        lambda state, chunk: has_speech["value"])
    return a, stt, has_speech, player


def _start_loop(a):
    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    session.audio_loop_task = asyncio.create_task(
        a._audio_processing_loop(session))
    return session


async def _stop(a):
    a.running = False
    await a._teardown_session()


# ---------------------------------------------------------------------------
# Hold + merge (no turn until the utterance reads complete)
# ---------------------------------------------------------------------------

async def test_incomplete_fragment_held_then_merged_into_one_turn(
        config_factory, speaches_url, vllm_url, monkeypatch):
    a, stt, _, _ = _make_assistant(
        config_factory, speaches_url, vllm_url, monkeypatch,
        endpoint_max_silence_ms="10000")  # deadline must not interfere

    dispatched = []

    async def fake_run_turn(session, text):
        dispatched.append(text)

    monkeypatch.setattr(a, "_run_turn", fake_run_turn)
    session = _start_loop(a)
    try:
        stt.push(FRAG_INCOMPLETE)
        await _wait_for(lambda: session.held_fragment == FRAG_INCOMPLETE)
        # Held, NOT dispatched.
        await asyncio.sleep(0.1)
        assert dispatched == []

        stt.push(FRAG_FOLLOWUP)
        await _wait_for(lambda: dispatched)
        assert dispatched == [MERGED], "expected exactly ONE merged turn"
        assert session.held_fragment is None
    finally:
        await _stop(a)


async def test_complete_fragment_dispatches_immediately(
        config_factory, speaches_url, vllm_url, monkeypatch):
    a, stt, _, _ = _make_assistant(
        config_factory, speaches_url, vllm_url, monkeypatch)

    dispatched = []

    async def fake_run_turn(session, text):
        dispatched.append(text)

    monkeypatch.setattr(a, "_run_turn", fake_run_turn)
    session = _start_loop(a)
    try:
        stt.push("what's the weather like today?")
        await _wait_for(lambda: dispatched)
        assert dispatched == ["what's the weather like today?"]
        assert session.held_fragment is None
    finally:
        await _stop(a)


# ---------------------------------------------------------------------------
# Max-silence deadline: a held fragment is never held forever
# ---------------------------------------------------------------------------

async def test_held_fragment_dispatches_after_max_silence_deadline(
        config_factory, speaches_url, vllm_url, monkeypatch):
    a, stt, _, _ = _make_assistant(
        config_factory, speaches_url, vllm_url, monkeypatch,
        endpoint_max_silence_ms="200")

    dispatched = []

    async def fake_run_turn(session, text):
        dispatched.append(text)

    monkeypatch.setattr(a, "_run_turn", fake_run_turn)
    session = _start_loop(a)
    try:
        stt.push(FRAG_INCOMPLETE)
        await _wait_for(lambda: session.held_fragment == FRAG_INCOMPLETE)
        # No follow-up speech: the deadline dispatches the fragment as-is.
        await _wait_for(lambda: dispatched)
        assert dispatched == [FRAG_INCOMPLETE]
        assert session.held_fragment is None
    finally:
        await _stop(a)


# ---------------------------------------------------------------------------
# Cancel-merge: caller resumes before the assistant is audibly speaking
# ---------------------------------------------------------------------------

FIRST = "I need a reservation."
SECOND = "for six people tonight."
CANCEL_MERGED = f"{FIRST} {SECOND}"
RESPONSE = "Booked a table for six tonight at seven."


async def test_cancel_merge_redispatches_single_turn_without_ghost_history(
        config_factory, speaches_url, vllm_url, monkeypatch):
    a, stt, has_speech, player = _make_assistant(
        config_factory, speaches_url, vllm_url, monkeypatch,
        endpoint_max_silence_ms="10000")

    gen_started = asyncio.Event()
    gen_events = []
    gen_histories = []

    async def fake_generate(conversation_history, call_context=None):
        gen_histories.append(list(conversation_history))
        if len(gen_histories) == 1:
            gen_started.set()
            try:
                # First (speculative) turn: LLM still thinking, no audio yet.
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # The existing machinery must close the stream / cancel the
                # in-flight generation cleanly.
                gen_events.append("first_call_cancelled")
                raise
        return RESPONSE

    async def fake_synthesize(text):
        return b"\x00" * 640

    sent = []

    async def fake_send(call, audio, tag=None):
        sent.append(tag)

    monkeypatch.setattr(a.llm_engine, "generate_response", fake_generate)
    monkeypatch.setattr(a.audio_pipeline, "synthesize", fake_synthesize)
    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = _start_loop(a)
    try:
        # A complete-looking fragment dispatches a turn immediately.
        stt.push(FIRST)
        await asyncio.wait_for(gen_started.wait(), timeout=5)
        assert session.speculative_turn_text == FIRST

        # Caller resumes while the LLM is still thinking (nothing playing:
        # player.has_audio() is False) -> cancel-merge, not barge-in and not
        # pending_transcription.
        has_speech["value"] = True
        await _wait_for(lambda: session.held_fragment == FIRST)
        assert gen_events == ["first_call_cancelled"]
        assert session.turn_task is None or session.turn_task.done()
        # The re-seed happened AFTER _cancel_turn cleared held state.
        assert session.pending_transcription is None

        # The follow-up speech arrives; ONE merged turn runs to completion.
        has_speech["value"] = False
        stt.push(SECOND)
        await _wait_for(lambda: any(
            m["role"] == "assistant" for m in session.conversation_history))

        user_turns = [m["content"] for m in session.conversation_history
                      if m["role"] == "user"]
        assistant_turns = [m["content"] for m in session.conversation_history
                           if m["role"] == "assistant"]
        # Exactly one merged user utterance — the cancelled fragment's user
        # entry must not linger.
        assert user_turns == [CANCEL_MERGED]
        # No assistant turn from the cancelled attempt (it produced no
        # audio), only the merged turn's full response.
        assert assistant_turns == [RESPONSE]
        # The merged turn's LLM call saw the merged utterance.
        assert len(gen_histories) == 2
        last_user = [m for m in gen_histories[1] if m["role"] == "user"][-1]
        assert last_user["content"] == CANCEL_MERGED
        # The persisted transcript matches history: the cancelled fragment's
        # user turn was retracted, so no phantom duplicate reaches
        # GET /call/{id}/transcript, webhooks, or caller-memory extraction.
        record = a.transcripts.get(session.transcript_id)
        assert [t["content"] for t in record["turns"] if t["role"] == "user"] \
            == [CANCEL_MERGED]
        assert [t["content"] for t in record["turns"]
                if t["role"] == "assistant"] == [RESPONSE]
    finally:
        await _stop(a)


async def test_short_noise_blip_does_not_cancel_inflight_turn(
        config_factory, speaches_url, vllm_url, monkeypatch):
    """Cancel-merge has the same duration debounce as barge-in: a brief
    VAD-positive blip (click/pop/cough) while the LLM is thinking must NOT
    cancel the in-flight turn."""
    a, stt, has_speech, _ = _make_assistant(
        config_factory, speaches_url, vllm_url, monkeypatch,
        endpoint_max_silence_ms="10000")

    gen_started = asyncio.Event()
    cancelled = []

    async def fake_generate(conversation_history, call_context=None):
        gen_started.set()
        try:
            await asyncio.Event().wait()  # LLM "thinking" until cancelled
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    monkeypatch.setattr(a.llm_engine, "generate_response", fake_generate)

    session = _start_loop(a)
    try:
        stt.push(FIRST)
        await asyncio.wait_for(gen_started.wait(), timeout=5)

        # ~200ms of speech-positive audio (10 x 20ms chunks at real-time
        # pacing): a blip, comfortably under the 400ms debounce.
        has_speech["value"] = True
        await asyncio.sleep(0.2)
        has_speech["value"] = False
        await asyncio.sleep(0.3)

        assert cancelled == [], "noise blip must not cancel the LLM turn"
        assert session.turn_task and not session.turn_task.done()
        assert session.speculative_turn_text == FIRST
        assert session.held_fragment is None
    finally:
        await _stop(a)


async def test_cancel_merge_folds_pending_transcription(
        config_factory, speaches_url, vllm_url, monkeypatch):
    """Speech that completed while the turn was in flight is parked in
    pending_transcription; a later cancel-merge must fold it into the merged
    utterance instead of letting _cancel_turn silently discard it."""
    a, stt, has_speech, _ = _make_assistant(
        config_factory, speaches_url, vllm_url, monkeypatch,
        endpoint_max_silence_ms="10000",
        barge_in_min_duration="100")  # speed up the sustained-speech gate

    gen_started = asyncio.Event()
    gen_histories = []

    async def fake_generate(conversation_history, call_context=None):
        gen_histories.append(list(conversation_history))
        if len(gen_histories) == 1:
            gen_started.set()
            await asyncio.Event().wait()  # blocks until cancel-merge
        return RESPONSE

    async def fake_synthesize(text):
        return b"\x00" * 640

    async def fake_send(call, audio, tag=None):
        pass

    monkeypatch.setattr(a.llm_engine, "generate_response", fake_generate)
    monkeypatch.setattr(a.audio_pipeline, "synthesize", fake_synthesize)
    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = _start_loop(a)
    try:
        stt.push(FIRST)
        await asyncio.wait_for(gen_started.wait(), timeout=5)

        # A short addition completes while the turn is in flight -> parked
        # in pending_transcription (turn keeps running).
        stt.push("and tomorrow")
        await _wait_for(lambda: session.pending_transcription == "and tomorrow")

        # Caller resumes (sustained speech, nothing playing) -> cancel-merge
        # must keep BOTH the dispatched fragment and the pending addition.
        has_speech["value"] = True
        await _wait_for(lambda: session.held_fragment is not None
                        and session.turn_task is None)
        assert session.held_fragment == f"{FIRST} and tomorrow"
        has_speech["value"] = False

        stt.push("around noon.")
        await _wait_for(lambda: any(
            m["role"] == "assistant" for m in session.conversation_history))

        merged = f"{FIRST} and tomorrow around noon."
        user_turns = [m["content"] for m in session.conversation_history
                      if m["role"] == "user"]
        assert user_turns == [merged]
        last_user = [m for m in gen_histories[1] if m["role"] == "user"][-1]
        assert last_user["content"] == merged
    finally:
        await _stop(a)


async def test_cancel_merge_pops_history_for_unstripped_stt_text(
        config_factory, speaches_url, vllm_url, monkeypatch):
    """Realtime STT hands over unstripped transcripts (Whisper loves a
    leading space). History records the stripped text, so the cancel-merge
    retraction must compare stripped — otherwise the fragment's user entry
    lingers and the merged re-dispatch duplicates it in the LLM context."""
    a, stt, has_speech, _ = _make_assistant(
        config_factory, speaches_url, vllm_url, monkeypatch,
        endpoint_max_silence_ms="10000",
        barge_in_min_duration="100")

    gen_started = asyncio.Event()
    gen_histories = []

    async def fake_generate(conversation_history, call_context=None):
        gen_histories.append(list(conversation_history))
        if len(gen_histories) == 1:
            gen_started.set()
            await asyncio.Event().wait()
        return RESPONSE

    async def fake_synthesize(text):
        return b"\x00" * 640

    async def fake_send(call, audio, tag=None):
        pass

    monkeypatch.setattr(a.llm_engine, "generate_response", fake_generate)
    monkeypatch.setattr(a.audio_pipeline, "synthesize", fake_synthesize)
    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = _start_loop(a)
    try:
        stt.push(f"  {FIRST}")  # unstripped, as the realtime client delivers
        await asyncio.wait_for(gen_started.wait(), timeout=5)

        has_speech["value"] = True
        await _wait_for(lambda: session.held_fragment == FIRST)
        has_speech["value"] = False

        stt.push(SECOND)
        await _wait_for(lambda: any(
            m["role"] == "assistant" for m in session.conversation_history))

        user_turns = [m["content"] for m in session.conversation_history
                      if m["role"] == "user"]
        assert user_turns == [CANCEL_MERGED]
        record = a.transcripts.get(session.transcript_id)
        assert [t["content"] for t in record["turns"] if t["role"] == "user"] \
            == [CANCEL_MERGED]
    finally:
        await _stop(a)


async def test_turn_that_already_spoke_is_not_cancel_merged(
        config_factory, speaches_url, vllm_url, monkeypatch):
    """Once a turn's first audio is enqueued, cancel-merge's "produced no
    audio" precondition is gone for good — caller speech in an
    inter-sentence playback gap (player momentarily drained while sentence 2
    is still in TTS) must NOT cancel the half-delivered turn; it flows to
    pending_transcription instead."""
    a, stt, has_speech, _ = _make_assistant(
        config_factory, speaches_url, vllm_url, monkeypatch,
        endpoint_max_silence_ms="10000",
        barge_in_min_duration="100")

    spoke = asyncio.Event()
    cancelled = []

    async def fake_stream(conversation_history, call_context=None):
        yield {"type": "sentence", "text": "Sure."}
        try:
            await asyncio.Event().wait()  # sentence 2 stuck in LLM/TTS
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def fake_synthesize(text):
        return b"\x00" * 640

    async def fake_send(call, audio, tag=None):
        spoke.set()

    monkeypatch.setattr(a.llm_engine, "stream_response", fake_stream)
    monkeypatch.setattr(a.audio_pipeline, "synthesize", fake_synthesize)
    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = _start_loop(a)
    try:
        stt.push(FIRST)
        await asyncio.wait_for(spoke.wait(), timeout=5)
        # First audio enqueued -> the cancel-merge branch is disarmed.
        await _wait_for(lambda: session.speculative_turn_text is None)

        # The fake player reports no audio (the inter-sentence gap), and the
        # caller reacts — well past the sustained-speech gate.
        has_speech["value"] = True
        await asyncio.sleep(0.8)
        has_speech["value"] = False

        assert cancelled == [], "half-delivered turn must not be cancelled"
        assert session.turn_task and not session.turn_task.done()
        assert session.held_fragment is None

        # Speech during the gap is a follow-up for AFTER the turn.
        stt.push("thanks")
        await _wait_for(lambda: session.pending_transcription == "thanks")
    finally:
        await _stop(a)
