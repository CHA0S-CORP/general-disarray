"""Component tests: barge-in truthfulness (the playback ledger).

When the caller interrupts mid-response, conversation history and the
transcript must record only the prefix the caller actually heard — not the
full generated response the old code wrote before playback even started.
"""
import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.component

MARKER = " [interrupted by caller]"

# Three sentences, each > 25 chars so split_into_sentences keeps them as
# three separate playback chunks.
RESPONSE = (
    "The weather right now is sunny and mild outside. "
    "Later this afternoon clouds will move in from the west. "
    "Tomorrow morning expect some light rain before it clears."
).strip()


def _call(uri="sip:1001@host"):
    return SimpleNamespace(is_active=True, remote_uri=uri, media_ready=False)


class FakePlaylistPlayer:
    """Playback-ledger double: the test scripts completed/current/fraction.

    ``audio_pending`` scripts has_audio(): while True, the real turn parks in
    _wait_for_playback_drain (audio enqueued but not yet played out)."""

    def __init__(self):
        self.completed = []
        self.current = None
        self.fraction = 0.0
        self.cleared = False
        self.audio_pending = False

    def snapshot(self):
        return (list(self.completed), self.current, self.fraction)

    def clear(self):
        self.cleared = True
        self.audio_pending = False

    def has_audio(self):
        return self.audio_pending


@pytest.fixture
def assistant(config_factory, speaches_url, vllm_url, monkeypatch):
    """Real SIPAIAssistant (mock-SIP mode) with a scripted LLM + instant TTS."""
    from main import SIPAIAssistant
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
    )
    a = SIPAIAssistant(cfg)

    # Scripted at the engine seam: with no OpenAI client the engine's
    # stream_response takes the default path, which runs generate_response
    # and replays its text sentence-by-sentence (same chunks as before).
    async def fake_generate(conversation_history, call_context=None):
        return RESPONSE

    async def fake_synthesize(text):
        return b"\x00" * 640  # pretend-TTS: no HTTP, no pipeline start needed

    monkeypatch.setattr(a.llm_engine, "generate_response", fake_generate)
    monkeypatch.setattr(a.audio_pipeline, "synthesize", fake_synthesize)
    return a


def _wire_player(monkeypatch, assistant):
    player = FakePlaylistPlayer()
    monkeypatch.setattr(assistant.sip_handler, "get_playlist_player",
                        lambda call_info: player)
    return player


async def test_barge_in_records_only_heard_prefix(assistant, monkeypatch):
    a = assistant
    player = _wire_player(monkeypatch, a)

    second_chunk_playing = asyncio.Event()
    sent_tags = []

    async def fake_send(call, audio, tag=None):
        sent_tags.append(tag)
        if len(sent_tags) == 1:
            # Chunk 1 finished playing before the interruption.
            player.completed.append(tag)
        elif len(sent_tags) == 2:
            # Chunk 2 is mid-playback (80%) when the caller barges in; the
            # turn task is cancelled while parked at this await.
            player.current = tag
            player.fraction = 0.8
            second_chunk_playing.set()
            await asyncio.Event().wait()  # block until cancelled

    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    turn = asyncio.create_task(
        a._handle_transcription(session, "what is the weather like"))
    await asyncio.wait_for(second_chunk_playing.wait(), timeout=5)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    content = assistant_turns[0]["content"]

    # A strict, non-empty prefix of the generated response + the marker.
    assert content.endswith(MARKER)
    heard = content[:-len(MARKER)]
    assert heard
    assert RESPONSE.startswith(heard)
    assert heard != RESPONSE
    # Chunks 1 and 2 were heard (2 at 80% >= threshold), chunk 3 never played.
    assert heard == (
        "The weather right now is sunny and mild outside. "
        "Later this afternoon clouds will move in from the west.")

    # Transcript store shows the same truncated text.
    record = a.transcripts.get(session.transcript_id)
    transcript_assistant = [t for t in record["turns"] if t["role"] == "assistant"]
    assert [t["content"] for t in transcript_assistant] == [content]

    await a._teardown_session()


async def test_barge_in_before_anything_heard_records_nothing(assistant, monkeypatch):
    a = assistant
    _wire_player(monkeypatch, a)  # player never completes anything

    first_send = asyncio.Event()

    async def fake_send(call, audio, tag=None):
        first_send.set()
        await asyncio.Event().wait()  # chunk 1 never finishes

    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    turn = asyncio.create_task(a._handle_transcription(session, "hello there"))
    await asyncio.wait_for(first_send.wait(), timeout=5)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert [m for m in session.conversation_history
            if m["role"] == "assistant"] == []
    record = a.transcripts.get(session.transcript_id)
    assert [t for t in record["turns"] if t["role"] == "assistant"] == []

    await a._teardown_session()


async def _wait_for(predicate, timeout=5.0):
    """Poll until predicate() is true (the real code has no event to await)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        assert asyncio.get_event_loop().time() < deadline, "condition never met"
        await asyncio.sleep(0.01)


async def test_barge_in_during_playback_tail_records_only_heard_prefix(
        assistant, monkeypatch):
    """The realistic case: send_audio is NON-blocking (the real enqueue never
    suspends), so _speak returns with seconds of audio still queued. The turn
    must stay alive until playback drains, and a barge-in in that tail must
    still truncate history."""
    a = assistant
    player = _wire_player(monkeypatch, a)
    player.audio_pending = True  # enqueued audio still playing after _speak

    sent_tags = []

    async def fake_send(call, audio, tag=None):
        sent_tags.append(tag)  # realistic: enqueue-and-return, no blocking

    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    turn = asyncio.create_task(
        a._handle_transcription(session, "what is the weather like"))
    session.turn_task = turn

    # All three chunks enqueued; the turn must now be parked waiting for the
    # player to drain — NOT completed with the full response pre-recorded.
    await _wait_for(lambda: len(sent_tags) == 3)
    await asyncio.sleep(0.05)
    assert not turn.done(), (
        "turn completed while audio was still queued/playing — history was "
        "written before playback resolved")
    assert [m for m in session.conversation_history
            if m["role"] == "assistant"] == []

    # Caller barges in: chunk 1 finished, chunk 2 is 80% played, chunk 3 was
    # still queued and gets flushed unheard.
    player.completed = [sent_tags[0]]
    player.current = sent_tags[1]
    player.fraction = 0.8
    await a._handle_barge_in(session)
    assert player.cleared

    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    content = assistant_turns[0]["content"]
    assert content == (
        "The weather right now is sunny and mild outside. "
        "Later this afternoon clouds will move in from the west." + MARKER)

    record = a.transcripts.get(session.transcript_id)
    transcript_assistant = [t for t in record["turns"] if t["role"] == "assistant"]
    assert [t["content"] for t in transcript_assistant] == [content]

    await a._teardown_session()


async def test_turn_completes_with_full_response_after_playback_drains(
        assistant, monkeypatch):
    """Regression guard for the drain wait itself: once the player runs dry,
    the turn finishes and records the full response with no marker."""
    a = assistant
    player = _wire_player(monkeypatch, a)
    player.audio_pending = True

    sent_tags = []

    async def fake_send(call, audio, tag=None):
        sent_tags.append(tag)

    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    turn = asyncio.create_task(
        a._handle_transcription(session, "what is the weather like"))

    await _wait_for(lambda: len(sent_tags) == 3)
    await asyncio.sleep(0.05)
    assert not turn.done()

    # Playback finishes: every chunk played to completion, queue empty.
    player.completed = list(sent_tags)
    player.audio_pending = False
    await asyncio.wait_for(turn, timeout=5)

    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert assistant_turns == [{"role": "assistant", "content": RESPONSE}]

    await a._teardown_session()


async def test_tts_failure_mid_response_records_only_played_chunks(
        assistant, monkeypatch):
    """Speaches down mid-response: synthesize returns b'' for chunks 2-3, so
    they are never enqueued. History must keep only chunk 1, not the full
    generated text."""
    a = assistant
    player = _wire_player(monkeypatch, a)

    calls = []

    async def failing_synthesize(text):
        calls.append(text)
        return b"\x00" * 640 if len(calls) == 1 else b""

    async def fake_send(call, audio, tag=None):
        if tag is not None:
            player.completed.append(tag)  # chunk 1 plays to completion

    monkeypatch.setattr(a.audio_pipeline, "synthesize", failing_synthesize)
    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    await a._handle_transcription(session, "what is the weather like")

    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert assistant_turns == [{
        "role": "assistant",
        "content": "The weather right now is sunny and mild outside.",
    }]
    record = a.transcripts.get(session.transcript_id)
    transcript_assistant = [t for t in record["turns"] if t["role"] == "assistant"]
    assert [t["content"] for t in transcript_assistant] == [
        "The weather right now is sunny and mild outside."]

    await a._teardown_session()


async def test_uninterrupted_turn_records_full_response(assistant, monkeypatch):
    a = assistant
    player = _wire_player(monkeypatch, a)

    async def fake_send(call, audio, tag=None):
        if tag is not None:
            player.completed.append(tag)  # every chunk plays to completion

    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    await a._handle_transcription(session, "what is the weather like")

    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert assistant_turns == [{"role": "assistant", "content": RESPONSE}]
    assert MARKER not in assistant_turns[0]["content"]

    record = a.transcripts.get(session.transcript_id)
    transcript_assistant = [t for t in record["turns"] if t["role"] == "assistant"]
    assert [t["content"] for t in transcript_assistant] == [RESPONSE]
    assert session.active_ledger is None

    await a._teardown_session()


GOODBYE = "Goodbye, thanks so much for calling today."


async def test_farewell_barge_in_records_interrupted_goodbye(assistant, monkeypatch):
    """A barge-in while the goodbye is audibly playing must abort the hangup
    AND mark the goodbye as interrupted (goodbyes are cached, so _speak
    returns before the caller has heard a word)."""
    a = assistant
    player = _wire_player(monkeypatch, a)
    player.audio_pending = True  # goodbye enqueued, still playing

    hangups = []

    async def fake_hangup(call_info):
        hangups.append(call_info)

    sent_tags = []

    async def fake_send(call, audio, tag=None):
        sent_tags.append(tag)  # non-blocking, like the real enqueue

    monkeypatch.setattr(a, "get_random_goodbye", lambda: GOODBYE)
    monkeypatch.setattr(a.sip_handler, "hangup_call", fake_hangup)
    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    turn = asyncio.create_task(a._handle_transcription(session, "goodbye"))
    session.turn_task = turn

    await _wait_for(lambda: len(sent_tags) == 1)
    await asyncio.sleep(0.05)
    assert not turn.done(), "farewell turn ended before the goodbye played out"

    # Caller changes their mind 80% into the goodbye.
    player.current = sent_tags[0]
    player.fraction = 0.8
    await a._handle_barge_in(session)
    assert player.cleared
    assert hangups == [], "barge-in during the goodbye must abort the hangup"

    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert assistant_turns == [
        {"role": "assistant", "content": GOODBYE + MARKER}]
    record = a.transcripts.get(session.transcript_id)
    transcript_assistant = [t for t in record["turns"] if t["role"] == "assistant"]
    assert [t["content"] for t in transcript_assistant] == [GOODBYE + MARKER]

    await a._teardown_session()


async def test_farewell_completion_records_full_goodbye_and_hangs_up(
        assistant, monkeypatch):
    a = assistant
    import main as main_mod
    monkeypatch.setattr(main_mod, "HANGUP_DELAY_SECONDS", 0)
    player = _wire_player(monkeypatch, a)

    hangups = []

    async def fake_hangup(call_info):
        hangups.append(call_info)

    async def fake_send(call, audio, tag=None):
        if tag is not None:
            player.completed.append(tag)  # goodbye plays to completion

    monkeypatch.setattr(a, "get_random_goodbye", lambda: GOODBYE)
    monkeypatch.setattr(a.sip_handler, "hangup_call", fake_hangup)
    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    await a._handle_transcription(session, "goodbye")

    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert assistant_turns == [{"role": "assistant", "content": GOODBYE}]
    assert MARKER not in assistant_turns[0]["content"]
    assert len(hangups) == 1

    await a._teardown_session()
