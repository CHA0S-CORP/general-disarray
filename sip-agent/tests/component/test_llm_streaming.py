"""Component tests: LLM→TTS token streaming (stream_response).

Drives the engine's streaming paths against the mock vLLM server's SSE
endpoint, plus the full main.py consumption path (real SIPAIAssistant).

The hard invariant everywhere: the whitespace-normalized concatenation of the
yielded `sentence` events equals the whitespace-normalized `final` event —
main speaks sentences and records final, so a divergence would make history
lie about what the caller heard.
"""
import asyncio
import re
from types import SimpleNamespace

import pytest
import pytest_asyncio

import mock_vllm
from llm_engine import LLMEngine
from mock_vllm import ECHO_PHRASE, STREAM_NATIVE_TEXT, STREAM_STORY

pytestmark = pytest.mark.component


def _norm(text):
    return " ".join(text.split())


async def _collect(agen):
    events = []
    async for event in agen:
        events.append(event)
    return events


def _check_invariant(events):
    """Sentence concat == final (whitespace-normalized); final is terminal."""
    sentences = [e["text"] for e in events if e["type"] == "sentence"]
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1
    assert events[-1] is finals[0]
    assert _norm(" ".join(sentences)) == _norm(finals[0]["text"])
    return sentences, finals[0]["text"]


@pytest_asyncio.fixture
async def engine(assistant):
    eng = LLMEngine(assistant.config, assistant.tool_manager)
    await eng.start()
    yield eng
    await eng.stop()


@pytest_asyncio.fixture
async def native_engine(assistant, config_factory):
    cfg = config_factory(
        speaches_api_url=assistant.config.speaches_api_url,
        llm_base_url=assistant.config.llm_base_url,
        llm_model="mock-model",
        llm_tool_calling="native",
    )
    eng = LLMEngine(cfg, assistant.tool_manager)
    await eng.start()
    yield eng
    await eng.stop()


# --- classic (text-marker) streaming ------------------------------------


async def test_classic_stream_speaks_sentences_before_stream_end(engine):
    """First sentence event arrives while the server is still holding the
    rest of the completion behind the gate — incremental, not batch."""
    mock_vllm.STREAM_GATE.clear()
    agen = engine.stream_response(
        [{"role": "user", "content": "give us a gated ramble now"}])
    try:
        first = await asyncio.wait_for(agen.__anext__(), timeout=5)
    finally:
        mock_vllm.STREAM_GATE.set()
    assert first["type"] == "sentence"
    assert first["text"] == (
        "Sentence one is comfortably longer than the merge threshold.")

    events = [first] + await _collect(agen)
    sentences, final = _check_invariant(events)
    assert len(sentences) == 3
    assert _norm(final) == _norm(STREAM_STORY)
    assert mock_vllm.REQUESTS[-1].get("stream") is True


async def test_classic_stream_marker_executes_tool_and_never_emits_marker(engine):
    events = await _collect(engine.stream_response(
        [{"role": "user", "content": "please simon says something"}]))
    sentences, final = _check_invariant(events)
    # The marker text never reaches a speakable sentence event.
    assert all("[TOOL" not in s and "[" not in s for s in sentences)
    assert "[TOOL" not in final
    # The tool ran and its speak_result message is spoken after the prefix.
    assert ECHO_PHRASE in final
    assert final.startswith("Sure thing.")
    assert mock_vllm.REQUESTS[-1].get("stream") is True


async def test_live_data_question_takes_non_streaming_path(engine):
    """Grounding pre-classification: a live-data question must NOT stream —
    it keeps today's full grounding-retry behavior (nudged re-run)."""
    before = len(mock_vllm.REQUESTS)
    events = await _collect(engine.stream_response(
        [{"role": "user", "content": "what time is it"}]))
    reqs = mock_vllm.REQUESTS[before:]
    assert reqs, "no LLM request made"
    assert all(not r.get("stream") for r in reqs)
    # Existing grounding behavior intact: original + nudged re-run.
    assert len(reqs) == 2
    assert any(mock_vllm.NUDGE_MARKER in str(m.get("content") or "")
               for m in reqs[-1]["messages"] if m.get("role") == "system")
    _, final = _check_invariant(events)
    assert re.search(r"\d{1,2}:\d{2} (AM|PM)", final), final


async def test_llm_streaming_false_uses_non_streaming_requests(
        assistant, config_factory):
    cfg = config_factory(
        speaches_api_url=assistant.config.speaches_api_url,
        llm_base_url=assistant.config.llm_base_url,
        llm_model="mock-model",
        llm_streaming="false",
    )
    eng = LLMEngine(cfg, assistant.tool_manager)
    await eng.start()
    try:
        before = len(mock_vllm.REQUESTS)
        events = await _collect(eng.stream_response(
            [{"role": "user", "content": "hello there"}]))
    finally:
        await eng.stop()
    reqs = mock_vllm.REQUESTS[before:]
    assert len(reqs) == 1 and not reqs[0].get("stream")
    sentences, final = _check_invariant(events)
    assert final == "Sure, I can help with that."
    assert sentences == [final]


class _ExplodingStream:
    """Backend stream that dies mid-response, before a sentence boundary."""

    def __init__(self, texts):
        self._texts = texts
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for text in self._texts:
            yield _content_chunk(text)
        raise RuntimeError("connection reset by peer")

    async def close(self):
        self.closed = True


async def test_classic_midstream_error_before_first_sentence_speaks_error_phrase(
        assistant, config_factory):
    """A stream that dies after tokens arrived but before any sentence was
    emitted must fall back to the error phrase — not flush the dangling
    fragment (here even an unterminated [TOOL: marker) as the spoken turn."""
    cfg = config_factory()
    eng = LLMEngine(cfg, assistant.tool_manager)
    fake_stream = _ExplodingStream(["Sure. ", "[TOOL", ":WEA"])

    async def create(**kwargs):
        return fake_stream

    eng.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    events = await _collect(eng.stream_response(
        [{"role": "user", "content": "tell me something nice"}]))
    sentences, final = _check_invariant(events)
    assert final in cfg.phrases.errors
    assert "Sure" not in final and "[TOOL" not in final
    assert all("[TOOL" not in s for s in sentences)
    assert fake_stream.closed


async def test_classic_midstream_error_after_spoken_sentence_keeps_prefix(
        assistant, config_factory):
    """Counterpart: once a sentence WAS spoken it can't be unspoken — the
    turn keeps the heard prefix rather than replacing it with an error."""
    first = "This opening sentence is comfortably past the merge threshold. "
    cfg = config_factory()
    eng = LLMEngine(cfg, assistant.tool_manager)
    fake_stream = _ExplodingStream([first, "And then it di"])

    async def create(**kwargs):
        return fake_stream

    eng.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    events = await _collect(eng.stream_response(
        [{"role": "user", "content": "tell me something nice"}]))
    sentences, final = _check_invariant(events)
    assert sentences[0] == first.strip()
    assert final not in cfg.phrases.errors
    assert fake_stream.closed


async def test_stream_setup_error_replays_precached_phrase_whole(
        assistant, config_factory):
    """Error phrases are pre-cached whole in the TTS cache: the replay path
    must hand a multi-sentence pre-cached phrase to main as ONE chunk so the
    whole-text cache lookup hits (splitting it would issue live TTS calls on
    the failure path)."""
    phrase = "I didn't quite catch that. One more time?"
    cfg = config_factory(phrases_errors=f'["{phrase}"]')
    eng = LLMEngine(cfg, assistant.tool_manager)

    async def create(**kwargs):
        raise RuntimeError("vLLM unreachable")

    eng.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    events = await _collect(eng.stream_response(
        [{"role": "user", "content": "tell me something nice"}]))
    sentences, final = _check_invariant(events)
    assert final == phrase
    assert sentences == [phrase]


async def test_non_streaming_fallback_phrase_stays_whole(
        assistant, config_factory):
    """Same whole-phrase guarantee on the default (_stream_via_generate)
    path: generate_response's error phrase must not be pre-split either."""
    phrase = "I didn't quite catch that. One more time?"
    cfg = config_factory(phrases_errors=f'["{phrase}"]', llm_streaming="false")
    eng = LLMEngine(cfg, assistant.tool_manager)

    async def create(**kwargs):
        raise RuntimeError("vLLM unreachable")

    eng.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    events = await _collect(eng.stream_response(
        [{"role": "user", "content": "tell me something nice"}]))
    sentences, final = _check_invariant(events)
    assert final == phrase
    assert sentences == [phrase]


async def test_tts_sentence_streaming_false_disables_token_streaming(
        assistant, config_factory):
    """TTS_SENTENCE_STREAMING=false means whole-text TTS: token streaming
    (which emits per-sentence chunks by construction) must not activate, and
    the turn takes the default path with a single whole-text chunk."""
    cfg = config_factory(
        speaches_api_url=assistant.config.speaches_api_url,
        llm_base_url=assistant.config.llm_base_url,
        llm_model="mock-model",
        tts_sentence_streaming="false",
    )
    eng = LLMEngine(cfg, assistant.tool_manager)
    await eng.start()
    try:
        before = len(mock_vllm.REQUESTS)
        events = await _collect(eng.stream_response(
            [{"role": "user", "content": "give us a plain ramble now"}]))
    finally:
        await eng.stop()
    reqs = mock_vllm.REQUESTS[before:]
    assert len(reqs) == 1 and not reqs[0].get("stream")
    sentences, final = _check_invariant(events)
    assert _norm(final) == _norm(STREAM_STORY)   # multi-sentence response...
    assert sentences == [final]                  # ...spoken as ONE chunk


async def test_engine_without_client_still_streams_via_default_path(assistant):
    """Mock mode (no OpenAI client): the default replay path serves the same
    event contract, so main has exactly one consumption path."""
    eng = LLMEngine(assistant.config, assistant.tool_manager)
    assert eng.client is None
    events = await _collect(eng.stream_response(
        [{"role": "user", "content": "help me out"}]))
    sentences, final = _check_invariant(events)
    assert sentences and final


# --- native tool-calling streaming ---------------------------------------


async def test_native_pure_content_streams_incrementally(native_engine):
    mock_vllm.STREAM_GATE.clear()
    before = len(mock_vllm.REQUESTS)
    agen = native_engine.stream_response(
        [{"role": "user", "content": "gated explain it to me"}])
    try:
        first = await asyncio.wait_for(agen.__anext__(), timeout=5)
    finally:
        mock_vllm.STREAM_GATE.set()
    assert first["type"] == "sentence"

    events = [first] + await _collect(agen)
    sentences, final = _check_invariant(events)
    assert len(sentences) == 3
    assert _norm(final) == _norm(STREAM_NATIVE_TEXT)
    req = mock_vllm.REQUESTS[before]
    assert req.get("stream") is True
    assert req.get("tools")


async def test_native_tool_call_first_falls_back_and_executes(native_engine):
    before = len(mock_vllm.REQUESTS)
    events = await _collect(native_engine.stream_response(
        [{"role": "user", "content": "do some calc for me"}]))
    sentences, final = _check_invariant(events)
    assert "4" in final          # CALC 2+2 executed for real and relayed
    assert sentences, "tool-round turns must still produce spoken sentences"

    reqs = mock_vllm.REQUESTS[before:]
    assert len(reqs) == 2
    assert reqs[0].get("stream") is True        # round 0 streamed
    assert not reqs[1].get("stream")            # continuation is non-streaming
    tool_msgs = [m for m in reqs[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and "4" in tool_msgs[0]["content"]


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        self.closed = True


def _content_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=text, tool_calls=None),
        finish_reason=None)])


def _tool_chunk(name, arguments):
    tcd = SimpleNamespace(index=0, id="tc-late", type="function",
                          function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=None, tool_calls=[tcd]),
        finish_reason=None)])


async def test_native_post_release_tool_call_keeps_invariant(
        assistant, config_factory):
    """The rare case: a tool_call delta AFTER the 16-token hold released and
    a sentence was already spoken. Emission stops, the tool executes, the
    turn continues non-streaming, and final == emitted prefix + follow-up."""
    first_sentence = ("Alpha beta gamma delta epsilon zeta eta theta iota "
                      "kappa lambda mu keeps going strong.")
    words = [w + " " for w in first_sentence.split()] + ["And", " then", " some"]
    chunks = ([_content_chunk(w) for w in words]
              + [_tool_chunk("CALC", '{"expression": "2+2"}')]
              + [SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="tool_calls")])])

    cfg = config_factory(llm_tool_calling="native")
    eng = LLMEngine(cfg, assistant.tool_manager)
    requests = []
    fake_stream = _FakeStream(chunks)

    async def create(**kwargs):
        requests.append(kwargs)
        if kwargs.get("stream"):
            return fake_stream
        tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
        msg = SimpleNamespace(
            content=f"The follow-up answer is: {tool_msgs[-1]['content']}",
            tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=msg, finish_reason="stop")], usage=None)

    eng.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    events = await _collect(eng.stream_response(
        [{"role": "user", "content": "please crunch those numbers"}]))
    sentences, final = _check_invariant(events)
    assert sentences[0] == first_sentence      # spoken before the tool call
    assert final.startswith(first_sentence)    # emitted prefix stays a prefix
    assert "4" in final                        # CALC executed and relayed
    assert fake_stream.closed                  # backend stream closed
    assert requests[0].get("stream") is True
    assert not requests[1].get("stream")


async def test_native_stream_withholds_tools_when_round_budget_is_zero(
        assistant, config_factory):
    """LLM_MAX_TOOL_ROUNDS=0 means no tool rounds: like _generate_native's
    budget guard, the streamed round-0 request must NOT carry the tools
    param, so the model can only answer in text (a tools-bearing request
    could return tool_calls, whose execution the zero budget forbids)."""
    cfg = config_factory(llm_tool_calling="native", llm_max_tool_rounds="0")
    eng = LLMEngine(cfg, assistant.tool_manager)
    text = "A plain text answer that easily clears the merge threshold."
    fake_stream = _FakeStream(
        [_content_chunk(text)]
        + [SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=None),
            finish_reason="stop")])])
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return fake_stream

    eng.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    events = await _collect(eng.stream_response(
        [{"role": "user", "content": "please transfer me to the front desk"}]))
    sentences, final = _check_invariant(events)
    assert requests and requests[0].get("stream") is True
    assert requests[0].get("tools") is None
    assert _norm(final) == _norm(text)


# --- cancellation closes the underlying HTTP stream ----------------------


async def test_aclose_shuts_down_backend_stream(engine):
    mock_vllm.STREAM_ABORTS.clear()
    agen = engine.stream_response(
        [{"role": "user", "content": "endless chatter"}])
    first = await asyncio.wait_for(agen.__anext__(), timeout=5)
    assert first["type"] == "sentence"
    await agen.aclose()

    for _ in range(100):                        # up to ~5s
        if mock_vllm.STREAM_ABORTS:
            break
        await asyncio.sleep(0.05)
    assert mock_vllm.STREAM_ABORTS, (
        "backend stream kept generating after the consumer closed")


# --- main.py consumption path (real SIPAIAssistant) ----------------------


MARKER = " [interrupted by caller]"


def _call(uri="sip:1001@host"):
    return SimpleNamespace(is_active=True, remote_uri=uri, media_ready=False)


class FakePlaylistPlayer:
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


@pytest_asyncio.fixture
async def live_assistant(config_factory, speaches_url, vllm_url, monkeypatch):
    """Real SIPAIAssistant with a real (streaming) engine on the mock vLLM,
    fake TTS recording every synthesized text, and a scripted player."""
    from main import SIPAIAssistant
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
    )
    a = SIPAIAssistant(cfg)
    await a.llm_engine.start()

    a.spoken = []          # every text handed to TTS, in order
    a.player = FakePlaylistPlayer()

    async def fake_synthesize(text):
        a.spoken.append(text)
        return b"\x00" * 640

    async def fake_send(call, audio, tag=None):
        if tag is not None:
            a.player.completed.append(tag)  # chunks play to completion

    monkeypatch.setattr(a.audio_pipeline, "synthesize", fake_synthesize)
    monkeypatch.setattr(a.sip_handler, "send_audio", fake_send)
    monkeypatch.setattr(a.sip_handler, "get_playlist_player",
                        lambda call_info: a.player)
    yield a
    await a._teardown_session()
    await a.llm_engine.stop()


async def _wait_for(predicate, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        assert asyncio.get_event_loop().time() < deadline, "condition never met"
        await asyncio.sleep(0.01)


async def test_main_speaks_streamed_sentences_and_records_final(live_assistant):
    """End-to-end through _handle_transcription: TTS is called per sentence
    BEFORE the LLM stream completes, and history records exactly the
    concatenation of the spoken sentences."""
    a = live_assistant
    mock_vllm.STREAM_GATE.clear()
    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    turn = asyncio.create_task(
        a._handle_transcription(session, "give us a gated ramble now"))
    try:
        # Sentence 1 reaches TTS while the mock still holds the rest of the
        # completion behind the gate: incremental, not batch.
        await _wait_for(lambda: len(a.spoken) >= 1)
        assert a.spoken[0] == (
            "Sentence one is comfortably longer than the merge threshold.")
    finally:
        mock_vllm.STREAM_GATE.set()
    await asyncio.wait_for(turn, timeout=10)

    assert len(a.spoken) == 3
    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    # The invariant, observed at the system boundary: recorded text is
    # exactly what was spoken.
    assert assistant_turns[0]["content"] == " ".join(a.spoken)
    record = a.transcripts.get(session.transcript_id)
    transcript_assistant = [t for t in record["turns"] if t["role"] == "assistant"]
    assert [t["content"] for t in transcript_assistant] == [
        assistant_turns[0]["content"]]


async def test_main_marker_response_never_reaches_tts(live_assistant):
    a = live_assistant
    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    await asyncio.wait_for(
        a._handle_transcription(session, "please simon says something"),
        timeout=10)

    assert a.spoken, "nothing was spoken"
    assert all("[TOOL" not in s for s in a.spoken)
    spoken_all = " ".join(a.spoken)
    assert ECHO_PHRASE in spoken_all
    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert assistant_turns[-1]["content"] == spoken_all


async def test_main_barge_in_mid_stream_truncates_and_closes_stream(
        live_assistant):
    a = live_assistant
    mock_vllm.STREAM_ABORTS.clear()
    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    turn = asyncio.create_task(
        a._handle_transcription(session, "endless chatter"))
    session.turn_task = turn

    await _wait_for(lambda: len(a.spoken) >= 2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    # History holds only the heard prefix, marked as interrupted.
    assistant_turns = [m for m in session.conversation_history
                       if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    content = assistant_turns[0]["content"]
    assert content.endswith(MARKER)
    heard = content[:-len(MARKER)]
    assert heard.startswith("This is endless sentence number 1")
    assert " ".join(a.spoken).startswith(heard)

    # ...and the engine's underlying HTTP stream was actually closed, so the
    # backend stopped generating.
    for _ in range(100):
        if mock_vllm.STREAM_ABORTS:
            break
        await asyncio.sleep(0.05)
    assert mock_vllm.STREAM_ABORTS, (
        "backend stream kept generating after the barge-in")


async def test_main_barge_in_during_tts_requires_explicit_stream_aclose(
        live_assistant, monkeypatch):
    """Discriminating coverage for main's `finally: await stream.aclose()`.

    On a real barge-in the cancel lands inside TTS/playback — NOT inside
    stream.__anext__ — so the event generator is left suspended at a yield
    and only main's explicit aclose tears down the backend stream. This test
    pins that window (TTS blocks until the turn is cancelled) and holds a
    reference to the generator so garbage-collection finalization cannot
    close it as a side effect: the assertions below pass only via main's
    explicit aclose call.
    """
    a = live_assistant
    mock_vllm.STREAM_ABORTS.clear()

    held_streams = []
    real_stream_response = a.llm_engine.stream_response

    def capturing_stream_response(*args, **kwargs):
        gen = real_stream_response(*args, **kwargs)
        held_streams.append(gen)
        return gen

    monkeypatch.setattr(a.llm_engine, "stream_response",
                        capturing_stream_response)

    tts_entered = asyncio.Event()

    async def blocking_synthesize(text):
        a.spoken.append(text)
        tts_entered.set()
        await asyncio.Event().wait()     # parks until the turn is cancelled

    monkeypatch.setattr(a.audio_pipeline, "synthesize", blocking_synthesize)

    session = a._begin_session(_call(), "inbound", "sip:1001@host")
    turn = asyncio.create_task(
        a._handle_transcription(session, "endless chatter"))
    session.turn_task = turn

    await asyncio.wait_for(tts_entered.wait(), timeout=5)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    # The event generator was explicitly closed (a merely-abandoned suspended
    # generator would resume and yield here instead).
    assert held_streams
    with pytest.raises(StopAsyncIteration):
        await held_streams[0].__anext__()

    # And closing it propagated to the backend HTTP stream.
    for _ in range(100):
        if mock_vllm.STREAM_ABORTS:
            break
        await asyncio.sleep(0.05)
    assert mock_vllm.STREAM_ABORTS, (
        "backend stream not closed by main's explicit stream.aclose()")
