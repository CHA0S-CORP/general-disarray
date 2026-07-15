"""In-process mock of an OpenAI-compatible LLM server (stands in for vLLM).

Returns scripted `chat.completion` responses keyed off the last user message so
component tests can drive the text-based tool-call path deterministically:

  - "...simon..."/"repeat"/"echo"  -> emits [TOOL:SIMON_SAYS:text=the eagle has landed]
  - "...calc..."/"plus"/"math"      -> emits [TOOL:CALC:expression=2+2]
  - "...bye..."/"hang up"           -> emits [TOOL:HANGUP]
  - anything else                   -> a plain reply with no tool markers
"""
import asyncio
import json
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Fixed timestamp (no wall-clock needed; keeps responses reproducible).
_CREATED = 1700000000

ECHO_PHRASE = "the eagle has landed"

# Request bodies received by /v1/chat/completions, newest last. Tests that
# assert on what the engine actually sent (system prompt, sampling params)
# read this.
REQUESTS = []

# When True, tool_choice="required" requests get an OpenAI-style 400 —
# drives the grounding retry's "backend doesn't support it" fallback.
REJECT_TOOL_CHOICE = False

# Marker text the engines inject on the nudge fallback re-run.
NUDGE_MARKER = "do not answer from memory"


SPOKEN_REWRITE = "The deploy failed at five oh three PM on July eighth."

# --- streaming (stream=True) support -----------------------------------------

# Multi-sentence text served by the streaming scripts ("ramble"). Three
# sentences, each past the 25-char merge threshold.
STREAM_STORY = (
    "Sentence one is comfortably longer than the merge threshold. "
    "Sentence two also runs well past the merge threshold. "
    "Sentence three wraps everything up with plenty of length."
)

# Long pure-content answer for native-mode streaming ("explain"): word-by-word
# deltas, far more than the engine's 16-token hold.
STREAM_NATIVE_TEXT = (
    "Here is a thorough explanation that keeps going for quite a while. "
    "It has more than enough words to release the streaming hold. "
    "And it finishes with a third full sentence for good measure."
)

# "gated ..." streaming scripts send their first sentence, then wait for this
# event before sending the rest — lets tests prove sentences are consumed
# BEFORE the stream completes. The mock server runs in another thread's event
# loop, so this is a threading.Event polled with short sleeps.
STREAM_GATE = threading.Event()

# Prompts (lowercased last-user text) whose streaming response generator was
# torn down before reaching [DONE] — i.e. the client closed the HTTP stream
# mid-generation (barge-in). Appended from the mock server thread.
STREAM_ABORTS = []


def _sse_chunk(body: dict, delta: dict, finish_reason=None) -> str:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": _CREATED,
        "model": body.get("model", "mock-model"),
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _pieces(text: str, size: int):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _word_deltas(text: str):
    words = text.split(" ")
    return [w + " " for w in words[:-1]] + [words[-1]]


async def _await_gate():
    while not STREAM_GATE.is_set():
        await asyncio.sleep(0.01)


async def _stream_body(body: dict, last_user: str):
    """SSE generator for stream=True requests. Scripts keyed off the last
    user message, mirroring the non-streaming scripts."""
    t = last_user.lower()
    done = False
    try:
        yield _sse_chunk(body, {"role": "assistant"})

        if "endless" in t:
            # Never finishes on its own: complete sentences forever, so the
            # client speaks some audio and then must CLOSE the stream.
            n = 0
            while True:
                n += 1
                yield _sse_chunk(body, {
                    "content": ("This is endless sentence number "
                                f"{n}, easily long enough to emit. ")})
                await asyncio.sleep(0.02)

        if body.get("tools"):
            if "calc" in t or "math" in t:
                # Tool round: id/name first, arguments split across deltas.
                yield _sse_chunk(body, {"tool_calls": [{
                    "index": 0, "id": "call_mock_1", "type": "function",
                    "function": {"name": "CALC", "arguments": ""}}]})
                yield _sse_chunk(body, {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": '{"expression"'}}]})
                yield _sse_chunk(body, {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": ': "2+2"}'}}]})
                yield _sse_chunk(body, {}, finish_reason="tool_calls")
                yield "data: [DONE]\n\n"
                done = True
                return
            # Pure content, word-by-word (>16 deltas -> hold releases).
            deltas = _word_deltas(STREAM_NATIVE_TEXT)
            if "gated" in t:
                head, tail = deltas[:20], deltas[20:]
                for d in head:
                    yield _sse_chunk(body, {"content": d})
                await _await_gate()
                deltas = tail
            for d in deltas:
                yield _sse_chunk(body, {"content": d})
        else:
            if "simon" in t or "repeat" in t or "echo" in t:
                # Marker script, 3-char deltas: the [TOOL: prefix crosses
                # delta boundaries.
                text = _script(last_user)
                for piece in _pieces(text, 3):
                    yield _sse_chunk(body, {"content": piece})
            elif "ramble" in t or "story" in t:
                text = STREAM_STORY
                if "gated" in t:
                    first_len = text.index(". ") + 2
                    for piece in _pieces(text[:first_len], 4):
                        yield _sse_chunk(body, {"content": piece})
                    await _await_gate()
                    text = text[first_len:]
                for piece in _pieces(text, 4):
                    yield _sse_chunk(body, {"content": piece})
            else:
                for piece in _pieces(_script(last_user), 4):
                    yield _sse_chunk(body, {"content": piece})

        yield _sse_chunk(body, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"
        done = True
    finally:
        if not done:
            STREAM_ABORTS.append(t)


def _script(text: str) -> str:
    t = text.lower()
    if "deploy failed at" in t:
        # Canned reformat_for_speech rewrite (raw alert text in, prose out).
        return SPOKEN_REWRITE
    if "simon" in t or "repeat" in t or "echo" in t:
        return f"Sure thing. [TOOL:SIMON_SAYS:text={ECHO_PHRASE}]"
    if "calc" in t or "plus" in t or "math" in t or "multiply" in t:
        return "Let me compute. [TOOL:CALC:expression=2+2]"
    if "bye" in t or "hang up" in t or "goodbye" in t:
        return "Goodbye. [TOOL:HANGUP]"
    if "ramble" in t or "story" in t:
        # Same multi-sentence text as the streaming script, so tests can
        # compare chunking behavior across the two paths.
        return STREAM_STORY
    return "Sure, I can help with that."


def build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": "mock-model", "object": "model", "created": _CREATED, "owned_by": "mock"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        REQUESTS.append(body)
        messages = body.get("messages", [])
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content") or ""
                break

        if body.get("stream"):
            return StreamingResponse(_stream_body(body, last_user),
                                     media_type="text/event-stream")

        # --- native tool-calling script (only when the request binds tools) ---
        if body.get("tools"):
            t = last_user.lower()
            tool_msgs = [m for m in messages if m.get("role") == "tool"]

            # Grounding-retry scripting: honor (or reject) tool_choice.
            if body.get("tool_choice") == "required":
                if REJECT_TOOL_CHOICE:
                    return JSONResponse(status_code=400, content={"error": {
                        "message": "tool_choice 'required' is not supported",
                        "type": "invalid_request_error"}})
                if not tool_msgs:
                    return _tool_call_response(body, "DATETIME", {})
            # Nudge fallback re-run: obey the injected grounding instruction.
            if not tool_msgs and any(
                    NUDGE_MARKER in str(m.get("content") or "")
                    for m in messages if m.get("role") == "system"):
                return _tool_call_response(body, "DATETIME", {})

            # "loop forever": keeps demanding tools no matter what came back,
            # so round/recursion limits can be exercised.
            if "loop forever" in t:
                return _tool_call_response(body, "SIMON_SAYS",
                                           {"text": "again"})
            if "joke" in t:
                # speak_result scenario: the model sees the joke come back but
                # only comments on it — the engine must still relay the joke.
                if tool_msgs:
                    return _completion_response(
                        body, "Hope that made you smile. Want another?")
                return _tool_call_response(body, "JOKE", {"category": "dad"})
            if "what time" in t and not tool_msgs:
                # Fabricated live-data answer with zero tool calls — the
                # grounding-retry trigger scenario.
                return _completion_response(body, "It is three o'clock.")
            if "gpu" in t and not tool_msgs:
                # Promises to check but ends the turn — the dead-air scenario.
                return _completion_response(
                    body, "Let me check on that, one moment.")
            if tool_msgs:
                # A tool result came back -> produce the final spoken answer
                # referencing it (proves the round trip fed results back).
                return _completion_response(
                    body, f"As requested: {tool_msgs[-1].get('content', '')}")
            if "simon" in t or "repeat" in t or "echo" in t:
                return _tool_call_response(body, "SIMON_SAYS",
                                           {"text": ECHO_PHRASE})

        # Text-mode nudge re-run: obey the injected grounding instruction by
        # emitting the marker the engine demanded.
        if any(NUDGE_MARKER in str(m.get("content") or "")
               for m in messages if m.get("role") == "system"):
            return _completion_response(body, "Checking. [TOOL:DATETIME]")

        content = _script(last_user)
        return _completion_response(body, content)

    return app


def _completion_response(body: dict, content: str) -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": _CREATED,
        "model": body.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tool_call_response(body: dict, name: str, arguments: dict) -> dict:
    import json as _json
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": _CREATED,
        "model": body.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_mock_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": _json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
