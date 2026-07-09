"""In-process mock of an OpenAI-compatible LLM server (stands in for vLLM).

Returns scripted `chat.completion` responses keyed off the last user message so
component tests can drive the text-based tool-call path deterministically:

  - "...simon..."/"repeat"/"echo"  -> emits [TOOL:SIMON_SAYS:text=the eagle has landed]
  - "...calc..."/"plus"/"math"      -> emits [TOOL:CALC:expression=2+2]
  - "...bye..."/"hang up"           -> emits [TOOL:HANGUP]
  - anything else                   -> a plain reply with no tool markers
"""
from fastapi import FastAPI, Request

# Fixed timestamp (no wall-clock needed; keeps responses reproducible).
_CREATED = 1700000000

ECHO_PHRASE = "the eagle has landed"

# Request bodies received by /v1/chat/completions, newest last. Tests that
# assert on what the engine actually sent (system prompt, sampling params)
# read this.
REQUESTS = []


SPOKEN_REWRITE = "The deploy failed at five oh three PM on July eighth."


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

        # --- native tool-calling script (only when the request binds tools) ---
        if body.get("tools"):
            t = last_user.lower()
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            # "loop forever": keeps demanding tools no matter what came back,
            # so round/recursion limits can be exercised.
            if "loop forever" in t:
                return _tool_call_response(body, "SIMON_SAYS",
                                           {"text": "again"})
            if tool_msgs:
                # A tool result came back -> produce the final spoken answer
                # referencing it (proves the round trip fed results back).
                return _completion_response(
                    body, f"As requested: {tool_msgs[-1].get('content', '')}")
            if "simon" in t or "repeat" in t or "echo" in t:
                return _tool_call_response(body, "SIMON_SAYS",
                                           {"text": ECHO_PHRASE})

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
