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


def _script(text: str) -> str:
    t = text.lower()
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
        messages = body.get("messages", [])
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content") or ""
                break
        content = _script(last_user)
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

    return app
