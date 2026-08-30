#!/usr/bin/env python3
"""Fire one representative request at the LLM so the first real phone call
does not pay JIT/autotune cost.

Measured on GB10 with Ling-3.0-flash-int4, ~3.3K-token prompt + 29 tools:

    cold (first request ever): 35.6 s   <-- exceeds sip-agent's
    warm (second):              2.1 s        LLM_AGENT_TIMEOUT_S=30
    warm (third):               1.0 s

So without this, the first caller after a restart or reboot hits the agent
timeout and gets nothing. SGLang's own --skip-server-warmup=False warmup uses a
trivial prompt and does not cover the long-prompt-with-tools kernel shapes, and
--warmups only runs named functions built into warmup.py, so neither helps.

This deliberately mimics the real sip-agent turn shape (long system prompt +
large tool array) rather than sending "hello", because the cost is per kernel
shape. Failures are logged and ignored - a warmup that cannot run must never
block the stack from coming up.
"""

import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("LLM_WARMUP_URL", "http://vllm:8000/v1")
MODEL = os.environ.get("LLM_MODEL", "ling-3.0-flash")
TIMEOUT = float(os.environ.get("LLM_WARMUP_TIMEOUT_S", "180"))

# Roughly the sip-agent tool registry, in count and schema shape.
NAMES = [
    "alerts", "calc", "callback", "cancel", "coin", "datetime", "dice",
    "drink_recipe", "forecast", "forget", "gpu_status", "hangup", "joke",
    "knowledge", "memory", "news", "note", "number_fact", "quote", "recall",
    "remind", "schedule", "search", "spell", "timer", "transfer", "trivia",
    "weather", "wiki",
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": n,
            "description": (
                f"{n.replace('_', ' ').title()} tool for the voice assistant. "
                f"Use when the caller asks about {n.replace('_', ' ')}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what the caller asked"},
                    "detail": {"type": "string", "enum": ["short", "full"]},
                },
                "required": ["query"],
            },
        },
    }
    for n in NAMES
]
SYS = (
    "You are a helpful voice assistant answering a phone call. Keep replies short "
    "and conversational - one or two sentences, no markdown, no lists, since your "
    "text is read aloud. Call a tool for live information rather than guessing. "
    "Never invent facts. "
) * 3


def fire(label, user, tools):
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        "max_tokens": 64,
        "temperature": 0.6,
        "top_p": 0.85,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(
        BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read())
    el = time.perf_counter() - t0
    ptok = (d.get("usage") or {}).get("prompt_tokens", "?")
    print(f"warmup: {label:<12} {el:6.2f}s  prompt={ptok} tok", flush=True)


def main():
    # Both shapes: a plain reply and a tool-call reply.
    for label, user, tools in (
        ("chit-chat", "Hi there, how are you doing today?", TOOLS),
        ("tool-call", "What's the weather like in Oslo right now?", TOOLS),
        ("no-tools", "Say OK.", None),
    ):
        try:
            fire(label, user, tools)
        except Exception as e:  # noqa: BLE001 - never block startup
            print(f"warmup: {label} FAILED (ignored): {e!r}", file=sys.stderr, flush=True)
    print("warmup: done", flush=True)


if __name__ == "__main__":
    main()
