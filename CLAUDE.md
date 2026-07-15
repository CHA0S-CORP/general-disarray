# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`general-disarray` ("General Disarray") is a self-hosted, voice-powered AI phone assistant: it answers SIP phone calls, transcribes speech, runs an LLM, executes tools, and speaks back — all on local/NVIDIA hardware (notably DGX Spark / GB10). The Python app lives in `sip-agent/src/`. License is AGPL-3.0.

This is a git repo with submodules (see `.gitmodules`):
- `speaches/` — STT (Whisper) + TTS (Kokoro/Piper) server, upstream `speaches-ai/speaches`. **Has its own `CLAUDE.md` — follow it when editing inside `speaches/`** (it mandates modern `list`/`dict` typing, `ruff`, `pyright`, `pytest`; different rules than the agent).
- `nvitop/` — GPU monitoring + Prometheus exporter (vendored upstream). Don't modify.
- `docs/` — documentation source (published to readme.io).

## Architecture

The agent is **pure orchestration** — all ML inference is offloaded to external HTTP services, so the container ships no models and stays lightweight. Services are wired together in the compose files:

- **sip-agent** — this Python app (FastAPI + PJSUA2). Ports 5060 (SIP) + 10000-10100/udp (RTP) + 8080 (REST API).
- **vllm** — LLM inference, OpenAI-compatible API on :8000.
- **speaches** — unified STT + TTS, OpenAI-compatible API on :8001.
- **redis** — outbound-call queue on :6379.
- **n8n** — optional workflow automation on :5678.

Call flow: PJSIP receives RTP audio → VAD/STT (Speaches) → LLM (vLLM) → tool calls parsed from text → TTS (Speaches) → RTP back to caller.

### Code map (`sip-agent/src/`)

- `main.py` — `SIPAIAssistant` orchestrator + `async def main()` entry point. Owns the audio loop, barge-in, greeting/phrase playback (sentence-streaming TTS), outbound calls, callback scheduling, and graceful shutdown (drains the active call). Wires up all components and starts FastAPI + the Redis call-queue worker.
- `call_session.py` — `CallSession`: all per-call state (conversation history, in-flight turn task, pending transcript, audio loop). `SIPAIAssistant.session` is the single active session; `assistant.current_call`/`conversation_history` are read-only compat properties over it. Replace sessions via `_teardown_session()` + `_begin_session()` (hold `_call_lock`).
- `transcript_store.py` — `TranscriptStore`: per-call conversation transcripts (live + bounded LRU + persisted JSON under `data/transcripts/`), served by `GET /call/{id}/transcript`.
- `config.py` — `Config` dataclass; **every setting is an env var** read via `os.getenv` with a default. Source of truth for configuration. Also holds `PhrasesConfig` and the base `system_prompt`.
- `sip_handler.py` — SIP/RTP via **PJSUA2** (`pjsua2`); falls back to a mock handler if the lib is missing.
- `audio_pipeline.py` — `LowLatencyAudioPipeline`: WebRTC VAD, STT/TTS calls to Speaches, and a TTS audio cache (common phrases pre-cached at startup).
- `realtime_client.py` — `RealtimeWebSocketClient`: optional low-latency streaming STT over Speaches' `/v1/realtime` WebSocket (OpenAI Realtime API protocol). Used only when `STT_MODE=realtime`; default is `batch` (file upload).
- `llm_engine.py` — LLM abstraction over an OpenAI-compatible client. **Tool calls default to text markers**: the LLM emits `[TOOL:NAME:param=value,...]` markers, which are regex-parsed, executed, then stripped from the spoken response. Set `LLM_TOOL_CALLING=native` to use the OpenAI tools/function-calling API instead (`_generate_native`, bounded by `LLM_MAX_TOOL_ROUNDS`; the text parser still runs afterwards as a safety net via `_apply_marker_tools`). Backends selected by `LLM_BACKEND` (`vllm` default, `ollama`, `lmstudio`, `langgraph`) via `create_llm_engine()`. Also hosts the one-shot utility completions (`reformat_for_speech`, `summarize_text`) and injects caller memory / rolling summary / knowledge context from `call_context` in `_build_system_prompt`. LLM responses can legitimately be `None` (e.g. vLLM/gpt-oss returning empty content); `_generate` already guards this — preserve that handling.
- `langchain_engine.py` — `LangChainEngine(LLMEngine)`: the agentic engine (`LLM_BACKEND=langgraph`), a LangGraph ReAct loop over the same backend that can chain several tool calls per turn (`LLM_MAX_TOOL_ROUNDS` rounds, `LLM_AGENT_TIMEOUT_S` wall clock). Wants `LLM_TOOL_CALLING=native` plus a vLLM started with tool-call parsing (`VLLM_TOOL_ARGS` in the compose files; the parser must match the model — `hermes` for Qwen3, `openai` for gpt-oss); in `text` mode the agent runs unbound and the marker parser still handles tools. LangChain imports are guarded: missing deps fall back to the classic engine, runtime errors fall back to a spoken error phrase.
- `caller_memory.py` — `CallerMemoryStore`: cross-call memory per caller (`data/caller_memory/<caller>.json`, keyed by the SIP URI user part). Facts are LLM-extracted from the transcript after each call (fire-and-forget from both teardown paths in `main.py`) and injected into the system prompt at call start. `CALLER_MEMORY_ENABLED` (default on); fail-open everywhere.
- `knowledge_base.py` + `plugins/knowledge_tool.py` — RAG: `.txt`/`.md` files in `data/knowledge/` are chunked and embedded locally (fastembed ONNX on CPU; model cached under `data/models/fastembed`) into a persisted langchain `InMemoryVectorStore` (`data/knowledge_index/`), retrievable via the `KNOWLEDGE` tool in every tool-calling mode (plus optional `KNOWLEDGE_AUTO_INJECT` prompt injection). The tool registers only when the KB is `available` (enabled + deps + documents present); restart to reindex changed documents.
- `context_manager.py` — rolling conversation summary: when a call outgrows `MAX_CONVERSATION_TURNS`, overflow turns are folded into `session.rolling_summary` by a background task (never on the speaking path) and injected as a "Conversation so far" block instead of being silently dropped. `CONVERSATION_SUMMARY_ENABLED` (default on).
- `tool_manager.py` — Loads the built-in tool set, runs the background scheduler for timers/callbacks/scheduled-calls (`_run_scheduler`; callbacks and scheduled calls persist across restarts via `data/scheduled_tasks.json`), and executes tool calls. Outbound-call tasks are serialized by `_outbound_call_lock` (one live call session). The `CALLBACK` tool is special-cased here so it defaults to the current caller's number.
- `tool_plugins.py` — `BaseTool` / `ToolResult` / `ToolStatus` base classes plus `PluginLoader`/`ToolRegistry` for filesystem plugin discovery.
- `api.py` — FastAPI app (`create_api`): outbound calls, tool listing/execution, TTS `/speak`, scheduling CRUD.
- `call_queue.py` — Redis-backed, concurrency-limited outbound call queue.
- `grounding.py` — pure detectors behind the "never guess" enforcement: when a live-data question (weather/time/quakes/GPU/alerts/search) ends a turn with zero tool calls — or the reply merely promises to check — both engines re-run once forcing tool use (`GROUNDING_RETRY_ENABLED`, `grounding_retry` log event). Spoken times/schedules use `LOCAL_TIMEZONE` (`config.local_timezone`); a second inbound INVITE during a live call gets 486 Busy (`SIP_BUSY_REJECT`, `sip_handler._busy`).
- `virtual_numbers.py` — `VirtualNumberRegistry`: ephemeral single-use inbound extensions (`POST /virtual-numbers`). The dialed To-URI is captured in `sip_handler.onIncomingCall` (`CallInfo.local_uri`) and matched in `_on_call_received`; a matched call gets the entry's `purpose` injected into the system prompt (+ optional custom greeting), and on call end the outcome/transcript is webhooked and the number consumed. TTL-swept, persisted to `data/virtual_numbers.json`, gated by `VIRTUAL_NUMBERS_ENABLED`.
- `retry_utils.py` — retry/backoff helpers for the external API calls (attempts/delays come from `API_RETRY_*` config).
- `telemetry.py` — OpenTelemetry init + a `Metrics` helper used pervasively for Prometheus metrics. `logging_utils.py` — structured JSON logging via `log_event`.

### Tool loading

1. **Built-ins** — `tool_manager.py` hard-codes the built-in tool classes in `_load_tools()` and wraps each in a `PluginToolWrapper`. **To add a built-in tool, import it and add it to the `tool_classes` list there.** Enablement is gated by config flags in `_should_enable_tool` (e.g. `ENABLE_TIMER_TOOL`, `ENABLE_WEATHER_TOOL`).
2. **Auto-discovery** — after the built-ins, `_discover_extra_plugins()` uses `tool_plugins.PluginLoader` to scan the `plugins/` directories **plus `data/plugins/`** (the mounted data volume), so deployments can drop a tool file in without rebuilding the image. Gated by `ENABLE_PLUGIN_AUTODISCOVERY` (default true); discovered tools never override built-ins.

Built-in tools live in `sip-agent/src/plugins/`. Each subclasses `BaseTool` and implements `async def execute(self, params) -> ToolResult`:

- **Core**: weather (NWS current conditions; needs `WEATHER_LATITUDE`/`WEATHER_LONGITUDE`), timer, callback, hangup, status, cancel, datetime, calc, joke, simon_says, knowledge (RAG)
- **Fun**: `random_tools.py` (DICE/COIN), `trivia_tool.py` (game state on `session.tool_state`), `story_tool.py` (one-shot LLM via `llm_engine.summarize_text`), `drink_tool.py` (DRINK_RECIPE — TheCocktailDB)
- **Information**: `web_search_tool.py` (SearxNG), `nws_weather_tool.py` (FORECAST — api.weather.gov, needs a descriptive User-Agent), `space_weather_tool.py` (KP_INDEX — NOAA SWPC), `quake_tool.py` (QUAKES — USGS feeds; `sort=recent|biggest|nearest`, result always names `most_recent`/`largest`, data capped at 10). Shared plumbing (JSON fetch, number-to-words, spoken time-ago, home-coordinates gate) lives in `plugins/helpers.py`.
- **Memory + automation**: `memory_tools.py` (REMEMBER/FORGET over `CallerMemoryStore.add_fact`/`remove_facts`), `workflow_tool.py` (TRIGGER_WORKFLOW — fires webhooks named in `data/workflows.json` via `deliver_webhook`)
- **Ops**: `gpu_status_tool.py` + `alerts_tool.py` (Prometheus/Alertmanager queries), `container_tool.py` (CONTAINER_CTL — docker Engine API over the socket, allowlist-gated, restart requires `confirm=true`)
- **Telephony**: `transfer_tool.py` (TRANSFER — blind REFER via `SIPHandler.transfer_call`, which marshals `Call.xfer` onto the PJSIP thread through `_queue_command`)

`BaseTool.speak_result = True` marks informational tools whose result message is spoken verbatim in text-marker mode (`llm_engine._apply_marker_tools`); in native/langgraph mode `_fold_unspoken_results` prepends the message when the model failed to relay it. Tools needing per-call state must use `session.tool_state[...]` — tool instances are singletons across calls.

### REST API endpoints (`api.py`)

`/health` (add `?deep=true` to probe vLLM/Speaches/Redis), `/queue`, `POST /call`, `GET /call/{id}`, `GET /call/{id}/transcript`, `/tools`, `/tools/{name}`, `POST /tools/{name}/call`, `POST /tools/{name}/execute`, `POST /webhook/call`, `POST /speak`, `POST /play` (raw audio body played into the active call), the `/schedule` CRUD set (`POST` / `GET` / `GET {id}` / `DELETE {id}`), and the `/virtual-numbers` CRUD set (ephemeral single-use inbound extensions; see `virtual_numbers.py`). Outbound-call requests support an optional `choice` prompt that collects a spoken response — or a DTMF keypress (`ChoiceOption.dtmf`, default 1-based position) — and POSTs it to a `callback_url`. Outgoing webhooks are SSRF-pinned, retried with backoff, and HMAC-signed when `WEBHOOK_SIGNING_SECRET` is set (`deliver_webhook`). Mutating endpoints support bearer/X-API-Key auth (`API_AUTH_TOKEN`) and token-bucket rate limiting (`RATE_LIMIT_RPM`).

## Common commands

```bash
# --- Run the stack (from repo root) ---
cp sip-agent/.env.example sip-agent/.env     # then edit SIP_*, LLM_MODEL, etc.
docker compose up -d                                                              # base stack
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d   # + Grafana/Prometheus/Loki/Tempo
docker compose -f docker-compose.dgx.yml up -d                                   # DGX Spark variant (its own self-contained compose)

# Health check
curl http://localhost:8080/health | jq

# Logs
docker logs -f sip-agent
python tools/view-logs.py -f          # formatted/structured log viewer
./tools/sipshark.sh                   # termshark wrapper for SIP/RTP capture (add -r for RTP)

# Run locally (outside Docker) — requires pjsua2 + reachable Speaches/vLLM
cd sip-agent && pip install -r requirements.txt && python src/main.py

# --- SIP agent tests (layered: unit / component / e2e) ---
cd sip-agent
pip install -r requirements.txt -r requirements-test.txt
pytest tests/unit tests/component -q        # fast tiers (no GPU, no docker) — run in CI (.github/workflows/tests.yml)
pytest -m e2e                               # real docker-compose stack + SIP calls (DGX only, manual)
ruff check --select E9,F63,F7,F82 src tests # error-level lint gate used by CI

# --- speaches submodule (its own tooling; follow speaches/CLAUDE.md) ---
cd speaches   # ruff format/check, pyright, pytest
```

## Conventions & constraints

- **Configuration is entirely env-var driven** through `config.py`. Add new settings there as `field(default_factory=lambda: os.getenv(...))` and surface them in the compose files + `.env.example`. Don't hardcode values in component modules.
- The agent targets Python 3.11 and uses `List`/`Dict` typing. The `speaches` submodule mandates modern `list`/`dict` typing and other rules in its own `CLAUDE.md` — apply those only inside `speaches/`.
- TTS for common phrases (greetings, acknowledgments, etc.) is pre-cached at startup from `config.phrases` (see `PhrasesConfig.get_all_phrases_for_cache`); new fixed phrases should flow through that path for instant playback. Phrases can be overridden via `PHRASES_*` env vars (JSON array or comma-separated) or a `data/phrases.json` file.
- Commit style uses emoji-prefixed conventional commits (e.g. `✨ feat:`, `fix:`).
- `docker-compose.dgx.yml` is a separate, self-contained compose for the DGX Spark / GB10 target (recently split out) — keep it in sync with the base compose when changing service wiring.
- `examples/n8n-nodes-general-disarray/` is a custom n8n node package for the agent's REST API. Its `dist/` is bind-mounted into the n8n container (`N8N_CUSTOM_EXTENSIONS=/custom-nodes`) — run `examples/n8n-nodes-general-disarray/build.sh` before recreating the n8n container, and keep the n8n service blocks in both compose files in lockstep.
