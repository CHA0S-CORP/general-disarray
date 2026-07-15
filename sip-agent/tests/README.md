# SIP agent test suite

A layered suite for the SIP AI agent. Two tiers, selected with pytest markers.

| Tier | Marker | Runs where | What it covers |
|------|--------|-----------|----------------|
| Unit | `unit` | anywhere (no Docker/GPU) | pure logic: CALC, tool-call protocol parsing, param coercion, VAD end-of-utterance, config, choice matching, SSRF/extension validation, pure tools |
| Component | `component` | anywhere (no Docker/GPU) | real agent code against **in-process mock Speaches/vLLM** + `fakeredis`: REST API (TestClient), audio pipeline STT/TTS, LLM engine tool round-trip, tool manager, call queue |
| E2E | `e2e` | **DGX / GPU only** | the real `docker-compose.dgx.yml` stack + **real SIP calls** via a `pjsua` softphone |

## Setup

```bash
cd sip-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
```

## Tier 1 — unit + component (the everyday gate)

Fast, deterministic, no Docker or GPU. This is what you run on every change.

```bash
pytest -m "unit or component"        # ~115 tests, a couple seconds
pytest -m unit                       # units only
pytest -m component                  # component only
```

The component tier starts two tiny FastAPI mock servers (`tests/component/mocks/`)
on ephemeral localhost ports and points the agent's real HTTP/OpenAI clients at
them — no model weights, no network.

## Tier 2 — real-stack e2e (DGX)

Brings up the full stack and places actual phone calls. Run on the DGX.

```bash
# Cold: bring the stack up, wait for models (minutes), run, tear down.
pytest -m e2e

# Warm loop (recommended during iteration): start the stack once yourself...
docker compose -f docker-compose.dgx.yml up -d
# ...then reuse it (no up/down, no model reload):
E2E_USE_RUNNING=1 pytest -m e2e
```

What it does:
- A containerized **`pjsua` softphone** (built from `tests/e2e/Dockerfile.softphone`)
  dials the agent, streams a silence-padded WAV question over RTP, and records the
  reply.
- Question WAVs are generated lazily from the live stack's Speaches TTS
  (`tests/e2e/audio/gen_audio.py`) — nothing binary is committed.
- Assertions are layered: **(0)** the captured audio is non-silent, **(1)** the
  agent's structured JSON log shows the pipeline stages firing (the primary,
  deterministic gate — `sip_incoming_call → user_speech → assistant_response`),
  **(2)** the deterministic phrase (`SIMON_SAYS` / `CALC`) appears in the reply
  text and/or the transcribed audio.

### E2E environment knobs

| Var | Default | Purpose |
|-----|---------|---------|
| `E2E_USE_RUNNING` | `0` | `1` = run against an already-up stack (skip up/down) |
| `E2E_KEEP_STACK` | `0` | `1` = bring up once, skip teardown |
| `E2E_COMPOSE_FILE` | `<repo>/docker-compose.dgx.yml` | compose file to use |
| `E2E_NETWORK` | `general-disarray_default` | compose network the softphone joins |
| `E2E_AGENT_API` | `http://localhost:8080` | agent REST API base |
| `E2E_SPEACHES` | `http://localhost:8001` | Speaches base (for fixture gen + Layer 2) |
| `E2E_SIP_TARGET` | `sip:ai-assistant@sip-agent:5060` | inbound dial target |
| `E2E_SOFTPHONE_IMAGE` | _(build locally)_ | use a prebuilt softphone image |

The **outbound** test (`test_outbound_call.py`) additionally needs the stack
started with `OUTBOUND_ALLOW_SIP_URI=true` so the agent will dial the softphone's
`sip:` URI; it skips with guidance otherwise.

## Sanity-checking the harness

To confirm the e2e assertions actually catch regressions (not pass vacuously),
break a stage and watch a test fail at the right point, e.g.:

```bash
docker compose -f docker-compose.dgx.yml stop vllm
E2E_USE_RUNNING=1 pytest -m e2e -k inbound   # fails at the assistant_response assertion
```
