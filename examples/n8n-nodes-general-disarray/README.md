# n8n-nodes-general-disarray

Custom [n8n](https://n8n.io) nodes for the **General Disarray** SIP AI phone assistant.

The package ships two nodes plus one credential type:

- **SIP Agent** (action node) — drives the agent's REST API on `:8080`: make outbound calls, check call status, fetch transcripts, speak into the active call, run tools, schedule calls, and probe system health/queue state.
- **SIP Agent Trigger** — a webhook trigger that receives the agent's **call-lifecycle events** (`call.started` / `call.ended`, see below) and **choice-callback** POSTs (the JSON the agent sends to your `callback_url` when a call with a choice prompt completes), with an event filter and optional HMAC-SHA256 signature verification.
- **SIP Agent API** credential — base URL, optional API token, and the webhook signing secret.

## Build

No local Node.js toolchain required — the build runs inside a `node:20-alpine` container and works on arm64 (DGX Spark / GB10):

```bash
cd examples/n8n-nodes-general-disarray
./build.sh
```

Requires only Docker. Compiled output lands in `dist/` (owned by your host user), which is what gets mounted into n8n. Re-run `./build.sh` after any source change.

## Install into this stack

The compose files are **already wired**: both `docker-compose.yml` and `docker-compose.dgx.yml` set `N8N_CUSTOM_EXTENSIONS=/custom-nodes` on the n8n service and mount `./examples/n8n-nodes-general-disarray/dist` read-only at `/custom-nodes/n8n-nodes-general-disarray`. (Only `dist/` is mounted on purpose — mounting the package root would let n8n's `**/*.node.js` glob sweep `node_modules`.)

So installation is just: build, then (re)create the n8n container so it picks the nodes up.

```bash
cd examples/n8n-nodes-general-disarray && ./build.sh && cd ../..

# DGX Spark stack:
docker compose -f docker-compose.dgx.yml up -d n8n

# Base stack — identical wiring, same command shape:
docker compose up -d n8n
```

The two nodes then appear in the n8n editor as **SIP Agent** and **SIP Agent Trigger**.

## Credential setup

In n8n, create a **SIP Agent API** credential:

| Field | Value | Notes |
|---|---|---|
| Base URL | `http://sip-agent:8080` | Container-internal hostname; correct for n8n running in the same compose network. |
| API Token | value of `API_AUTH_TOKEN` from `.env` | Leave empty if the agent runs without auth. |
| Auth Header Style | `X-API-Key` (default) or `Authorization: Bearer` | The agent accepts both. |
| Webhook Signing Secret | value of `WEBHOOK_SIGNING_SECRET` from `.env` | Used **only** by the SIP Agent Trigger to verify incoming callback signatures. |

> **Note:** the credential's built-in test calls `GET /health`, which is unauthenticated. A passing test confirms connectivity and the base URL, **not** that the API token is correct — token errors only surface on the first mutating request (e.g. `POST /call` returning 401).

## Resources & Operations

| Resource | Operation | Endpoint | Notes |
|---|---|---|---|
| Call | Make | `POST /call` | Message + extension; optional choice prompt, callback URL, ring timeout, custom call ID. |
| Call | Get Status | `GET /call/{id}` | API always returns 200; `status` may be `"not_found"`. Optional *Error on Not Found* toggle turns that into a node error. |
| Call | Get Transcript | `GET /call/{id}/transcript` | HTTP 404 if the transcript doesn't exist. |
| Speak | Say | `POST /speak` | Speaks into the currently active call (query params: `message`, optional `call_id` guard). |
| Tool | Get Many | `GET /tools` | One output item per tool. |
| Tool | Get | `GET /tools/{name}` | Tool names are uppercase (e.g. `WEATHER`); the server uppercases anyway. |
| Tool | Execute | `POST /tools/{name}/execute` | JSON params; optional *Speak Result* into the active call. |
| Tool | Execute and Call | `POST /tools/{name}/call` | Runs the tool, then calls out speaking prefix + result + suffix; supports choice prompts. A `"tool_failed"` status is passed through, not thrown. |
| Schedule | Create | `POST /schedule` | Message **or** tool content; delay **or** at-time; optional timezone, recurrence (daily/weekdays/weekends/cron), callback URL. |
| Schedule | Get Many | `GET /schedule` | One output item per scheduled call. |
| Schedule | Get | `GET /schedule/{id}` | 404 if unknown. |
| Schedule | Delete | `DELETE /schedule/{id}` | 404 if unknown. |
| System | Health | `GET /health` | Optional *Deep* toggle adds `?deep=true` (probes vLLM/Speaches/Redis). |
| System | Get Queue | `GET /queue` | Outbound call queue status. |

## Reformat for Speech

Call:Make, Tool:Execute and Call, Schedule:Create, and Speak:Say expose a **Reformat for Speech** toggle (`reformat_for_speech` in the API). When on, the agent's own LLM rewrites the message into natural spoken form before it is voiced — `ALERT: svc-api p99=340ms @ 2026-07-08T17:03Z` becomes something like "Alert: the API service's ninety-ninth percentile latency is three hundred forty milliseconds, as of July eighth at five oh three PM" — preserving every fact (numbers, IDs, dates, statuses) rather than stripping them. Adds roughly one to two seconds of latency (one local LLM round-trip) before dialing; for schedules the rewrite happens at call time so tool-generated content is covered. Fail-open: on any LLM failure or timeout the original text is spoken unchanged.

## Choice prompt + callback walkthrough

This is the flagship flow: call someone, ask them a question, branch a workflow on their answer.

> **⚠️ You MUST set `WEBHOOK_ALLOW_PRIVATE=true` in `.env` and restart sip-agent** (`docker compose up -d sip-agent`). The callback URL `http://n8n:5678/...` resolves to a private docker bridge-network address, and the agent's SSRF guard rejects private destinations by default — your `POST /call` will fail with **HTTP 400** ("callback_url resolves to a private address") until you enable it.

1. **Add a SIP Agent Trigger** node to a workflow and copy its **production** webhook URL. Because n8n runs behind `WEBHOOK_URL=http://n8n:5678/` in this stack, the URL looks like `http://n8n:5678/webhook/<uuid>/webhook`. That hostname is **container-internal**: it is exactly what the sip-agent container needs to deliver the callback across the compose network, but it is **not browsable from your workstation** — don't be surprised when it doesn't open in a browser. (For manual "Listen for test event" runs, the test URL follows the same host; deliveries still work because both containers share the network.)
2. **Activate the workflow** (production webhooks are only registered while the workflow is active).
3. **Make the call** with a SIP Agent node — resource *Call*, operation *Make*:
   - **Message**: `Hi, this is the assistant confirming your appointment tomorrow at 3 PM.`
   - **Extension**: the number/extension to dial.
   - **Callback URL**: the trigger's production URL from step 1.
   - **Choice** → prompt: `Do you confirm? Say yes or no, or press 1 for yes, 2 for no.` with options `yes` (synonyms: `yeah, confirm, sure`, DTMF 1) and `no` (synonyms: `nope, cancel`, DTMF 2).
4. **When the call completes**, the agent POSTs JSON to the trigger:

   ```json
   {
     "call_id": "…",
     "status": "completed",
     "extension": "1001",
     "duration_seconds": 24.7,
     "message_played": true,
     "choice_response": "yes",
     "choice_raw_text": "yeah sure",
     "machine_answered": false
   }
   ```

   `choice_response` is the matched option value (absent if nothing matched); branch on it with an IF node.

An importable example workflow (Trigger → IF on `{{$json.choice_response}}` → Confirmed / Declined) is in [`examples/choice-callback-workflow.json`](examples/choice-callback-workflow.json) — in n8n use *Import from File*.

## Call lifecycle events (trigger on any call)

The agent can push signed `call.started` / `call.ended` events for **every** call — inbound or outbound, no per-call `callback_url` needed — so a workflow can trigger whenever the assistant is on a call.

> **⚠️ Same SSRF caveat as above:** the n8n trigger URL is a private address, so `WEBHOOK_ALLOW_PRIVATE=true` must be set in `.env`.

1. **Add a SIP Agent Trigger** node, pick the events you want under **Events** (*Call Started*, *Call Ended*, and/or *Choice Result / Call Outcome*), activate the workflow, and copy its production URL.
2. In `.env`, set:

   ```bash
   CALL_EVENT_WEBHOOK_URL=http://n8n:5678/webhook/<uuid>/webhook
   CALL_EVENTS=call.started,call.ended        # or a subset
   CALL_EVENT_INCLUDE_TRANSCRIPT=true         # embed the transcript in call.ended
   WEBHOOK_ALLOW_PRIVATE=true
   ```

   then recreate the agent: `docker compose -f docker-compose.dgx.yml up -d sip-agent`.
3. The agent now POSTs to the trigger on every call:

   ```json
   {
     "event": "call.started",
     "call_id": "in-1751970000-3",
     "sip_call_id": "4",
     "direction": "inbound",
     "remote_uri": "sip:1001@pbx",
     "started_at": "2026-07-08T17:00:00+00:00",
     "timestamp": "2026-07-08T17:00:00+00:00"
   }
   ```

   `call.ended` adds `duration_seconds` and (unless `CALL_EVENT_INCLUDE_TRANSCRIPT=false`) the full `transcript` record — `{call_id, direction, remote_uri, started_at, ended_at, turns: [{role, content, ts}]}` — so an "after every call" workflow can summarize, archive, or alert on the conversation without an extra API round-trip.

Notes:

- The event feed is signed with `WEBHOOK_SIGNING_SECRET` exactly like the choice callbacks, so HMAC verification (next section) applies unchanged.
- Legacy choice/outcome callbacks carry **no `event` field**; the trigger classifies them as *Choice Result / Call Outcome*. Existing workflows keep working (the Events default selects everything).
- Deliveries for deselected events are acknowledged with 200 (no retries on the agent side) but start no execution.
- One agent URL feeds one trigger; to handle multiple event types differently in a single workflow, select several events and branch on `{{$json.event}}` with a Switch node.

## Verifying webhook signatures (HMAC)

To authenticate callbacks end-to-end:

1. Set `WEBHOOK_SIGNING_SECRET=<random secret>` in `.env` and restart sip-agent. The agent then signs every outgoing webhook with `X-Timestamp` and `X-Signature: sha256=<hex>` headers, where the digest is HMAC-SHA256 over `"<timestamp>." + raw body`.
2. Put the **same secret** in the *Webhook Signing Secret* field of your SIP Agent API credential, and attach that credential to the **SIP Agent Trigger** node.
3. Enable **Require Signature** on the trigger. Unsigned, stale (older than *Tolerance (Seconds)*, default 300), or tampered requests are rejected with 401 before your workflow runs.

If *Require Signature* is off but a secret is configured and a request carries an `X-Signature` header, the trigger still verifies it opportunistically — bad signatures are rejected either way.

## Note on node type names (custom dir vs. npm package)

Nodes loaded from `N8N_CUSTOM_EXTENSIONS` get the **`CUSTOM.`** package prefix, so saved workflows reference them as `CUSTOM.sipAgent` / `CUSTOM.sipAgentTrigger`. If you later publish this package to npm and install it as a community package instead, the type strings change (to `n8n-nodes-general-disarray.sipAgent` etc.) and existing workflows will show the nodes as unrecognized until you re-add them or edit the workflow JSON. Keep that in mind before building a large library of workflows on the custom-directory install.

## License

AGPL-3.0, same as the rest of General Disarray.
