# Alert Bridge

Turns Prometheus Alertmanager (or Grafana) webhook notifications into phone
calls through the SIP agent: the on-call engineer gets a call describing the
alert and is asked to say **"acknowledge"**. If the call is not acknowledged
(no answer, timeout, or answering machine), the bridge escalates to the
secondary on-call number.

```
Alertmanager ──POST /alert──▶ alert-bridge ──POST /call──▶ sip-agent ──📞──▶ on-call
                                   ▲                                          │
                                   └────────── POST /ack (webhook) ◀──────────┘
```

## Run

The main stack must be up first (the bridge joins its docker network):

```bash
# from the repo root
docker compose up -d                     # or docker-compose.dgx.yml
ONCALL_PRIMARY=5551234567 ONCALL_SECONDARY=5559876543 \
  docker compose -f examples/alert-bridge/docker-compose.yml up -d --build
```

The agent must allow webhooks to the bridge's private address:
`WEBHOOK_ALLOW_PRIVATE=true` on the sip-agent (or expose the bridge publicly
and set `BRIDGE_CALLBACK_URL`).

## Point Alertmanager at it

```yaml
receivers:
  - name: 'phone-alert'
    webhook_configs:
      - url: 'http://alert-bridge:8000/alert'
        send_resolved: true
```

## Test it by hand

```bash
curl -s -X POST http://localhost:8100/alert -H 'Content-Type: application/json' -d '{
  "status": "firing",
  "alerts": [{
    "status": "firing",
    "labels": {"alertname": "DatabaseDown", "severity": "critical"},
    "annotations": {"description": "The production database is not responding."}
  }]
}' | jq
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `AGENT_API_URL` | `http://sip-agent:8080` | sip-agent API base URL |
| `AGENT_API_TOKEN` | – | Bearer token when the agent sets `API_AUTH_TOKEN` |
| `ONCALL_PRIMARY` | – (required) | Number/extension called first |
| `ONCALL_SECONDARY` | – | Escalation target when unacknowledged |
| `BRIDGE_CALLBACK_URL` | `http://alert-bridge:8000/ack` | Where the agent POSTs call results |
| `ACK_TIMEOUT_S` | `30` | Seconds to wait for a spoken acknowledgment |
| `CALL_ON_RESOLVED` | `false` | Also call when alerts resolve |
| `WEBHOOK_SIGNING_SECRET` | – | Verify the agent's `X-Signature` HMAC on `/ack` (set the same value on the agent) |

Endpoints: `POST /alert` (Alertmanager payload or a bare single-alert dict),
`POST /ack` (agent webhook; HMAC-verified when the secret is set), `GET /health`
(also probes the agent).
