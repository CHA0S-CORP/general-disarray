"""Alert bridge: turns Alertmanager webhooks into phone calls via the SIP agent.

Receives Prometheus Alertmanager (or Grafana) webhook notifications on /alert,
places a call to the primary on-call number through the agent's POST /call API
with a spoken-acknowledgment choice prompt, and escalates to the secondary
number when the call is not acknowledged.

Configuration (env vars):
    AGENT_API_URL           Base URL of the sip-agent API (default http://sip-agent:8080)
    AGENT_API_TOKEN         Bearer token if the agent has API_AUTH_TOKEN set
    ONCALL_PRIMARY          Extension/number to call first (required)
    ONCALL_SECONDARY        Extension/number to escalate to (optional)
    BRIDGE_CALLBACK_URL     URL the agent should POST call results to
                            (default http://alert-bridge:8000/ack)
    ACK_TIMEOUT_S           Seconds the callee has to say "acknowledge" (default 30)
    CALL_ON_RESOLVED        Also call when an alert resolves (default false)
    WEBHOOK_SIGNING_SECRET  If set, /ack requires a valid X-Signature header
                            (must match the agent's WEBHOOK_SIGNING_SECRET)
    LOG_LEVEL               Logging level (default INFO)
"""
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("alert_bridge")

AGENT_API_URL = os.getenv("AGENT_API_URL", "http://sip-agent:8080").rstrip("/")
AGENT_API_TOKEN = os.getenv("AGENT_API_TOKEN", "")
ONCALL_PRIMARY = os.getenv("ONCALL_PRIMARY", "")
ONCALL_SECONDARY = os.getenv("ONCALL_SECONDARY", "")
BRIDGE_CALLBACK_URL = os.getenv("BRIDGE_CALLBACK_URL", "http://alert-bridge:8000/ack")
ACK_TIMEOUT_S = int(os.getenv("ACK_TIMEOUT_S", "30"))
CALL_ON_RESOLVED = os.getenv("CALL_ON_RESOLVED", "false").lower() == "true"
WEBHOOK_SIGNING_SECRET = os.getenv("WEBHOOK_SIGNING_SECRET", "")
# Reject signed callbacks older than this many seconds (replay defense).
MAX_SIGNATURE_AGE_S = 300

app = FastAPI(title="SIP Alert Bridge")

# call_id -> alert context, so /ack knows what to escalate. In-memory on
# purpose: an alert bridge that lost state will simply re-call on the next
# Alertmanager repeat_interval.
_pending_calls: Dict[str, Dict[str, Any]] = {}

ACK_CHOICE = {
    "prompt": "Say acknowledge to confirm you received this alert.",
    "options": [
        {
            "value": "acknowledged",
            "synonyms": ["acknowledge", "ack", "got it", "on it", "confirmed"],
        }
    ],
    "timeout_seconds": ACK_TIMEOUT_S,
}


def _agent_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if AGENT_API_TOKEN:
        headers["Authorization"] = f"Bearer {AGENT_API_TOKEN}"
    return headers


def _format_message(alert: Dict[str, Any]) -> str:
    labels = alert.get("labels", {})
    annotations = alert.get("annotations", {})
    name = labels.get("alertname", "unknown alert")
    severity = labels.get("severity", "warning")
    description = (
        annotations.get("description")
        or annotations.get("summary")
        or "No description provided."
    )
    if alert.get("status") == "resolved":
        return f"Resolved: {name}. {description}"
    return f"Alert! {name}. Severity: {severity}. {description}"


async def _place_call(extension: str, message: str, context: Dict[str, Any]) -> str:
    """Ask the agent to place a call; returns the agent's call_id."""
    payload = {
        "extension": extension,
        "message": message,
        "callback_url": BRIDGE_CALLBACK_URL,
        "choice": ACK_CHOICE,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{AGENT_API_URL}/call", json=payload, headers=_agent_headers()
        )
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"agent rejected call ({resp.status_code}): {resp.text[:200]}",
        )
    call_id = resp.json().get("call_id", "")
    _pending_calls[call_id] = context
    logger.info("Placed call %s to %s: %s", call_id, extension, message)
    return call_id


def _verify_signature(request_body: bytes, timestamp: str, signature: str) -> bool:
    """Verify the agent's X-Signature over '<timestamp>.<body>'."""
    if not timestamp or not signature.startswith("sha256="):
        return False
    try:
        age = abs(time.time() - int(timestamp))
    except ValueError:
        return False
    if age > MAX_SIGNATURE_AGE_S:
        return False
    mac = hmac.new(
        WEBHOOK_SIGNING_SECRET.encode(),
        f"{timestamp}.".encode() + request_body,
        hashlib.sha256,
    )
    return hmac.compare_digest(signature, f"sha256={mac.hexdigest()}")


@app.get("/health")
async def health() -> Dict[str, Any]:
    agent = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{AGENT_API_URL}/health")
        agent = "up" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        agent = f"down ({type(e).__name__})"
    return {"status": "healthy", "agent": agent, "pending_calls": len(_pending_calls)}


@app.post("/alert")
async def handle_alert(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Alertmanager webhook receiver.

    Accepts the standard Alertmanager payload ({"alerts": [...]}) and, for
    convenience, a bare single-alert dict ({"labels": ..., "annotations": ...}).
    """
    if not ONCALL_PRIMARY:
        raise HTTPException(status_code=503, detail="ONCALL_PRIMARY is not configured")

    alerts = payload.get("alerts")
    if alerts is None:
        alerts = [payload]  # single-alert convenience form

    placed = []
    skipped = 0
    for alert in alerts:
        status = alert.get("status") or payload.get("status") or "firing"
        alert = {**alert, "status": status}
        if status == "resolved" and not CALL_ON_RESOLVED:
            skipped += 1
            continue
        message = _format_message(alert)
        call_id = await _place_call(
            ONCALL_PRIMARY,
            message,
            {"message": message, "escalated": False, "alert": alert},
        )
        placed.append(call_id)

    return {"status": "calling" if placed else "no_calls", "call_ids": placed,
            "skipped_resolved": skipped}


@app.post("/ack")
async def handle_ack(request: Request) -> Dict[str, Any]:
    """Call-result webhook from the agent; escalates unacknowledged alerts."""
    body = await request.body()
    if WEBHOOK_SIGNING_SECRET:
        if not _verify_signature(
            body,
            request.headers.get("X-Timestamp", ""),
            request.headers.get("X-Signature", ""),
        ):
            raise HTTPException(status_code=401, detail="invalid webhook signature")

    payload = await request.json()
    call_id = payload.get("call_id", "")
    context = _pending_calls.pop(call_id, None)
    choice = payload.get("choice_response")

    if choice == "acknowledged":
        logger.info("Alert acknowledged on call %s", call_id)
        return {"status": "acknowledged"}

    logger.warning(
        "Call %s not acknowledged (status=%s choice=%r machine=%s)",
        call_id, payload.get("status"), choice, payload.get("machine_answered"),
    )
    if context is None or context.get("escalated") or not ONCALL_SECONDARY:
        return {"status": "unacknowledged", "escalated": False}

    escalation_id = await _place_call(
        ONCALL_SECONDARY,
        f"Escalation, primary on-call did not acknowledge. {context['message']}",
        {**context, "escalated": True},
    )
    logger.warning("Escalated call %s -> %s to secondary", call_id, escalation_id)
    return {"status": "unacknowledged", "escalated": True, "call_id": escalation_id}
