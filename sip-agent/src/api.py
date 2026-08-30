"""
Outbound Call API
=================
REST API for initiating outbound notification calls with optional response collection.
"""

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import socket
import time
from urllib.parse import urlparse
import httpx
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from admin_events import EventBus
from call_session import set_current_session
from identity_verification import is_safe_caller_id
from dtmf_collect import collect_dtmf_code

from telemetry import create_span, Metrics
from logging_utils import log_event
from retry_utils import retry_async, RetryError

# Reserved caller id under which GET /verify/otp serves the GLOBAL TOTP secret.
GLOBAL_OTP_ID = "global"

if TYPE_CHECKING:
    from main import SIPAIAssistant
    from call_queue import CallQueue

logger = logging.getLogger(__name__)


# ============================================================================
# Request validation / security helpers
# ============================================================================

class _VerifyCallDone(Exception):
    """Internal signal inside run_verify_call: a terminal non-answer outcome
    (no answer / initiate failure) with status already set — jump to the shared
    teardown + webhook exit rather than running the prompt/verify loop."""


class RequestRejected(HTTPException):
    """Raised when a request is rejected at the handler boundary.

    Carries an HTTP status code (400 for bad input, 409 for conflicts, 429
    for backpressure) instead of a generic 500. Subclasses HTTPException so
    FastAPI surfaces it natively, while non-HTTP callers (the voice CALLBACK
    path, the scheduler) can still catch it by name.
    """

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


def make_auth_dependency(token: str):
    """Build a FastAPI dependency enforcing a shared secret.

    If ``token`` is empty, auth is disabled (the dependency is a no-op) so
    existing dev setups keep working. When set, callers must present the token
    via ``Authorization: Bearer <token>`` or ``X-API-Key: <token>``.
    """

    async def _verify(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
    ):
        if not token:
            return
        provided = x_api_key
        if not provided and authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer":
                provided = value.strip()
        if not provided or provided != token:
            raise HTTPException(status_code=401, detail="Invalid or missing API credentials")

    return _verify


async def csrf_protect(request: Request):
    """Reject cross-site browser requests to state-changing endpoints.

    In the (supported) tokenless localhost mode a bodyless, header-free POST
    is a CORS-"simple" request: any web page the operator's browser visits can
    fire it without a preflight (drive-by CSRF). Browsers attach fetch
    metadata (``Sec-Fetch-Site``) and/or an ``Origin`` header to such
    requests, so we reject anything that self-identifies as cross-site.
    Non-browser clients (curl, n8n, scripts) send neither header and pass
    through untouched, as do same-origin requests from the admin page.
    """
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site:
        # Modern browsers: trust the fetch metadata outright ("none" is a
        # user-initiated navigation; anything not same-origin is rejected).
        if fetch_site in ("same-origin", "none"):
            return
        raise HTTPException(status_code=403,
                            detail="Cross-site browser requests are not allowed")
    origin = request.headers.get("origin")
    if origin:
        # Older browsers without fetch metadata: compare Origin to Host.
        host = request.headers.get("host", "")
        origin_host = urlparse(origin).netloc
        if not origin_host or not host or origin_host.lower() != host.lower():
            raise HTTPException(status_code=403,
                                detail="Cross-origin browser requests are not allowed")


class RateLimiter:
    """In-memory token bucket per client key."""

    MAX_BUCKETS = 10_000

    def __init__(self, rpm: int, burst: int):
        self.rate = rpm / 60.0
        self.burst = float(burst)
        self._buckets: Dict[str, tuple] = {}  # key -> (tokens, last_refill_ts)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        if len(self._buckets) > self.MAX_BUCKETS:
            # Drop buckets that have fully refilled — they carry no state.
            self._buckets = {
                k: (tokens, last) for k, (tokens, last) in self._buckets.items()
                if tokens + (now - last) * self.rate < self.burst
            }
        tokens, last = self._buckets.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        allowed = tokens >= 1.0
        self._buckets[key] = (tokens - 1.0 if allowed else tokens, now)
        return allowed


def make_rate_limit_dependency(config):
    """FastAPI dependency enforcing RATE_LIMIT_RPM on mutating endpoints.

    Clients are keyed by their API credential when presented, else by client
    IP. A no-op when rate limiting is disabled (RATE_LIMIT_RPM=0).
    """
    if config.rate_limit_rpm <= 0:
        async def _noop():
            return
        return _noop

    limiter = RateLimiter(config.rate_limit_rpm,
                          config.rate_limit_burst or config.rate_limit_rpm)

    async def _limit(request: Request):
        key = (request.headers.get("X-API-Key")
               or request.headers.get("Authorization")
               or (request.client.host if request.client else "unknown"))
        if not limiter.allow(key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded; try again later")

    return _limit


def _host_is_blocked(host: str) -> bool:
    """Return True if a hostname resolves to a non-public address.

    Boolean wrapper around _resolve_allowed_ips so the pre-validation path and
    the send-time pinning path apply exactly the same SSRF policy.
    """
    try:
        _resolve_allowed_ips(host)
        return False
    except ValueError:
        return True


def _resolve_allowed_ips(host: str) -> List[str]:
    """Resolve a host and return its IPs only if every one is public.

    Raises ValueError if the host is unresolvable or any resolved address falls
    in a loopback/private/link-local/reserved/multicast/unspecified range. The
    all-or-nothing check defends against DNS-rebinding: a name that resolves to
    even one internal address is rejected outright.
    """
    addrs: List[str] = []
    try:
        addrs.append(str(ipaddress.ip_address(host)))
    except ValueError:
        try:
            for _family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
                addrs.append(sockaddr[0])
        except (socket.gaierror, ValueError) as e:
            raise ValueError(f"unresolvable host: {host}") from e
    if not addrs:
        raise ValueError(f"no addresses for host: {host}")
    validated: List[str] = []
    for addr in addrs:
        # Strip any IPv6 scope id (e.g. 'fe80::1%eth0') before parsing.
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
        if (ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"host {host} resolves to disallowed address {addr}")
        validated.append(str(ip))
    return validated


async def pin_webhook_target(url: str, config):
    """Re-resolve a webhook URL at send time and pin it to a validated public IP.

    Returns ``(request_url, headers, extensions)`` ready to hand to ``httpx``:
    the host is replaced with a validated public IP while the original Host
    header and TLS SNI are preserved, closing the check-to-use
    (DNS-rebinding / TOCTOU) gap. When ``webhook_allow_private`` is set the URL
    is returned unchanged. Raises ``ValueError`` if the host is unresolvable or
    resolves to any non-public address.

    Shared by the REST callback path and the scheduled-call webhook so both
    enforce identical SSRF protection at the moment the request is sent.
    """
    headers = {"Content-Type": "application/json"}
    extensions = None
    request_url = url
    if not config.webhook_allow_private:
        parsed = httpx.URL(url)
        host = parsed.host
        validated_ips = await asyncio.get_event_loop().run_in_executor(
            None, _resolve_allowed_ips, host)
        # getaddrinfo may sort AAAA records first; prefer an IPv4 address so
        # dual-stack targets still work from hosts without IPv6 egress.
        ipv4 = [ip for ip in validated_ips if ":" not in ip]
        request_url = parsed.copy_with(host=(ipv4[0] if ipv4 else validated_ips[0]))
        # Preserve the original authority for the Host header and TLS SNI so the
        # request still reaches the intended (public) target after IP pinning.
        host_authority = parsed.netloc.decode("ascii")
        if "@" in host_authority:
            host_authority = host_authority.rsplit("@", 1)[1]
        headers["Host"] = host_authority
        extensions = {"sni_hostname": host}
    return request_url, headers, extensions


class _WebhookRejected(Exception):
    """A webhook POST got a 4xx response — retrying won't help."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


async def deliver_webhook(url: str, payload: Dict[str, Any], config,
                          api_name: str = "webhook") -> bool:
    """Deliver a webhook POST with SSRF pinning, HMAC signing and retries.

    - Re-resolves and IP-pins the target at send time (DNS-rebinding defense).
    - When WEBHOOK_SIGNING_SECRET is set, adds X-Timestamp and
      X-Signature: sha256=<HMAC(secret, "<timestamp>.<body>")> over the exact
      bytes sent, so receivers can verify authenticity and freshness.
    - Retries transport errors and 5xx responses with the API_RETRY_* backoff;
      4xx responses are not retried.

    Returns True on delivery, False otherwise; never raises. Shared by the
    REST callback path and the scheduled-call webhook so both enforce
    identical protection.
    """
    try:
        request_url, headers, extensions = await pin_webhook_target(url, config)
    except ValueError as e:
        logger.error(f"Webhook target rejected at send time for {url}: {e}")
        Metrics.record_callback_failed("ssrf_blocked")
        return False

    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    if config.webhook_signing_secret:
        ts = str(int(time.time()))
        mac = hmac.new(config.webhook_signing_secret.encode(),
                       f"{ts}.".encode() + body, hashlib.sha256)
        headers["X-Timestamp"] = ts
        headers["X-Signature"] = f"sha256={mac.hexdigest()}"

    async def _post():
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            resp = await client.post(request_url, content=body,
                                     headers=headers, extensions=extensions)
            if resp.status_code >= 500:
                resp.raise_for_status()  # retryable
            if resp.status_code >= 400:
                raise _WebhookRejected(resp.status_code)
            return resp

    try:
        await retry_async(
            _post, api_name=api_name, config=config,
            retryable_exceptions=(httpx.TransportError, httpx.HTTPStatusError))
        Metrics.record_callback_success()
        return True
    except _WebhookRejected as e:
        logger.error(f"Webhook to {url} rejected with HTTP {e.status_code}")
        Metrics.record_callback_failed(f"http_{e.status_code}")
        return False
    except RetryError as e:
        logger.error(f"Webhook to {url} failed after retries: {e}")
        Metrics.record_callback_failed(
            type(e.last_error).__name__ if e.last_error else "retry_exhausted")
        return False
    except Exception as e:
        logger.error(f"Failed to send webhook to {url}: {e}")
        Metrics.record_callback_failed(type(e).__name__)
        return False


async def validate_callback_url(url: Optional[str], config) -> None:
    """Validate a caller-supplied webhook URL, raising RequestRejected on failure.

    No-op when ``url`` is None. DNS resolution runs in a thread to avoid
    blocking the event loop.
    """
    if not url:
        return
    parsed = urlparse(url)
    allowed_schemes = {"https"} if config.webhook_require_https else {"http", "https"}
    if parsed.scheme not in allowed_schemes:
        raise RequestRejected(400, f"callback_url scheme must be one of {sorted(allowed_schemes)}")
    if not parsed.hostname:
        raise RequestRejected(400, "callback_url has no host")
    if not config.webhook_allow_private:
        blocked = await asyncio.get_event_loop().run_in_executor(
            None, _host_is_blocked, parsed.hostname)
        if blocked:
            raise RequestRejected(400, "callback_url resolves to a disallowed (private/internal) address")


def check_extension_allowed(extension: str, config) -> Optional[str]:
    """Return an error string if a dial target violates policy, else None.

    Unless ``outbound_allow_sip_uri`` is set, rejects raw ``sip:`` URIs and
    ``@domain`` parts so callers cannot route calls to arbitrary SIP domains.
    An optional regex (``outbound_extension_pattern``) further restricts it.

    Non-raising variant so non-HTTP callers (the voice CALLBACK path, the
    scheduler) can enforce the same policy the REST endpoints do.
    """
    if not extension or not extension.strip():
        return "extension is required"
    if not config.outbound_allow_sip_uri:
        if extension.startswith("sip:") or "@" in extension:
            return "extension must be a bare number/extension (raw SIP URIs are not allowed)"
    pattern = config.outbound_extension_pattern
    if pattern and not re.fullmatch(pattern, extension):
        return "extension does not match the allowed pattern"
    return None


def validate_extension(extension: str, config) -> None:
    """Validate a caller-supplied dial target, raising RequestRejected on failure."""
    error = check_extension_allowed(extension, config)
    if error:
        raise RequestRejected(400, error)


async def _maybe_reformat(assistant, text: str, flag: bool) -> str:
    """Opt-in LLM rewrite of a message into natural spoken form.

    Fail-open by construction: reformat_for_speech itself falls back to the
    original text, and assistants without an LLM engine skip the step.
    """
    engine = getattr(assistant, "llm_engine", None)
    if not flag or not text or engine is None:
        return text
    return await engine.reformat_for_speech(
        text, assistant.config.message_reformat_timeout_s)


def _compose_message(prefix: Optional[str], body: Optional[str], suffix: Optional[str]) -> str:
    """Join an optional prefix, body, and suffix into a single spoken message."""
    parts = [p.strip() for p in (prefix, body, suffix) if p and p.strip()]
    return " ".join(parts)


def tool_result_success(result) -> bool:
    """True if a ToolResult completed successfully.

    Handles both enum (ToolStatus.SUCCESS) and plain-string status values.
    """
    status = getattr(result.status, "value", result.status)
    return str(status).lower() == "success"


# ============================================================================
# API Models
# ============================================================================

class ChoiceOption(BaseModel):
    """A choice option for the user."""
    value: str = Field(..., description="The value to return if selected")
    synonyms: List[str] = Field(default_factory=list, description="Alternative phrases that map to this choice")
    dtmf: Optional[str] = Field(
        default=None, pattern=r"^[0-9*#]$",
        description="Phone key that selects this option (defaults to its 1-based position)")


class ChoicePrompt(BaseModel):
    """Configuration for collecting user choice."""
    prompt: str = Field(..., description="Question to ask the user")
    options: List[ChoiceOption] = Field(..., min_length=1, description="Valid choice options")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="How long to wait for response")
    repeat_count: int = Field(default=2, ge=1, le=10, description="How many times to repeat prompt if no response")


class ChoiceCallbackModel(BaseModel):
    """Base for call requests that accept a `choice` prompt.

    Collecting a spoken choice only makes sense if there is somewhere to
    deliver it, so `callback_url` is required whenever `choice` is set.
    """

    @model_validator(mode='after')
    def validate_callback_url_required_for_choice(self):
        if getattr(self, "choice", None) is not None and not getattr(self, "callback_url", None):
            raise ValueError("callback_url is required when choice is specified")
        return self


class OutboundCallRequest(ChoiceCallbackModel):
    """Request to initiate an outbound notification call."""
    message: str = Field(..., description="Message to speak to the recipient")
    extension: str = Field(..., description="SIP extension or phone number to call")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL to POST results to (required if choice is specified)")
    ring_timeout: int = Field(default=30, ge=1, le=600, description="Seconds to wait for call to be answered")
    choice: Optional[ChoicePrompt] = Field(default=None, description="Optional choice prompt for collecting response")
    call_id: Optional[str] = Field(default=None, description="Optional caller-provided ID for tracking")
    reformat_for_speech: bool = Field(
        default=False,
        description="Rewrite the message into natural spoken form via the LLM "
                    "(preserves all facts; falls back to the original on failure)")
    caller_name: Optional[str] = Field(
        default=None, max_length=64,
        description="Display name shown to the person being called, e.g. "
                    "'Weather Alert'. Overrides the From header for this call "
                    "only; unset keeps the agent's registered identity. An "
                    "internal PBX passes this through to the handset, but a "
                    "PSTN carrier will typically replace it with its own CNAM.")


class CallStatus(str, Enum):
    """Status of an outbound call."""
    QUEUED = "queued"
    RINGING = "ringing"
    ANSWERED = "answered"
    COMPLETED = "completed"
    NO_ANSWER = "no_answer"
    FAILED = "failed"
    BUSY = "busy"
    # Verify calls only: the caller hung up before any code was checked.
    HANGUP = "hangup"


class OutboundCallResponse(BaseModel):
    """Response to outbound call request."""
    call_id: str
    status: CallStatus
    message: str
    queue_position: Optional[int] = None


class WebhookPayload(BaseModel):
    """Payload sent to callback webhook."""
    call_id: str
    status: CallStatus
    extension: str
    duration_seconds: float
    message_played: bool
    choice_response: Optional[str] = None
    choice_raw_text: Optional[str] = None
    # Only set when AMD_ENABLED: True if the answerer sounded like a machine.
    machine_answered: Optional[bool] = None
    error: Optional[str] = None


class ToolExecuteRequest(BaseModel):
    """Request to execute a tool."""
    tool: Optional[str] = Field(default=None, description="Optional tool name; the path parameter is authoritative. If set, it must match the path.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the tool")
    speak_result: bool = Field(default=False, description="Speak the result to the active call")
    call_id: Optional[str] = Field(default=None, description="Specific call to speak to (if multiple calls active)")


class ToolCallRequest(ChoiceCallbackModel):
    """Request to execute a tool and call someone with the result."""
    tool: Optional[str] = Field(default=None, description="Optional tool name; the path parameter is authoritative. If set, it must match the path.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the tool")
    extension: str = Field(..., description="SIP extension or phone number to call")
    prefix: Optional[str] = Field(default=None, description="Message to speak before the tool result")
    suffix: Optional[str] = Field(default=None, description="Message to speak after the tool result")
    ring_timeout: int = Field(default=30, ge=1, le=600, description="Seconds to wait for call to be answered")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL to POST call results to (required if choice is specified)")
    choice: Optional[ChoicePrompt] = Field(default=None, description="Optional choice prompt for collecting a spoken response")
    call_id: Optional[str] = Field(default=None, description="Optional caller-provided ID for tracking")
    reformat_for_speech: bool = Field(
        default=False,
        description="Rewrite the composed message into natural spoken form via the LLM")


class WebhookCallRequest(ChoiceCallbackModel):
    """Generic webhook -> call request.

    Place an outbound call driven by an external webhook. Provide a static
    ``message`` and/or a ``tool`` to execute at call time; ``prefix``/``suffix``
    wrap the spoken body, and an optional ``choice`` collects a spoken response.
    """
    extension: str = Field(..., description="SIP extension or phone number to call")
    message: Optional[str] = Field(default=None, description="Static message to speak (used as the body when no tool, or alongside a tool)")
    tool: Optional[str] = Field(default=None, description="Optional tool to execute; its result becomes the spoken body")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the tool")
    prefix: Optional[str] = Field(default=None, description="Message to speak before the body")
    suffix: Optional[str] = Field(default=None, description="Message to speak after the body")
    ring_timeout: int = Field(default=30, ge=1, le=600, description="Seconds to wait for call to be answered")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL to POST call results to (required if choice is specified)")
    choice: Optional[ChoicePrompt] = Field(default=None, description="Optional choice prompt for collecting a spoken response")
    call_id: Optional[str] = Field(default=None, description="Optional caller-provided ID for tracking")
    reformat_for_speech: bool = Field(
        default=False,
        description="Rewrite the composed message into natural spoken form via the LLM")

    @model_validator(mode='after')
    def validate_message_or_tool(self):
        if not self.message and not self.tool:
            raise ValueError("Either message or tool must be provided")
        return self


class ToolCallResponse(BaseModel):
    """Response from tool call request."""
    call_id: str
    status: str
    tool: str
    tool_success: bool
    tool_message: str
    message: str


class ScheduledCallRequest(BaseModel):
    """Request to schedule a call for a future time."""
    extension: str = Field(..., description="SIP extension or phone number to call")
    message: Optional[str] = Field(default=None, description="Message to speak (if no tool specified)")
    tool: Optional[str] = Field(default=None, description="Tool to execute and speak result (e.g., WEATHER)")
    tool_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    delay_seconds: Optional[int] = Field(default=None, description="Seconds from now to make the call")
    at_time: Optional[str] = Field(default=None, description="ISO datetime or HH:MM time to make the call")
    timezone: Optional[str] = Field(default="America/Los_Angeles", description="Timezone for at_time (default: America/Los_Angeles)")
    prefix: Optional[str] = Field(default=None, description="Message to speak before tool result")
    suffix: Optional[str] = Field(default=None, description="Message to speak after tool result")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL to POST results to")
    recurring: Optional[str] = Field(default=None, description="Recurrence pattern: 'daily', 'weekdays', 'weekends', or cron expression")
    reformat_for_speech: bool = Field(
        default=False,
        description="Rewrite the composed message into natural spoken form via the LLM at call time")

    @model_validator(mode='after')
    def validate_time_or_delay(self):
        """Validate that either delay_seconds or at_time is provided."""
        if self.delay_seconds is None and self.at_time is None:
            raise ValueError("Either delay_seconds or at_time must be provided")
        if self.delay_seconds is not None and self.at_time is not None:
            raise ValueError("Provide either delay_seconds or at_time, not both")
        return self
    
    @model_validator(mode='after')
    def validate_message_or_tool(self):
        """Validate that either message or tool is provided."""
        if not self.message and not self.tool:
            raise ValueError("Either message or tool must be provided")
        return self


class ScheduledCallResponse(BaseModel):
    """Response from scheduled call request."""
    schedule_id: str
    status: str
    extension: str
    scheduled_for: str
    delay_seconds: int
    message: str
    recurring: Optional[str] = None


class ScheduledCallInfo(BaseModel):
    """Information about a scheduled call."""
    schedule_id: str
    extension: str
    scheduled_for: str
    remaining_seconds: int
    message: Optional[str] = None
    tool: Optional[str] = None
    recurring: Optional[str] = None
    status: str


class VirtualNumberRequest(BaseModel):
    """Request to create an ephemeral inbound extension."""
    number: Optional[str] = Field(
        default=None,
        description="Explicit extension (digits/*/#); omit to auto-allocate "
                    "from VIRTUAL_NUMBER_RANGE")
    ttl_s: Optional[int] = Field(
        default=None, gt=0,
        description="Seconds until the unused number expires "
                    "(default VIRTUAL_NUMBER_DEFAULT_TTL_S, clamped to max)")
    purpose: str = Field(
        ..., min_length=1, max_length=2000,
        description="What this number is for; injected into the system prompt "
                    "for the call that arrives on it")
    greeting: Optional[str] = Field(
        default=None, max_length=500,
        description="Custom greeting spoken instead of the default one")
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL to POST the call outcome (and transcript) to")
    include_transcript: bool = Field(
        default=True,
        description="Include the transcript in the completion webhook")
    persistent: bool = Field(
        default=False,
        description="Trigger number: never expires and is not consumed by "
                    "its calls — every call to it fires the webhook until "
                    "the number is deleted (ttl_s is ignored)")
    events: Optional[List[str]] = Field(
        default=None,
        description="Call-time webhooks to fire: answered (call matched, "
                    "before the greeting), first_speech (caller's first "
                    "utterance), speech (every utterance), completed (call "
                    "ended, + transcript). Default [\"completed\"]")


class VirtualNumberResponse(BaseModel):
    """A virtual number registry entry."""
    id: str
    number: str
    sip_uri: str
    status: str
    purpose: str
    expires_at: float
    created_at: float
    persistent: bool = False
    events: List[str] = []
    callback_url: str = ""


class VerifyRequest(BaseModel):
    """Out-of-band identity check for a caller (no live call needed)."""
    caller_id: str = Field(..., min_length=1, max_length=64,
                           description="Caller id (SIP URI user part)")
    pin: Optional[str] = Field(default=None, description="Static PIN to check")
    otp: Optional[str] = Field(default=None, description="One-time (TOTP) code to check")

    @model_validator(mode="after")
    def _at_least_one_factor(self):
        if not (self.pin or self.otp):
            raise ValueError("Provide a pin and/or otp to check")
        return self


class VerifyResponse(BaseModel):
    """Result of an identity check."""
    caller_id: str
    verified: bool
    method: Optional[str] = None  # "pin" | "otp" | None


class VerifyCallRequest(BaseModel):
    """Place an outbound call that verifies a caller's identity by keypad.

    The agent dials ``extension`` (defaulting to ``caller_id``), asks the person
    to key in their PIN or one-time code, and checks it against the credentials
    stored for ``caller_id``. Digits are entered by DTMF, never spoken, so the
    code never lands in the transcript.
    """
    caller_id: Optional[str] = Field(
        default=None, max_length=64,
        description="Caller id whose stored credentials are checked (SIP URI user "
                    "part). Defaults to the extension when omitted.")
    extension: Optional[str] = Field(
        default=None,
        description="SIP extension or number to dial (defaults to caller_id)")
    method: str = Field(
        default="auto", pattern=r"^(pin|otp|auto)$",
        description="Which factor to require: 'pin', 'otp', or 'auto' (either)")
    pin: Optional[str] = Field(
        default=None,
        description="Check the entered code against this PIN for this call "
                    "(instead of the caller's stored/global PIN)")
    totp_secret: Optional[str] = Field(
        default=None,
        description="Check the entered code against this base32 TOTP secret for "
                    "this call (instead of the stored/global secret)")
    totp_digits: Optional[int] = Field(
        default=None, ge=4, le=10,
        description="Digits in the TOTP code (defaults to VERIFY_TOTP_DIGITS)")
    totp_period: Optional[int] = Field(
        default=None, ge=5, le=300,
        description="TOTP step in seconds (defaults to VERIFY_TOTP_PERIOD)")
    totp_algorithm: Optional[str] = Field(
        default=None, pattern=r"^(?i:sha1|sha256|sha512)$",
        description="TOTP hash: SHA1|SHA256|SHA512 (defaults to VERIFY_TOTP_ALGORITHM)")
    totp_window: Optional[int] = Field(
        default=None, ge=0, le=10,
        description="Clock-skew steps to accept (defaults to VERIFY_TOTP_WINDOW)")
    prompt: Optional[str] = Field(
        default=None,
        description="Custom spoken prompt (defaults to VERIFY_CALL_PROMPT)")
    retry_phrase: Optional[str] = Field(
        default=None,
        description="Spoken line after a wrong code (defaults to VERIFY_CALL_RETRY_PHRASE)")
    success_phrase: Optional[str] = Field(
        default=None,
        description="Spoken line on success (defaults to VERIFY_CALL_SUCCESS_PHRASE)")
    fail_phrase: Optional[str] = Field(
        default=None,
        description="Spoken line on failure (defaults to VERIFY_CALL_FAIL_PHRASE)")
    ring_timeout: int = Field(default=30, ge=1, le=600,
                              description="Seconds to wait for the call to be answered")
    callback_url: Optional[str] = Field(
        default=None, description="Optional webhook URL to POST the result to")
    call_id: Optional[str] = Field(default=None, description="Optional caller-provided ID for tracking")


class VerifyCallResponse(BaseModel):
    """Result of an outbound identity-verification call."""
    call_id: str
    status: CallStatus
    verified: bool
    method: Optional[str] = None  # "pin" | "otp" | None
    attempts: int = 0
    error: Optional[str] = None


class VerifyCredentialsRequest(BaseModel):
    """Enroll or update a caller's verification factors."""
    caller_id: str = Field(..., min_length=1, max_length=64,
                           description="Caller id (SIP URI user part)")
    pin: Optional[str] = Field(default=None, description="Static PIN to set/rotate")
    totp_secret: Optional[str] = Field(
        default=None, description="Base32 TOTP secret to store (ignored if generate_totp)")
    generate_totp: bool = Field(
        default=False, description="Mint a fresh random TOTP secret for this caller")


class VerifyCredentialsResponse(BaseModel):
    """Public view of a caller's enrollment (never the secret or PIN hash)."""
    caller_id: str
    has_pin: bool
    has_totp: bool
    provisioning_uri: Optional[str] = None
    updated_at: Optional[str] = None


class OtpResponse(BaseModel):
    """Current TOTP code for a caller (for delivery/testing)."""
    caller_id: str
    otp: str
    expires_in_s: int


class ToolExecuteResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    tool: str
    message: str
    data: Optional[Dict[str, Any]] = None
    spoken: bool = False
    error: Optional[str] = None


class ToolInfo(BaseModel):
    """Information about an available tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool


# ============================================================================
# Outbound Call Handler
# ============================================================================

class OutboundCallHandler:
    """Handles outbound notification calls."""
    
    def __init__(self, assistant: 'SIPAIAssistant', call_queue: 'CallQueue' = None):
        self.assistant = assistant
        self.call_queue = call_queue
        self.pending_calls: Dict[str, OutboundCallRequest] = {}
        self._tasks: set = set()
        self._call_counter = 0
        
    def generate_call_id(self) -> str:
        """Generate a unique call ID."""
        self._call_counter += 1
        import time
        return f"out-{int(time.time())}-{self._call_counter}"
        
    async def initiate_call(self, request: OutboundCallRequest) -> tuple[str, int]:
        """
        Initiate an outbound call.
        Returns (call_id, queue_position).

        Raises RequestRejected on invalid input (400), duplicate call_id (409),
        or backpressure (429).
        """
        config = self.assistant.config

        # Validate dial target and webhook before doing anything else.
        validate_extension(request.extension, config)
        await validate_callback_url(request.callback_url, config)
        # Caller-provided IDs become Redis keys and transcript filenames.
        if request.call_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", request.call_id):
            raise RequestRejected(
                400, "call_id may only contain letters, digits, '.', '_', '-' (max 64 chars)")

        # Clamp caller-supplied timeouts to configured maxima so a single
        # request can't monopolise the call pipeline.
        request.ring_timeout = min(request.ring_timeout, config.max_ring_timeout_s)
        if request.choice:
            request.choice.timeout_seconds = min(
                request.choice.timeout_seconds, config.max_choice_timeout_s)
            request.choice.repeat_count = min(
                request.choice.repeat_count, config.max_choice_repeat)

        call_id = request.call_id or self.generate_call_id()

        # Opt-in LLM rewrite of the (already composed) message into spoken
        # form. Done before queueing so the reformatted text is what persists.
        request.message = await _maybe_reformat(
            self.assistant, request.message, request.reformat_for_speech)

        log_event(logger, logging.INFO, f"Initiating outbound call to {request.extension}",
                 event="outbound_call_initiated", call_id=call_id, extension=request.extension)

        # Use queue if available
        if self.call_queue:
            # Reject duplicate caller-provided IDs and apply queue-depth
            # backpressure. Only a call that is still queued or processing
            # conflicts — finished records persist in Redis for 24h and their
            # IDs may legitimately be reused (retries, recurring call IDs).
            if request.call_id:
                existing = await self.call_queue.get_call(call_id)
                if existing and existing.status.value in ("queued", "processing"):
                    raise RequestRejected(409, f"call_id '{call_id}' is already queued or in progress")
            queue_status = await self.call_queue.get_queue_status()
            if queue_status.get("queued", 0) >= config.max_queue_depth:
                raise RequestRejected(429, "Call queue is full; try again later")
            queued_call = await self.call_queue.enqueue(call_id, request)
            return call_id, queued_call.position
        else:
            # Direct execution (no queue)
            if call_id in self.pending_calls:
                raise RequestRejected(409, f"call_id '{call_id}' already in progress")
            if len(self.pending_calls) >= config.max_direct_concurrent_calls:
                raise RequestRejected(429, "Too many concurrent calls in progress; try again later")
            self.pending_calls[call_id] = request
            task = asyncio.create_task(self._execute_call(call_id, request))
            # Keep a strong reference so the task isn't GC'd, and drop it on done.
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return call_id, 0
        
    async def _execute_call(self, call_id: str, request: OutboundCallRequest):
        """Execute the outbound call flow.

        Returns the final (CallStatus, error) so callers (the queue worker)
        can persist the real outcome; expected failures (initiate failure,
        no answer) do not raise.
        """
        # Task body top: this notification call has no CallSession of its
        # own — explicitly unbind so nothing in this task (AMD, choice
        # collection, hangups) can resolve to a live conversational session
        # inherited from the spawning context.
        set_current_session(None)
        start_time = asyncio.get_event_loop().time()
        status = CallStatus.FAILED
        message_played = False
        choice_response = None
        choice_raw_text = None
        machine_answered = None
        error = None
        call_info = None
        hung_up = False

        transcripts = getattr(self.assistant, "transcripts", None)
        if transcripts:
            transcripts.start(call_id, "outbound-notification", request.extension)

        with create_span("api.execute_call", {
            "call.id": call_id,
            "call.extension": request.extension,
            "call.has_choice": request.choice is not None
        }) as span:
            try:
                # Build SIP URI
                extension = request.extension
                if not extension.startswith('sip:'):
                    if '@' not in extension:
                        extension = f"sip:{extension}@{self.assistant.config.sip_domain}"
                    else:
                        extension = f"sip:{extension}"
                
                span.set_attribute("call.sip_uri", extension)
                log_event(logger, logging.INFO, f"Making call to {extension}",
                         event="outbound_call_dialing", call_id=call_id, uri=extension)
                
                # Pre-generate TTS for message
                message_audio = await self.assistant.audio_pipeline.synthesize(request.message)
                if not message_audio:
                    raise Exception("Failed to generate TTS for message")
                
                # Pre-generate TTS for choice prompt if needed
                choice_audio = None
                if request.choice:
                    choice_audio = await self.assistant.audio_pipeline.synthesize(request.choice.prompt)
                
                # Make the call
                call_info = await self.assistant.sip_handler.make_call(
                    extension, caller_name=request.caller_name)
                if not call_info:
                    status = CallStatus.FAILED
                    error = "Failed to initiate call"
                    span.set_attribute("call.error", error)
                    Metrics.record_call_failed("outbound", "initiate_failed")
                    # Expected failure, not an exception: skip to the finally
                    # block (which still sends the webhook) without an ERROR log.
                    return status, error
                
                status = CallStatus.RINGING
                span.set_attribute("call.status", "ringing")
                
                # Wait for answer
                ring_start = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - ring_start < request.ring_timeout:
                    if getattr(call_info, 'is_active', False):
                        status = CallStatus.ANSWERED
                        span.set_attribute("call.status", "answered")
                        break
                    await asyncio.sleep(0.5)
                else:
                    status = CallStatus.NO_ANSWER
                    span.set_attribute("call.status", "no_answer")
                    log_event(logger, logging.WARNING, f"Call not answered within {request.ring_timeout}s",
                             event="outbound_call_no_answer", call_id=call_id)
                    Metrics.record_call_failed("outbound", "no_answer")
                    await self.assistant.sip_handler.hangup_call(call_info)
                    hung_up = True
                    # Routine outcome: leave error=None and skip to the finally
                    # block (webhook reports status=no_answer) without an ERROR log.
                    return status, error
                
                log_event(logger, logging.INFO, "Call answered",
                         event="outbound_call_answered", call_id=call_id)

                # Wait for media to be ready
                await asyncio.sleep(1)

                # Classify the answerer before speaking (config-gated). The
                # message still plays either way — voicemail delivery is often
                # wanted — but the webhook reports who (or what) answered.
                # Per-call audio state for AMD + choice collection (shared
                # across both stages of this call, isolated from the live
                # inbound session's pipeline state).
                audio_state = self.assistant.audio_pipeline.new_session_state()

                if self.assistant.config.amd_enabled:
                    machine_answered = await self._detect_answering_machine(
                        call_info, audio_state)
                    span.set_attribute("call.machine_answered", machine_answered)
                    log_event(logger, logging.INFO,
                             f"AMD result: {'machine' if machine_answered else 'human'}",
                             event="outbound_call_amd", call_id=call_id,
                             machine=machine_answered)

                # Play the message
                await self.assistant.sip_handler.send_audio(call_info, message_audio)
                audio_duration = len(message_audio) / (self.assistant.config.sample_rate * 2)
                await asyncio.sleep(audio_duration + 0.5)
                message_played = True
                span.set_attribute("call.message_played", True)
                if transcripts:
                    transcripts.add_turn(call_id, "assistant", request.message)
                
                log_event(logger, logging.INFO, "Message played",
                         event="outbound_call_message_played", call_id=call_id)
                
                # Handle choice collection if configured
                if request.choice and choice_audio:
                    choice_response, choice_raw_text = await self._collect_choice(
                        call_id, call_info, request.choice, choice_audio,
                        audio_state
                    )
                    if transcripts:
                        transcripts.add_turn(call_id, "assistant", request.choice.prompt)
                        if choice_raw_text:
                            transcripts.add_turn(call_id, "user", choice_raw_text)

                    span.set_attribute("call.choice_response", choice_response or "none")
                    log_event(logger, logging.INFO, f"Choice collected: {choice_response}",
                             event="outbound_call_choice_collected", call_id=call_id, 
                             response=choice_response, raw_text=choice_raw_text)
                    
                    # Play acknowledgment if choice was matched
                    if choice_response and call_info.is_active:
                        try:
                            ack_audio = await self.assistant.audio_pipeline.synthesize("Acknowledged.")
                            if ack_audio:
                                await self.assistant.sip_handler.send_audio(call_info, ack_audio)
                                ack_duration = len(ack_audio) / (self.assistant.config.sample_rate * 2)
                                await asyncio.sleep(ack_duration + 0.3)
                                log_event(logger, logging.INFO, "Acknowledgment played",
                                         event="outbound_call_ack_played", call_id=call_id)
                        except Exception as e:
                            logger.warning(f"Failed to play acknowledgment: {e}")
                
                status = CallStatus.COMPLETED
                span.set_attribute("call.status", "completed")
                
                # Hang up
                if call_info.is_active:
                    await self.assistant.sip_handler.hangup_call(call_info)
                    hung_up = True

            except Exception as e:
                error = str(e)
                logger.error(f"Outbound call error: {e}", exc_info=True)
                span.record_exception(e)
                span.set_attribute("call.error", error)
                
            finally:
                # If the call was placed and is still up (e.g. an exception
                # fired after the call was answered), tear down the SIP/RTP leg
                # so we don't leak an active call. The success/no-answer paths
                # already hung up and set hung_up, so this won't double-hangup.
                if call_info is not None and not hung_up and getattr(call_info, 'is_active', False):
                    try:
                        await self.assistant.sip_handler.hangup_call(call_info)
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to hang up call during cleanup: {cleanup_err}")

                # Clean up
                if call_id in self.pending_calls:
                    del self.pending_calls[call_id]

                if transcripts:
                    transcripts.end(call_id)

                # Calculate duration
                duration = asyncio.get_event_loop().time() - start_time
                span.set_attribute("call.duration_s", round(duration, 2))
                
                # Send webhook if callback_url provided
                if request.callback_url:
                    await self._send_webhook(
                        request.callback_url,
                        WebhookPayload(
                            call_id=call_id,
                            status=status,
                            extension=request.extension,
                            duration_seconds=round(duration, 2),
                            message_played=message_played,
                            choice_response=choice_response,
                            choice_raw_text=choice_raw_text,
                            machine_answered=machine_answered,
                            error=error
                        )
                    )

        return status, error

    async def _detect_answering_machine(self, call_info, audio_state) -> bool:
        """Heuristic AMD: a human answers briefly ("Hello?") then waits; a
        machine greeting keeps talking. Listens for AMD_WINDOW_S and returns
        True once continuous speech exceeds AMD_MACHINE_SPEECH_MS.
        """
        cfg = self.assistant.config
        loop = asyncio.get_event_loop()
        window_end = loop.time() + cfg.amd_window_s
        continuous_ms = 0.0

        while loop.time() < window_end:
            if not getattr(call_info, 'is_active', False):
                return False
            chunk = None
            try:
                chunk = await self.assistant.sip_handler.receive_audio(
                    call_info, timeout=0.1)
            except Exception as e:
                logger.debug(f"AMD audio receive error: {e}")

            # update_noise=True: has_speech() is the ONLY per-chunk check in
            # this window (process_audio never runs during AMD), so it must
            # keep feeding the adaptive noise floor — with a frozen floor,
            # steady line noise would read as continuous machine speech.
            if chunk and self.assistant.audio_pipeline.has_speech(
                    audio_state, chunk, update_noise=True):
                # Chunks are variable-length (up to 100ms from receive_audio);
                # credit the real duration, not a flat chunk_duration_ms, or
                # amd_machine_speech_ms takes ~5x longer to reach than configured.
                continuous_ms += len(chunk) / 2 / cfg.sample_rate * 1000
                if continuous_ms >= cfg.amd_machine_speech_ms:
                    return True
            elif chunk:
                # Silence resets the run — human "Hello?" then quiet.
                continuous_ms = 0.0

            await asyncio.sleep(0.02)

        return False

    async def _collect_choice(
        self,
        call_id: str,
        call_info,
        choice: ChoicePrompt,
        choice_audio: bytes,
        audio_state
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Collect user choice via voice.
        Returns (matched_value, raw_transcription).
        """
        # Digits from before the prompt shouldn't pre-answer it; digits pressed
        # DURING the prompt (barge-in style) are kept.
        clear_dtmf = getattr(self.assistant.sip_handler, 'clear_dtmf', None)
        if clear_dtmf:
            clear_dtmf(call_info)

        last_text = None
        for attempt in range(choice.repeat_count):
            # Play prompt
            await self.assistant.sip_handler.send_audio(call_info, choice_audio)
            audio_duration = len(choice_audio) / (self.assistant.config.sample_rate * 2)
            await asyncio.sleep(audio_duration + 0.3)

            # Listen for a spoken response or a DTMF keypress
            response = await self._listen_for_response(
                call_info,
                timeout=choice.timeout_seconds,
                audio_state=audio_state
            )

            if response:
                kind, value = response
                if kind == "dtmf":
                    matched = self._match_dtmf(value, choice.options)
                    last_text = f"DTMF {value}"
                    if matched:
                        return matched, last_text
                else:
                    matched = self._match_choice(value, choice.options)
                    last_text = value
                    if matched:
                        return matched, value

                # No match - will retry if attempts remain
                log_event(logger, logging.INFO, f"No choice matched for: {last_text}",
                         event="outbound_call_choice_no_match", call_id=call_id,
                         attempt=attempt + 1, text=last_text)

        return None, last_text
        
    async def _listen_for_response(self, call_info, timeout: float,
                                   audio_state=None) -> Optional[tuple]:
        """Listen for a spoken response or a DTMF keypress.

        Returns ("speech", transcription) or ("dtmf", digit), or None on
        timeout/hangup. DTMF wins whenever a digit is buffered — it's an
        unambiguous signal, unlike STT.
        """
        start_time = asyncio.get_event_loop().time()
        get_dtmf = getattr(self.assistant.sip_handler, 'get_dtmf_digit', None)
        if audio_state is None:
            audio_state = self.assistant.audio_pipeline.new_session_state()

        # Speculative endpointing's short silence cutoff relies on main.py's
        # hold/merge machinery, which this collection path does not have —
        # fall back to the fixed timeout so a mid-answer hesitation ("um ...
        # option two") isn't committed as a fragment. Adaptive mode is
        # self-contained in the VAD and stays as configured.
        endpoint_mode = ("fixed"
                        if self.assistant.config.endpoint_mode == "speculative"
                        else None)

        while asyncio.get_event_loop().time() - start_time < timeout:
            if not getattr(call_info, 'is_active', False):
                break

            digit = get_dtmf(call_info) if get_dtmf else None
            if digit:
                return ("dtmf", digit)

            if not getattr(call_info, 'media_ready', False):
                await asyncio.sleep(0.1)
                continue

            try:
                audio_chunk = await self.assistant.sip_handler.receive_audio(
                    call_info,
                    timeout=0.1
                )

                if audio_chunk:
                    transcription = await self.assistant.audio_pipeline.process_audio(
                        audio_state, audio_chunk, endpoint_mode=endpoint_mode)
                    if transcription and len(transcription.strip()) > 1:
                        return ("speech", transcription.strip())

            except Exception as e:
                logger.debug(f"Audio receive error: {e}")

            await asyncio.sleep(0.05)

        return None

    def _match_dtmf(self, digit: str, options: List[ChoiceOption]) -> Optional[str]:
        """Match a DTMF digit to a choice option.

        An option's explicit `dtmf` key wins; options without one answer to
        their 1-based position in the list.
        """
        for idx, option in enumerate(options, start=1):
            if option.dtmf is not None:
                if digit == option.dtmf:
                    return option.value
            elif digit == str(idx):
                return option.value
        return None
        
    def _match_choice(self, text: str, options: List[ChoiceOption]) -> Optional[str]:
        """Match transcribed text to a choice option.

        Matches on whole words/phrases rather than bare substrings so that, e.g.,
        "I don't know" does not match the synonym "no" (a substring of "know")
        and "yesterday" does not match "yes".
        """
        text_lower = text.lower().strip()
        # Tokenize into words for whole-word checks.
        words = set(re.findall(r"\w+", text_lower))

        def phrase_present(phrase: str) -> bool:
            phrase = phrase.lower().strip()
            if not phrase:
                return False
            # Multi-word phrase: require it to appear on word boundaries.
            if " " in phrase:
                return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text_lower) is not None
            # Single token: require an exact word match.
            return phrase in words

        def matches(candidate: str) -> bool:
            candidate = candidate.lower().strip()
            if not candidate:
                return False
            if phrase_present(candidate):
                return True
            # Reverse containment: the whole utterance appears inside a longer
            # candidate on word boundaries, so a spoken "yes" still matches an
            # option whose only synonym is "yes please".
            return re.search(rf"(?<!\w){re.escape(text_lower)}(?!\w)", candidate) is not None

        for option in options:
            # Exact full-text value match
            if option.value.lower() == text_lower:
                return option.value
            # Synonym present as a whole word/phrase (either direction)
            if any(matches(synonym) for synonym in option.synonyms):
                return option.value
            # Value present as a whole word/phrase (either direction)
            if matches(option.value):
                return option.value

        return None
        
    async def _send_webhook(self, url: str, payload: WebhookPayload):
        """Send result to callback webhook (signed + retried via deliver_webhook)."""
        with create_span("api.send_webhook", {
            "webhook.url": url,
            "webhook.call_id": payload.call_id,
            "webhook.status": payload.status.value
        }) as span:
            log_event(logger, logging.INFO, f"Sending webhook to {url}",
                     event="outbound_call_webhook", url=url, status=payload.status)

            delivered = await deliver_webhook(
                url, payload.model_dump(), self.assistant.config,
                api_name="call_webhook")

            span.set_attribute("webhook.success", delivered)
            if delivered:
                log_event(logger, logging.INFO, "Webhook sent successfully",
                         event="outbound_call_webhook_success", url=url)

    # --- Outbound identity verification ---------------------------------
    async def _collect_code(self, call_info, timeout: float,
                            prompt_audio: Optional[bytes] = None) -> Optional[str]:
        """Speak the prompt and collect keypad digits for a PIN/OTP.

        Thin wrapper over the shared ``dtmf_collect.collect_dtmf_code`` loop
        (also used by the in-call VERIFY tool): first keypress mutes the prompt,
        '*' restarts entry, '#' or an inter-digit pause submits. NEVER log the
        returned code.
        """
        interdigit = float(getattr(self.assistant.config, "verify_dtmf_interdigit_s", 3.0))
        return await collect_dtmf_code(
            self.assistant.sip_handler, call_info, timeout=timeout,
            interdigit=interdigit, prompt_audio=prompt_audio)

    async def _say(self, call_info, text: str) -> None:
        """Speak a line into the live call and wait for it to finish (best-effort)."""
        try:
            audio = await self.assistant.audio_pipeline.synthesize(text)
            if audio and getattr(call_info, "is_active", False):
                await self.assistant.sip_handler.send_audio(call_info, audio)
                duration = len(audio) / (self.assistant.config.sample_rate * 2)
                await asyncio.sleep(duration + 0.3)
        except Exception as e:
            logger.debug(f"verify-call prompt playback failed: {e}")

    async def run_verify_call(self, request: 'VerifyCallRequest') -> 'VerifyCallResponse':
        """Dial the caller, collect a PIN/OTP by keypad, and verify it.

        Runs synchronously (the HTTP request awaits the verdict). Bounded by
        ring_timeout, VERIFY_DTMF_TIMEOUT_S and VERIFY_MAX_ATTEMPTS so the call
        can't run unbounded. Fails closed on the security decision (any error
        leaves verified=False) but never raises on an expected call outcome.
        """
        config = self.assistant.config
        caller_id = (request.caller_id or "").strip()
        extension = (request.extension or caller_id).strip()
        call_id = request.call_id or self.generate_call_id()

        # caller_id is optional: it names whose stored credentials to check (and
        # is the default dial target). Omitted, it defaults to the extension so
        # the extension's own enrollment is consulted. At least one is required.
        if not extension:
            raise RequestRejected(400, "Provide a caller_id or an extension to dial")
        if not caller_id and is_safe_caller_id(extension):
            caller_id = extension
        if caller_id and not is_safe_caller_id(caller_id):
            raise RequestRejected(400, "Invalid caller_id")
        validate_extension(extension, config)
        await validate_callback_url(request.callback_url, config)
        if request.call_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", request.call_id):
            raise RequestRejected(
                400, "call_id may only contain letters, digits, '.', '_', '-' (max 64 chars)")

        verifier = self.assistant.verifier

        # Per-request ("ad-hoc") credentials: when the caller supplies a PIN
        # and/or TOTP secret in the request, the entered code is checked against
        # exactly those, with no store/global lookup (an n8n workflow that holds
        # the factors itself, without enrolling the caller first).
        adhoc_pin = request.pin or None
        adhoc_secret = (request.totp_secret or "").strip().replace(" ", "").upper() or None
        if adhoc_secret and not re.fullmatch(r"[A-Z2-7]+=*", adhoc_secret):
            raise RequestRejected(400, "totp_secret must be base32")
        adhoc = bool(adhoc_pin or adhoc_secret)

        # No ad-hoc factor and no stored/global factor: nothing to check.
        if not adhoc and not verifier.can_verify(caller_id):
            raise RequestRejected(
                400, "No verification credentials configured: supply a pin/totp_secret "
                     "or a caller_id (or global VERIFY_*) with enrolled credentials")

        # Concurrency guard (verify calls are not queued — they're interactive).
        if call_id in self.pending_calls:
            raise RequestRejected(409, f"call_id '{call_id}' already in progress")
        if len(self.pending_calls) >= config.max_direct_concurrent_calls:
            raise RequestRejected(429, "Too many concurrent calls in progress; try again later")

        request.ring_timeout = min(request.ring_timeout, config.max_ring_timeout_s)
        # Park a marker so the concurrency guard and /call/{id} see this call.
        self.pending_calls[call_id] = request  # type: ignore[assignment]

        set_current_session(None)
        status = CallStatus.FAILED
        verified = False
        method: Optional[str] = None
        attempts = 0
        error: Optional[str] = None
        call_info = None
        hung_up = False

        with create_span("api.verify_call", {
            "call.id": call_id, "call.extension": extension, "verify.method": request.method,
        }) as span:
            try:
                uri = extension
                if not uri.startswith("sip:"):
                    uri = f"sip:{uri}@{config.sip_domain}" if "@" not in uri else f"sip:{uri}"

                call_info = await self.assistant.sip_handler.make_call(uri)
                if not call_info:
                    error = "Failed to initiate call"
                    Metrics.record_call_failed("verify", "initiate_failed")
                    raise _VerifyCallDone()

                status = CallStatus.RINGING
                ring_start = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - ring_start < request.ring_timeout:
                    if getattr(call_info, "is_active", False):
                        status = CallStatus.ANSWERED
                        break
                    await asyncio.sleep(0.5)
                else:
                    status = CallStatus.NO_ANSWER
                    Metrics.record_call_failed("verify", "no_answer")
                    await self.assistant.sip_handler.hangup_call(call_info)
                    hung_up = True
                    raise _VerifyCallDone()

                await asyncio.sleep(1)  # let media settle

                # Spoken lines: per-request override, else the configured default.
                prompt = request.prompt or config.verify_call_prompt
                retry_phrase = request.retry_phrase or config.verify_call_retry_phrase
                success_phrase = request.success_phrase or config.verify_call_success_phrase
                fail_phrase = request.fail_phrase or config.verify_call_fail_phrase
                dtmf_timeout = float(getattr(config, "verify_dtmf_timeout_s", 20.0))
                max_attempts = max(1, int(getattr(config, "verify_max_attempts", 3)))
                # Pre-synthesize the prompt once; it's replayed each attempt and
                # the caller's first keypress mutes it (barge-in) inside collect.
                prompt_audio = await self.assistant.audio_pipeline.synthesize(prompt)

                empty_entries = 0
                while attempts < max_attempts and getattr(call_info, "is_active", False):
                    code = await self._collect_code(call_info, dtmf_timeout, prompt_audio)
                    if not code:
                        if not getattr(call_info, "is_active", False):
                            break  # hung up mid-prompt: not a completed attempt
                        # Timeout with nothing keyed — not a wrong code (the in-call
                        # VERIFY tool doesn't burn an attempt either), but bound
                        # the re-prompts so the call can't run unbounded.
                        empty_entries += 1
                        if empty_entries >= max_attempts:
                            break
                        continue
                    if adhoc:
                        ok, used = await verifier.averify_explicit(
                            code, pin=adhoc_pin, totp_secret=adhoc_secret,
                            method=request.method, totp_digits=request.totp_digits,
                            totp_period=request.totp_period,
                            totp_algorithm=request.totp_algorithm,
                            totp_window=request.totp_window)
                    else:
                        ok, used = await verifier.averify(caller_id, code, method=request.method)
                    attempts += 1
                    # Diagnostic only — length, never the code itself.
                    log_event(logger, logging.DEBUG, "Verify attempt",
                              event="verify_call_attempt", call_id=call_id,
                              code_len=len(code), method=request.method,
                              adhoc=adhoc, matched=used, ok=ok)
                    if ok:
                        verified, method = True, used
                        break
                    if attempts < max_attempts:
                        await self._say(call_info, retry_phrase)

                # A caller who hung up before any code was checked is
                # distinguishable from a wrong code on the webhook.
                status = (CallStatus.COMPLETED
                          if getattr(call_info, "is_active", False) or attempts
                          else CallStatus.HANGUP)
                if status is CallStatus.HANGUP and error is None:
                    error = "Caller hung up before entering a code"
                # NEVER log the entered code — only the outcome.
                log_event(logger, logging.INFO,
                          f"Verify call {'succeeded' if verified else 'failed'}",
                          event="verify_call", outcome="ok" if verified else "failed",
                          caller=caller_id or extension, call_id=call_id,
                          method=method, attempts=attempts)
                span.set_attribute("verify.verified", verified)

                if getattr(call_info, "is_active", False):
                    await self._say(call_info, success_phrase if verified else fail_phrase)
                    if getattr(call_info, "is_active", False):
                        await self.assistant.sip_handler.hangup_call(call_info)
                        hung_up = True

            except _VerifyCallDone:
                # Terminal non-answer outcome — status/error already set; fall
                # through to the shared teardown + webhook exit below.
                pass
            except Exception as e:
                error = str(e)
                logger.error(f"Verify call error: {e}", exc_info=True)
                span.record_exception(e)
            finally:
                if call_info is not None and not hung_up and getattr(call_info, "is_active", False):
                    try:
                        await self.assistant.sip_handler.hangup_call(call_info)
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to hang up verify call: {cleanup_err}")
                self.pending_calls.pop(call_id, None)

        response = VerifyCallResponse(call_id=call_id, status=status, verified=verified,
                                      method=method, attempts=attempts, error=error)
        if request.callback_url:
            try:
                await deliver_webhook(request.callback_url, response.model_dump(mode="json"),
                                      config, api_name="verify_call_webhook")
            except Exception as e:
                logger.warning(f"verify-call webhook delivery failed: {e}")
        return response


# ============================================================================
# FastAPI Application
# ============================================================================

def create_api(assistant: 'SIPAIAssistant', call_queue: 'CallQueue' = None) -> FastAPI:
    """Create FastAPI application for outbound calls."""
    
    app = FastAPI(
        title="SIP AI Assistant API",
        description="API for outbound notification calls with optional response collection",
        version="1.0.0"
    )
    
    handler = OutboundCallHandler(assistant, call_queue)

    def _tool_failed_response(tool_label: str, tool_message: str) -> ToolCallResponse:
        return ToolCallResponse(
            call_id="",
            status="tool_failed",
            tool=tool_label,
            tool_success=False,
            tool_message=tool_message,
            message=f"Tool execution failed: {tool_message}"
        )

    async def _place_composed_call(request, full_message: str, tool_label: str,
                                   tool_message: str, event: str,
                                   response_message: str) -> ToolCallResponse:
        """Place the outbound call for a tool/webhook-driven request.

        Passes choice + call_id through so the webhook caller can also collect
        a spoken response. Shared by /tools/{name}/call and /webhook/call.
        """
        call_request = OutboundCallRequest(
            message=full_message,
            extension=request.extension,
            callback_url=request.callback_url,
            ring_timeout=request.ring_timeout,
            choice=request.choice,
            call_id=request.call_id,
            reformat_for_speech=getattr(request, "reformat_for_speech", False),
        )
        call_id, position = await handler.initiate_call(call_request)
        log_event(logger, logging.INFO, f"Call initiated: {call_id}",
                 event=event,
                 tool=tool_label or None,
                 call_id=call_id,
                 extension=request.extension)
        return ToolCallResponse(
            call_id=call_id,
            status="queued" if position > 0 else "initiated",
            tool=tool_label,
            tool_success=True,
            tool_message=tool_message,
            message=response_message,
        )

    # Auth + rate-limit dependencies for mutating endpoints. Both are no-ops
    # when unconfigured (API_AUTH_TOKEN unset / RATE_LIMIT_RPM=0).
    auth = make_auth_dependency(assistant.config.api_auth_token)
    rate_limit = make_rate_limit_dependency(assistant.config)
    protected = [Depends(auth), Depends(rate_limit)]

    # In-process event bus feeding the admin dashboard's SSE stream. The real
    # assistant attaches one at construction; attach one here too so any
    # assistant-shaped object (tests, embedding) gets a working bus.
    events_bus: EventBus = getattr(assistant, "events", None) or EventBus()
    if getattr(assistant, "events", None) is None:
        try:
            assistant.events = events_bus
        except Exception:
            pass
    if not assistant.config.api_auth_token:
        logger.warning(
            "API_AUTH_TOKEN is not set - REST API endpoints are unauthenticated. "
            "Set API_AUTH_TOKEN and/or bind API_HOST to a trusted interface in production."
        )

    # Cached dependency probes for /health?deep=true — the TTL keeps repeated
    # monitoring hits from hammering the backends.
    _deps_cache: Dict[str, Any] = {"ts": 0.0, "deps": None}
    _DEPS_CACHE_TTL_S = 10.0

    async def _probe_dependencies() -> Dict[str, str]:
        now = time.monotonic()
        if _deps_cache["deps"] is not None and now - _deps_cache["ts"] < _DEPS_CACHE_TTL_S:
            return _deps_cache["deps"]
        deps: Dict[str, str] = {}

        async def probe_http(name: str, url: str, headers: Optional[Dict[str, str]] = None):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(url, headers=headers)
                    deps[name] = "up" if resp.status_code < 500 else f"error (HTTP {resp.status_code})"
            except Exception as e:
                deps[name] = f"down ({type(e).__name__})"

        async def probe_redis():
            if not call_queue or not call_queue.redis:
                deps["redis"] = "disabled"
                return
            try:
                await asyncio.wait_for(call_queue.redis.ping(), timeout=2.0)
                deps["redis"] = "up"
            except Exception as e:
                deps["redis"] = f"down ({type(e).__name__})"

        llm_headers = None
        if assistant.config.llm_api_key:
            llm_headers = {"Authorization": f"Bearer {assistant.config.llm_api_key}"}
        await asyncio.gather(
            probe_http("vllm", assistant.config.llm_base_url.rstrip("/") + "/models", llm_headers),
            probe_http("speaches", assistant.config.speaches_api_url.rstrip("/") + "/health"),
            probe_redis(),
        )
        _deps_cache["ts"] = now
        _deps_cache["deps"] = deps
        return deps

    def _require_verified_for(tool_name: str, call_id: Optional[str]) -> None:
        """Apply the VERIFY_REQUIRED_TOOLS gate to REST tool execution.

        The same check tool_manager.execute_tool applies to LLM-driven calls:
        a gated tool runs only for a call whose caller has passed VERIFY this
        call. Over REST the relevant session is the one named by ``call_id``
        (else the single active call); with no live verified session the tool
        is refused (fail closed) with 403.
        """
        session = None
        try:
            sessions = _active_call_sessions(assistant)
            if call_id:
                session = next((sess for sess in sessions
                                if _session_matches(sess, call_id)), None)
            elif len(sessions) == 1:
                session = sessions[0]
        except Exception:
            session = None
        blocked = assistant.tool_manager.verification_block(tool_name, session)
        if blocked is not None:
            log_event(logger, logging.INFO, f"REST tool {tool_name} blocked: caller not verified",
                      event="verify_gate", tool=tool_name, outcome="blocked", source="api")
            raise HTTPException(status_code=403, detail=blocked.message)

    @app.get("/health")
    async def health_check(deep: bool = False):
        """Health check endpoint.

        The default is a cheap process-level liveness check (safe for container
        healthchecks). With ``?deep=true`` it also probes vLLM, Speaches and
        Redis (results cached ~10s) and reports per-dependency status; overall
        status becomes "degraded" if any dependency is down. The agent is pure
        orchestration, so a dead backend means calls will fail even though the
        process itself is alive.
        """
        result = {
            "status": "healthy",
            "sip_registered": assistant.sip_handler._registered.is_set() if hasattr(assistant.sip_handler, '_registered') else False
        }
        if call_queue:
            try:
                result["queue"] = await call_queue.get_queue_status()
            except Exception as e:
                result["queue"] = {"error": type(e).__name__}
                result["status"] = "degraded"
        if deep:
            deps = await _probe_dependencies()
            result["dependencies"] = deps
            if any(v.startswith(("down", "error")) for v in deps.values()):
                result["status"] = "degraded"
        return result
    
    @app.get("/queue")
    async def queue_status():
        """Get call queue status."""
        if not call_queue:
            return {"enabled": False}
        
        status = await call_queue.get_queue_status()
        return {
            "enabled": True,
            **status
        }
    
    @app.post("/call", response_model=OutboundCallResponse, dependencies=protected)
    async def initiate_call(request: OutboundCallRequest):
        """
        Initiate an outbound notification call.
        
        The call will be made asynchronously. If callback_url is provided,
        results will be POSTed there when the call completes.
        
        Calls are queued and processed sequentially to prevent overwhelming the SIP system.
        
        Note: callback_url is required when using choice collection.
        
        Simple notification example:
        ```json
        {
            "message": "Hello, this is a reminder about your appointment.",
            "extension": "1001"
        }
        ```
        
        Choice collection example (requires callback_url):
        ```json
        {
            "message": "Hello, this is a reminder about your appointment tomorrow at 2pm.",
            "extension": "1001",
            "callback_url": "https://example.com/webhook",
            "choice": {
                "prompt": "Say yes to confirm or no to cancel.",
                "options": [
                    {"value": "confirmed", "synonyms": ["yes", "yeah", "yep", "confirm"]},
                    {"value": "cancelled", "synonyms": ["no", "nope", "cancel"]}
                ],
                "timeout_seconds": 15
            }
        }
        ```
        """
        try:
            call_id, position = await handler.initiate_call(request)
            return OutboundCallResponse(
                call_id=call_id,
                status=CallStatus.QUEUED,
                message=f"Call queued at position {position}" if position > 0 else "Call initiated",
                queue_position=position if position > 0 else None
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to initiate call: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to initiate call")
    
    @app.get("/call/{call_id}")
    async def get_call_status(call_id: str):
        """Get status of a call."""
        # Check queue first
        if call_queue:
            queued_call = await call_queue.get_call(call_id)
            if queued_call:
                return {
                    "call_id": call_id,
                    "status": queued_call.status.value,
                    "queued_at": queued_call.queued_at,
                    "started_at": queued_call.started_at,
                    "completed_at": queued_call.completed_at,
                    "error": queued_call.error
                }
        
        # Check pending calls (direct execution mode)
        if call_id in handler.pending_calls:
            return {
                "call_id": call_id,
                "status": "in_progress",
                "extension": handler.pending_calls[call_id].extension
            }
            
        # 200 with an explicit not_found status, NOT a 404: in direct (no-queue)
        # mode finished calls are removed from pending_calls, and existing
        # clients poll this endpoint until completion — a 404 would make them
        # treat a successfully completed call as an error.
        return {"call_id": call_id, "status": "not_found"}

    # Authenticated + rate-limited unlike the other read endpoints: a
    # transcript is a verbatim record of what a caller said (addresses, PINs,
    # order numbers), and call_ids are guessable (prefix-<unix_second>-<n>).
    @app.get("/call/{call_id}/transcript", dependencies=protected)
    async def get_call_transcript(call_id: str):
        """Return the conversation transcript for a call (live or finished)."""
        store = getattr(assistant, "transcripts", None)
        transcript = store.get(call_id) if store else None
        if transcript is None:
            raise HTTPException(status_code=404, detail=f"No transcript for call '{call_id}'")
        return transcript

    # ==========================================================================
    # Admin dashboard API (call summaries, live call, SSE event stream, page)
    # ==========================================================================

    # Auth-gated like the transcript endpoint: call summaries and the live
    # event stream expose who called and what was said.
    @app.get("/calls", dependencies=protected)
    async def list_calls():
        """Recent call summaries (live + the in-memory LRU), newest first."""
        store = getattr(assistant, "transcripts", None)
        return store.list_recent() if store else []

    def _session_summary(session) -> Dict[str, Any]:
        history = getattr(session, "conversation_history", []) or []
        return {
            "call_id": session.transcript_id,
            "caller": getattr(session.call_info, "remote_uri", "") or "",
            "direction": session.direction,
            "duration_seconds": round(time.time() - session.start_time, 1),
            "turns": len([m for m in history if m.get("role") == "user"]),
        }

    @app.get("/calls/active", dependencies=protected)
    async def get_active_calls():
        """Summaries of ALL live call sessions (several may be active with
        MAX_CONCURRENT_CALLS > 1). ``calls`` is empty when idle."""
        sessions = _active_call_sessions(assistant)
        calls = [_session_summary(s) for s in sessions]
        return {"active": bool(calls), "count": len(calls), "calls": calls}

    @app.post("/calls/active/hangup",
              dependencies=protected + [Depends(csrf_protect)])
    async def hangup_active_call(call_id: Optional[str] = None):
        """Hang up a live call. 404 when there is no active call; with 2+
        active calls a call_id is required (409 lists the active ids)."""
        sessions = _active_call_sessions(assistant)
        if not sessions:
            raise HTTPException(status_code=404, detail="No active call")
        if call_id:
            session = next(
                (s for s in sessions if _session_matches(s, call_id)), None)
            if session is None:
                raise HTTPException(status_code=404,
                                    detail=f"No active call '{call_id}'")
        elif len(sessions) > 1:
            raise HTTPException(status_code=409, detail={
                "error": "Multiple active calls; specify call_id",
                "active_call_ids": [s.transcript_id for s in sessions]})
        else:
            session = sessions[0]
        log_event(logger, logging.INFO, "Admin hangup of active call",
                 event="admin_hangup", call_id=session.transcript_id)
        await assistant.sip_handler.hangup_call(session.call_info)
        return {"success": True, "call_id": session.transcript_id}

    @app.get("/admin/events", dependencies=protected)
    async def admin_event_stream():
        """Server-Sent Events stream of live call/turn/tool events.

        Plain SSE frames (`data: {json}\\n\\n`) with a `: keepalive` comment
        every ~15s. Browsers' EventSource cannot send auth headers, so the
        dashboard consumes this with fetch() + a streaming reader instead —
        the wire format is standard SSE either way.
        """
        async def _stream():
            # Subscribe INSIDE the generator, as its first statement: if the
            # client aborts before the response body starts streaming, the
            # generator is never started and a never-started async generator's
            # `finally` block never runs — so subscribing eagerly in the
            # handler would leak the queue on EventBus._subscribers forever
            # (one 256-slot queue per aborted connect, fed on every publish).
            # Subscribing lazily means the abort-before-first-iteration path
            # never subscribes at all, and every path that DOES subscribe
            # reaches the `finally` below on disconnect.
            q = events_bus.subscribe()
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"data: {json.dumps(item)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                events_bus.unsubscribe(q)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # No auth on the page itself: it is a static shell containing no data —
    # every data endpoint it calls is auth-gated individually.
    @app.get("/admin", include_in_schema=False)
    async def admin_page():
        """Serve the self-contained operator dashboard page."""
        if not assistant.config.admin_ui_enabled:
            raise HTTPException(status_code=404, detail="Admin UI is disabled")
        page = Path(__file__).parent / "admin" / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="Admin page not found")
        return FileResponse(page, media_type="text/html")

    # Store handler reference for queue worker
    app.state.handler = handler
    
    # ==========================================================================
    # Tool Execution API
    # ==========================================================================
    
    @app.get("/tools", response_model=List[ToolInfo])
    async def list_tools():
        """
        List all available tools.
        
        Returns information about each tool including name, description, 
        parameters, and whether it's enabled.
        """
        tools = []
        for name, tool in assistant.tool_manager.tools.items():
            tools.append(ToolInfo(
                name=name,
                description=getattr(tool, 'description', ''),
                parameters=getattr(tool, 'parameters', {}),
                enabled=getattr(tool, 'enabled', True)
            ))
        return sorted(tools, key=lambda t: t.name)
    
    @app.get("/tools/{tool_name}", response_model=ToolInfo)
    async def get_tool(tool_name: str):
        """Get information about a specific tool."""
        tool = assistant.tool_manager.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        return ToolInfo(
            name=getattr(tool, 'name', tool_name),
            description=getattr(tool, 'description', ''),
            parameters=getattr(tool, 'parameters', {}),
            enabled=getattr(tool, 'enabled', True)
        )
    
    @app.post("/tools/{tool_name}/call", response_model=ToolCallResponse, dependencies=protected)
    async def tool_call(tool_name: str, request: ToolCallRequest):
        """
        Execute a tool and call someone with the result.
        
        This endpoint is perfect for webhooks - it executes a tool (like WEATHER),
        then places an outbound call to speak the result to the recipient.
        
        Examples:
        
        Weather alert call:
        ```json
        POST /tools/WEATHER/call
        {
            "tool": "WEATHER",
            "extension": "1001",
            "prefix": "Good morning! Here's your weather update.",
            "suffix": "Have a great day!"
        }
        ```
        
        Scheduled weather call (from cron/Home Assistant):
        ```bash
        curl -X POST http://sip-agent:8080/tools/WEATHER/call \\
          -H "Content-Type: application/json" \\
          -d '{"tool": "WEATHER", "extension": "5551234567"}'
        ```
        
        DateTime announcement:
        ```json
        POST /tools/DATETIME/call
        {
            "tool": "DATETIME",
            "params": {"format": "full"},
            "extension": "1001",
            "prefix": "Attention please."
        }
        ```
        
        With callback for confirmation:
        ```json
        {
            "tool": "WEATHER",
            "extension": "1001",
            "callback_url": "https://example.com/webhook/weather-call-complete"
        }
        ```

        With a spoken confirmation prompt (requires callback_url):
        ```json
        {
            "tool": "WEATHER",
            "extension": "1001",
            "callback_url": "https://example.com/webhook",
            "suffix": "Press or say yes if you heard this.",
            "choice": {
                "prompt": "Say yes to confirm.",
                "options": [{"value": "confirmed", "synonyms": ["yes", "yeah", "ok"]}]
            }
        }
        ```
        """
        actual_tool_name = tool_name.upper()

        # The path is authoritative; if a body `tool` is supplied it must agree.
        if request.tool and request.tool.upper() != actual_tool_name:
            raise HTTPException(
                status_code=400,
                detail=f"Body tool '{request.tool}' does not match path tool '{actual_tool_name}'"
            )

        # Get the tool
        tool = assistant.tool_manager.get_tool(actual_tool_name)
        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{actual_tool_name}' not found. Use GET /tools to list available tools."
            )

        log_event(logger, logging.INFO, f"Tool call request: {actual_tool_name} -> {request.extension}",
                 event="api_tool_call", tool=actual_tool_name, extension=request.extension)
        _require_verified_for(actual_tool_name, None)

        try:
            # Execute the tool first
            result = await tool.execute(request.params)
            tool_message = result.message

            if not tool_result_success(result):
                log_event(logger, logging.WARNING, f"Tool failed: {tool_message}",
                         event="api_tool_call_tool_failed", tool=actual_tool_name)
                return _tool_failed_response(actual_tool_name, tool_message)

            # Build the full message (prefix + tool result + suffix)
            full_message = _compose_message(request.prefix, tool_message, request.suffix)

            return await _place_composed_call(
                request, full_message,
                tool_label=actual_tool_name,
                tool_message=tool_message,
                event="api_tool_call_initiated",
                response_message=f"Calling {request.extension} with {actual_tool_name} result",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Tool call failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Tool call failed")
    
    @app.post("/tools/{tool_name}/execute", response_model=ToolExecuteResponse, dependencies=protected)
    async def execute_tool(tool_name: str, request: ToolExecuteRequest = None):
        """
        Execute a tool and optionally speak the result.
        
        This endpoint allows external systems (webhooks, home automation, etc.)
        to trigger tool execution. The result can optionally be spoken to 
        an active call.
        
        Examples:
        
        Get weather (just data):
        ```json
        POST /tools/WEATHER/execute
        {"tool": "WEATHER"}
        ```
        
        Get weather and speak to call:
        ```json
        POST /tools/WEATHER/execute
        {"tool": "WEATHER", "speak_result": true}
        ```
        
        Execute calculation:
        ```json
        POST /tools/CALC/execute
        {"tool": "CALC", "params": {"expression": "25 * 4"}}
        ```
        
        Set a timer and announce it:
        ```json
        POST /tools/SET_TIMER/execute
        {
            "tool": "SET_TIMER",
            "params": {"duration": 300, "message": "Pizza is ready!"},
            "speak_result": true
        }
        ```
        """
        # Use request body or default
        if request is None:
            request = ToolExecuteRequest()

        # The path is authoritative; if a body `tool` is supplied it must agree.
        actual_tool_name = tool_name.upper()
        if request.tool and request.tool.upper() != actual_tool_name:
            raise HTTPException(
                status_code=400,
                detail=f"Body tool '{request.tool}' does not match path tool '{actual_tool_name}'"
            )

        # Get the tool
        tool = assistant.tool_manager.get_tool(actual_tool_name)
        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{actual_tool_name}' not found. Use GET /tools to list available tools."
            )
        
        log_event(logger, logging.INFO, f"API executing tool: {actual_tool_name}",
                 event="api_tool_execute", tool=actual_tool_name, params=request.params)
        _require_verified_for(actual_tool_name, request.call_id)

        try:
            # Execute the tool
            result = await tool.execute(request.params)

            success = tool_result_success(result)

            response = ToolExecuteResponse(
                success=success,
                tool=actual_tool_name,
                message=result.message,
                data=getattr(result, 'data', None),
                spoken=False,
                error=None if success else result.message
            )
            
            # Speak result to active call if requested
            if request.speak_result and success and result.message:
                try:
                    spoken = await _speak_to_call(assistant, result.message, request.call_id)
                except AmbiguousActiveCall:
                    # The tool itself succeeded; ambiguity only means the
                    # result couldn't be spoken anywhere unambiguous.
                    spoken = False
                    log_event(logger, logging.WARNING,
                             "Multiple active calls; pass call_id to speak the result",
                             event="api_tool_ambiguous_call")
                response.spoken = spoken
                if not spoken:
                    log_event(logger, logging.WARNING, "No active call to speak to",
                             event="api_tool_no_call")
            
            log_event(logger, logging.INFO, f"Tool executed: {actual_tool_name}",
                     event="api_tool_complete", tool=actual_tool_name, success=success)
            
            return response

        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Tool execution failed")

    @app.post("/webhook/call", response_model=ToolCallResponse, dependencies=protected)
    async def webhook_call(request: WebhookCallRequest):
        """
        Generic webhook -> outbound call.

        A single flexible entry point for external webhooks (cron, Home
        Assistant, n8n, alerting systems). Place a call with a static
        ``message`` and/or a ``tool`` executed at call time; ``prefix``/
        ``suffix`` wrap the spoken body, and an optional ``choice`` collects a
        spoken response (POSTed to ``callback_url``).

        Static announcement (no tool):
        ```json
        POST /webhook/call
        {
            "extension": "1001",
            "message": "The garage door has been open for 20 minutes."
        }
        ```

        Tool-driven with confirmation:
        ```json
        POST /webhook/call
        {
            "extension": "1001",
            "tool": "WEATHER",
            "prefix": "Good morning!",
            "callback_url": "https://example.com/webhook",
            "choice": {
                "prompt": "Say yes if you're awake.",
                "options": [{"value": "awake", "synonyms": ["yes", "yeah", "yep"]}]
            }
        }
        ```

        Tool result plus a static message:
        ```json
        POST /webhook/call
        {
            "extension": "5551234567",
            "tool": "DATETIME",
            "message": "Don't forget your 9am meeting.",
            "prefix": "Heads up."
        }
        ```
        """
        tool_message = ""
        actual_tool_name = None

        try:
            body_parts = []

            # Execute the tool (if any) to produce part of the spoken body.
            if request.tool:
                actual_tool_name = request.tool.upper()
                tool = assistant.tool_manager.get_tool(actual_tool_name)
                if not tool:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tool '{actual_tool_name}' not found. Use GET /tools to list available tools."
                    )
                log_event(logger, logging.INFO, f"Webhook call: {actual_tool_name} -> {request.extension}",
                         event="api_webhook_call", tool=actual_tool_name, extension=request.extension)
                result = await tool.execute(request.params)
                tool_message = result.message
                if not tool_result_success(result):
                    log_event(logger, logging.WARNING, f"Webhook tool failed: {tool_message}",
                             event="api_webhook_call_tool_failed", tool=actual_tool_name)
                    return _tool_failed_response(actual_tool_name or "", tool_message)
                body_parts.append(tool_message)

            # Append the static message (if any) after any tool result.
            if request.message:
                body_parts.append(request.message)

            full_message = _compose_message(request.prefix, " ".join(body_parts), request.suffix)
            if not full_message:
                raise HTTPException(status_code=400, detail="Resulting message is empty")

            return await _place_composed_call(
                request, full_message,
                tool_label=actual_tool_name or "",
                tool_message=tool_message,
                event="api_webhook_call_initiated",
                response_message=f"Calling {request.extension}",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Webhook call failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Webhook call failed")

    @app.post("/speak", dependencies=protected)
    async def speak_message(message: str, call_id: Optional[str] = None,
                            reformat_for_speech: bool = False):
        """
        Speak a message to the active call.

        This is useful for external systems to inject announcements
        into an ongoing call.

        Query params:
        - message: The text to speak
        - call_id: Optional specific call ID (if multiple calls active)
        - reformat_for_speech: Rewrite the message into spoken form via the LLM
        """
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        message = await _maybe_reformat(assistant, message, reformat_for_speech)
        try:
            spoken = await _speak_to_call(assistant, message, call_id)
        except AmbiguousActiveCall as e:
            raise HTTPException(status_code=409, detail={
                "error": "Multiple active calls; specify call_id",
                "active_call_ids": e.call_ids})

        if spoken:
            return {"success": True, "message": "Message spoken to call"}
        else:
            raise HTTPException(status_code=404, detail="No active call to speak to")

    @app.post("/play", dependencies=protected)
    async def play_audio(request: Request, call_id: Optional[str] = None):
        """
        Play an uploaded audio file into the active call.

        Send the audio file's bytes as the raw request body (any format
        libsndfile can decode: WAV, FLAC, OGG; MP3 with libsndfile >= 1.1).
        The audio is decoded, downmixed to mono, resampled to the call rate,
        and queued on the same playlist player /speak uses.

        Query params:
        - call_id: Optional specific call ID (if multiple calls active)
        """
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Audio body is required")
        max_bytes = assistant.config.play_max_bytes
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Audio exceeds PLAY_AUDIO_MAX_BYTES ({max_bytes})")

        from audio_pipeline import decode_audio_to_pcm16
        sample_rate = assistant.config.sample_rate
        try:
            pcm = await asyncio.get_event_loop().run_in_executor(
                None, decode_audio_to_pcm16, data, sample_rate)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            played = await _play_to_call(assistant, pcm, call_id)
        except AmbiguousActiveCall as e:
            raise HTTPException(status_code=409, detail={
                "error": "Multiple active calls; specify call_id",
                "active_call_ids": e.call_ids})
        if not played:
            raise HTTPException(status_code=404, detail="No active call to play to")
        return {
            "success": True,
            "message": "Audio queued for playback",
            "duration_s": round(len(pcm) / (sample_rate * 2), 2),
        }

    # ==========================================================================
    # Scheduled Calls API
    # ==========================================================================
    
    @app.post("/schedule", response_model=ScheduledCallResponse, dependencies=protected)
    async def schedule_call(request: ScheduledCallRequest):
        """
        Schedule a call for a future time.
        
        You can schedule a call with either a static message or a tool that
        will be executed at call time (e.g., WEATHER for fresh data).
        
        Time can be specified as:
        - delay_seconds: Number of seconds from now
        - at_time: ISO datetime (2025-01-15T07:00:00) or HH:MM time (07:00)
        
        Examples:
        
        Wake-up weather call in 8 hours:
        ```json
        {
            "extension": "1001",
            "tool": "WEATHER",
            "delay_seconds": 28800,
            "prefix": "Good morning! Here's your weather."
        }
        ```
        
        Daily 7am weather call:
        ```json
        {
            "extension": "1001",
            "tool": "WEATHER",
            "at_time": "07:00",
            "timezone": "America/Los_Angeles",
            "prefix": "Good morning!",
            "recurring": "daily"
        }
        ```
        
        Reminder call at specific time:
        ```json
        {
            "extension": "5551234567",
            "message": "This is your reminder to take your medication.",
            "at_time": "2025-01-15T09:00:00",
            "timezone": "America/New_York"
        }
        ```
        
        Weekday morning briefing:
        ```json
        {
            "extension": "1001",
            "tool": "WEATHER",
            "at_time": "06:30",
            "recurring": "weekdays",
            "prefix": "Good morning! Time to wake up."
        }
        ```
        """
        import pytz
        from datetime import datetime, timedelta

        # Validate dial target and webhook up front. RequestRejected is an
        # HTTPException, so failures surface directly as 400s.
        validate_extension(request.extension, assistant.config)
        await validate_callback_url(request.callback_url, assistant.config)

        # Calculate delay
        delay_seconds = request.delay_seconds
        scheduled_time = None

        if request.at_time:
            try:
                tz = pytz.timezone(request.timezone or "America/Los_Angeles")
                now = datetime.now(tz)
                
                # Parse time - either full ISO or just HH:MM
                if 'T' in request.at_time or '-' in request.at_time:
                    # Full ISO datetime
                    if request.at_time.endswith('Z'):
                        scheduled_time = datetime.fromisoformat(request.at_time.replace('Z', '+00:00'))
                    else:
                        scheduled_time = datetime.fromisoformat(request.at_time)
                        if scheduled_time.tzinfo is None:
                            scheduled_time = tz.localize(scheduled_time)
                else:
                    # Just HH:MM - schedule for today or tomorrow
                    hour, minute = map(int, request.at_time.split(':'))
                    scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # If time already passed today, schedule for tomorrow
                    if scheduled_time <= now:
                        scheduled_time += timedelta(days=1)
                
                delay_seconds = int((scheduled_time - now).total_seconds())

            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid time format: {e}")
            except HTTPException:
                # Don't let the generic handler below re-wrap intentional errors.
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error parsing time: {e}")

            # Past-time check lives outside the parse try/except so its
            # HTTPException isn't re-wrapped as an "Error parsing time" message.
            if delay_seconds < 0:
                raise HTTPException(status_code=400, detail="Scheduled time is in the past")

        # Build the task data
        task_data = {
            "extension": request.extension,
            "message": request.message,
            "tool": request.tool,
            "tool_params": request.tool_params,
            "prefix": request.prefix,
            "suffix": request.suffix,
            "callback_url": request.callback_url,
            "recurring": request.recurring,
            "timezone": request.timezone,
            "at_time": request.at_time,  # Store for recurring
            # Reformat happens at execution time (tool output only exists then).
            "reformat_for_speech": request.reformat_for_speech,
        }
        
        # Schedule the task
        task_id = await assistant.tool_manager.schedule_task(
            task_type="scheduled_call",
            delay_seconds=delay_seconds,
            message=request.message or f"Scheduled {request.tool} call",
            target_uri=request.extension,
            metadata=task_data
        )
        
        # Calculate scheduled time for response
        if scheduled_time:
            scheduled_for = scheduled_time.isoformat()
        else:
            tz = pytz.timezone(request.timezone or "America/Los_Angeles")
            scheduled_for = (datetime.now(tz) + timedelta(seconds=delay_seconds)).isoformat()
        
        log_event(logger, logging.INFO, f"Scheduled call: {task_id} -> {request.extension}",
                 event="call_scheduled",
                 schedule_id=task_id,
                 extension=request.extension,
                 delay=delay_seconds,
                 tool=request.tool)
        
        return ScheduledCallResponse(
            schedule_id=task_id,
            status="scheduled",
            extension=request.extension,
            scheduled_for=scheduled_for,
            delay_seconds=delay_seconds,
            message=f"Call scheduled for {scheduled_for}",
            recurring=request.recurring
        )
    
    @app.get("/schedule", response_model=List[ScheduledCallInfo], dependencies=protected)
    async def list_scheduled_calls():
        """List all scheduled calls."""
        scheduled = []
        # execute_at is naive LOCAL_TIMEZONE wall-clock (the scheduler's
        # clock, Config.local_now) — never compare it against the container
        # clock or remaining_seconds is hours off when the two differ.
        now = assistant.config.local_now()

        for task_id, task in assistant.tool_manager.scheduled_tasks.items():
            if task.task_type == "scheduled_call":
                remaining = max(0, int((task.execute_at - now).total_seconds()))
                metadata = task.metadata or {}

                scheduled.append(ScheduledCallInfo(
                    schedule_id=task_id,
                    extension=metadata.get("extension", task.target_uri or ""),
                    scheduled_for=task.execute_at.isoformat(),
                    remaining_seconds=remaining,
                    message=metadata.get("message"),
                    tool=metadata.get("tool"),
                    recurring=metadata.get("recurring"),
                    status="pending" if not task.completed else "completed"
                ))

        return sorted(scheduled, key=lambda x: x.remaining_seconds)

    @app.get("/schedule/{schedule_id}", response_model=ScheduledCallInfo, dependencies=protected)
    async def get_scheduled_call(schedule_id: str):
        """Get details of a scheduled call."""
        task = assistant.tool_manager.scheduled_tasks.get(schedule_id)

        if not task or task.task_type != "scheduled_call":
            raise HTTPException(status_code=404, detail="Scheduled call not found")

        now = assistant.config.local_now()
        remaining = max(0, int((task.execute_at - now).total_seconds()))
        metadata = task.metadata or {}

        return ScheduledCallInfo(
            schedule_id=schedule_id,
            extension=metadata.get("extension", task.target_uri or ""),
            scheduled_for=task.execute_at.isoformat(),
            remaining_seconds=remaining,
            message=metadata.get("message"),
            tool=metadata.get("tool"),
            recurring=metadata.get("recurring"),
            status="pending" if not task.completed else "completed"
        )

    @app.delete("/schedule/{schedule_id}", dependencies=protected)
    async def cancel_scheduled_call(schedule_id: str):
        """Cancel a scheduled call."""
        # cancel_task also removes it from the persisted task file.
        if not assistant.tool_manager.cancel_task(schedule_id):
            raise HTTPException(status_code=404, detail="Scheduled call not found")
        
        log_event(logger, logging.INFO, f"Cancelled scheduled call: {schedule_id}",
                 event="call_schedule_cancelled", schedule_id=schedule_id)
        
        return {"success": True, "message": f"Scheduled call {schedule_id} cancelled"}

    # ==========================================================================
    # Virtual Numbers API (ephemeral inbound extensions)
    # ==========================================================================

    def _require_virtual_numbers():
        if not assistant.config.virtual_numbers_enabled:
            raise RequestRejected(
                403, "Virtual numbers are disabled (set VIRTUAL_NUMBERS_ENABLED=true)")

    def _virtual_number_response(entry) -> VirtualNumberResponse:
        return VirtualNumberResponse(
            id=entry.id,
            number=entry.number,
            sip_uri=f"sip:{entry.number}@{assistant.config.sip_domain}",
            status="claimed" if entry.claimed else "active",
            purpose=entry.purpose,
            expires_at=entry.expires_at,
            created_at=entry.created_at,
            persistent=entry.persistent,
            events=list(entry.events),
            callback_url=entry.callback_url,
        )

    @app.post("/virtual-numbers", response_model=VirtualNumberResponse,
              dependencies=protected)
    async def create_virtual_number(request: VirtualNumberRequest):
        """
        Create an ephemeral inbound extension.

        The agent listens for a call dialed to the returned number in the
        background. When the call arrives it is answered as the normal
        assistant with `purpose` injected as context (plus the optional custom
        greeting); when the call ends, the outcome and transcript are POSTed
        to `callback_url` and the number is cleared. Unused numbers expire
        after `ttl_s` (an "expired" webhook fires instead).

        With `persistent: true` the number becomes a **trigger number**: it
        never expires, survives its calls, and every call to it fires the
        webhooks selected in `events` — e.g. `["answered", "first_speech"]`
        kicks a workflow off as soon as the call lands and again with what
        the caller first said. Payloads carry `event: virtual_number.<name>`,
        `caller`, `call_id`, and for speech events `text`.

        Example:
        ```json
        {
            "purpose": "The caller is confirming pizza order #4211 for pickup.",
            "greeting": "Hi! Calling about your pizza order?",
            "ttl_s": 1800,
            "callback_url": "https://n8n.local/webhook/pizza-call"
        }
        ```
        """
        _require_virtual_numbers()
        await validate_callback_url(request.callback_url, assistant.config)

        from virtual_numbers import VirtualNumberError
        try:
            entry = assistant.virtual_numbers.create(
                number=request.number,
                ttl_s=request.ttl_s,
                purpose=request.purpose,
                greeting=request.greeting or "",
                callback_url=request.callback_url or "",
                include_transcript=request.include_transcript,
                persistent=request.persistent,
                events=request.events,
            )
        except VirtualNumberError as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail)

        return _virtual_number_response(entry)

    @app.get("/virtual-numbers", response_model=List[VirtualNumberResponse],
             dependencies=protected)
    async def list_virtual_numbers():
        """List active virtual numbers."""
        _require_virtual_numbers()
        return [_virtual_number_response(e)
                for e in assistant.virtual_numbers.list_active()]

    @app.get("/virtual-numbers/{number_id}", response_model=VirtualNumberResponse,
             dependencies=protected)
    async def get_virtual_number(number_id: str):
        """Get one virtual number (404 once consumed/expired/deleted)."""
        _require_virtual_numbers()
        entry = assistant.virtual_numbers.get(number_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Virtual number not found")
        return _virtual_number_response(entry)

    @app.delete("/virtual-numbers/{number_id}", dependencies=protected)
    async def delete_virtual_number(number_id: str):
        """Delete a virtual number before it is used (no webhook fires)."""
        _require_virtual_numbers()
        if not assistant.virtual_numbers.delete(number_id):
            raise HTTPException(status_code=404, detail="Virtual number not found")
        return {"success": True, "message": f"Virtual number {number_id} deleted"}

    # --- Identity verification -------------------------------------------
    def _verify_credentials_view(caller_id: str) -> VerifyCredentialsResponse:
        view = assistant.verify_store.public_view(caller_id)
        uri = assistant.verify_store.provisioning_uri(
            caller_id, assistant.config.verify_issuer)
        return VerifyCredentialsResponse(
            caller_id=caller_id, has_pin=view["has_pin"], has_totp=view["has_totp"],
            provisioning_uri=uri, updated_at=view.get("updated_at"))

    @app.post("/verify", response_model=VerifyResponse, dependencies=protected)
    async def verify_caller(request: VerifyRequest):
        """Check a caller's PIN and/or OTP out-of-band (no live call needed).

        OTP is tried first when supplied, then the PIN; `verified` is true if
        either matches. Credentials resolve per-caller, then global fallback.
        """
        caller_id = (request.caller_id or "").strip()
        verifier = assistant.verifier
        ok = False
        method: Optional[str] = None
        if request.otp and await verifier.averify_totp(caller_id, request.otp):
            ok, method = True, "otp"
        elif request.pin and await verifier.averify_pin(caller_id, request.pin):
            ok, method = True, "pin"
        return VerifyResponse(caller_id=caller_id, verified=ok, method=method)

    @app.post("/verify/call", response_model=VerifyCallResponse, dependencies=protected)
    async def verify_call(request: VerifyCallRequest):
        """Place an outbound call that verifies a caller by keypad.

        Dials the caller (``extension``, defaulting to ``caller_id``), asks them
        to key in their PIN or one-time code, checks it, and returns the verdict
        synchronously (also POSTed to ``callback_url`` when set). Returns 400
        when the caller has no verification credentials configured.
        """
        try:
            return await handler.run_verify_call(request)
        except RequestRejected as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail)

    @app.post("/verify/credentials", response_model=VerifyCredentialsResponse,
              dependencies=protected)
    async def set_verify_credentials(request: VerifyCredentialsRequest):
        """Enroll/update a caller's PIN and/or TOTP secret.

        Returns the enrollment metadata plus an otpauth:// provisioning URI when
        a per-caller TOTP secret exists (import into an authenticator app). The
        raw secret and PIN are never returned.
        """
        caller_id = (request.caller_id or "").strip()
        if not is_safe_caller_id(caller_id):
            raise HTTPException(status_code=400, detail="Invalid caller_id")
        if not (request.pin or request.totp_secret or request.generate_totp):
            raise HTTPException(
                status_code=400,
                detail="Provide a pin, totp_secret, or generate_totp=true")
        # PBKDF2 hashing is CPU-bound; keep it off the call-serving event loop.
        result = await asyncio.to_thread(
            assistant.verify_store.set_credentials,
            caller_id, pin=request.pin, totp_secret=request.totp_secret,
            generate_totp=request.generate_totp)
        if result is None:
            raise HTTPException(status_code=400, detail="Could not store credentials")
        return _verify_credentials_view(caller_id)

    @app.get("/verify/credentials/{caller_id}",
             response_model=VerifyCredentialsResponse, dependencies=protected)
    async def get_verify_credentials(caller_id: str):
        """Enrollment metadata for a caller (404 when none). Never the secret/PIN."""
        if not assistant.verify_store.get(caller_id):
            raise HTTPException(status_code=404, detail="No credentials for this caller")
        return _verify_credentials_view(caller_id)

    @app.delete("/verify/credentials/{caller_id}", dependencies=protected)
    async def delete_verify_credentials(caller_id: str):
        """Remove a caller's enrolled credentials."""
        if not assistant.verify_store.delete(caller_id):
            raise HTTPException(status_code=404, detail="No credentials for this caller")
        return {"success": True, "message": f"Credentials for {caller_id} deleted"}

    @app.get("/verify/otp/{caller_id}", response_model=OtpResponse,
             dependencies=protected)
    async def get_current_otp(caller_id: str):
        """Current TOTP code for an ENROLLED caller's own secret.

        The global VERIFY_TOTP_SECRET is served only under the reserved id
        ``global`` — never as a silent fallback for an unknown/typo'd caller,
        which would deliver the shared code to the wrong recipient.
        """
        if caller_id == GLOBAL_OTP_ID:
            if not getattr(assistant.config, "verify_totp_secret", ""):
                raise HTTPException(status_code=404, detail="No global TOTP secret configured")
        elif not is_safe_caller_id(caller_id):
            raise HTTPException(status_code=400, detail="Invalid caller_id")
        elif not assistant.verifier.has_own_totp_secret(caller_id):
            raise HTTPException(status_code=404, detail="No TOTP secret for this caller")
        result = assistant.verifier.current_otp(caller_id if caller_id != GLOBAL_OTP_ID else "")
        if result is None:
            raise HTTPException(status_code=404, detail="No TOTP secret for this caller")
        code, remaining = result
        return OtpResponse(caller_id=caller_id, otp=code, expires_in_s=remaining)

    return app


class AmbiguousActiveCall(Exception):
    """Several calls are active and no call_id was given to pick one.

    Endpoints translate this into a 409 carrying the active call ids."""

    def __init__(self, call_ids: List[str]):
        self.call_ids = call_ids
        super().__init__("Multiple active calls; specify call_id")


def _active_call_sessions(assistant: 'SIPAIAssistant') -> List[Any]:
    """All registered call sessions, newest last.

    Prefers the session registry (assistant.sessions); falls back to the
    single-session attribute for assistants without a registry (test
    doubles / older shims)."""
    sessions = getattr(assistant, "sessions", None)
    if sessions:
        return list(sessions.values())
    single = getattr(assistant, "session", None)
    return [single] if single is not None else []


def _session_matches(session: Any, call_id: str) -> bool:
    """True when ``call_id`` names this session (transcript id — the id the
    API exposes — or the underlying SIP-level call id)."""
    if not call_id:
        return False
    if getattr(session, "transcript_id", None) == call_id:
        return True
    info = getattr(session, "call_info", None)
    if info is None:
        return False
    return (getattr(info, "call_id", None) == call_id
            or getattr(info, "id", None) == call_id)


def _resolve_active_call(assistant: 'SIPAIAssistant', call_id: Optional[str] = None):
    """The target call's CallInfo, or None when idle / call_id mismatch.

    With several active sessions and no call_id, raises AmbiguousActiveCall
    so callers can answer 409 with the list of active call ids."""
    sessions = _active_call_sessions(assistant)
    if sessions:
        if call_id:
            for session in sessions:
                if _session_matches(session, call_id):
                    return session.call_info
            logger.debug(f"No active session matches call_id {call_id}")
            return None
        if len(sessions) > 1:
            raise AmbiguousActiveCall(
                [getattr(s, "transcript_id", "") or "" for s in sessions])
        return sessions[0].call_info

    # No session registry entries: legacy single current_call (compat shims
    # and test doubles that only set assistant.current_call).
    current_call = getattr(assistant, 'current_call', None)

    if not current_call:
        logger.debug("No current_call attribute on assistant")
        return None

    # If call_id specified, verify it matches
    if call_id:
        current_call_id = getattr(current_call, 'call_id', None) or getattr(current_call, 'id', None)
        if current_call_id != call_id:
            logger.debug(f"Call ID mismatch: {current_call_id} != {call_id}")
            return None

    return current_call


async def _speak_to_call(assistant: 'SIPAIAssistant', message: str, call_id: Optional[str] = None) -> bool:
    """
    Speak a message to an active call.

    Returns True if message was spoken, False if no active call. Raises
    AmbiguousActiveCall when several calls are active and no call_id picks one.
    """
    current_call = _resolve_active_call(assistant, call_id)
    if not current_call:
        return False
    try:

        # Generate TTS
        audio_data = await assistant.audio_pipeline.synthesize(message)
        if not audio_data:
            logger.error("Failed to synthesize speech")
            return False

        # Send audio to call (send_audio requires the active CallInfo)
        await assistant.sip_handler.send_audio(current_call, audio_data)

        log_event(logger, logging.INFO, f"Spoke message to call: {message[:50]}...",
                 event="api_speak_success")

        return True

    except Exception as e:
        logger.error(f"Failed to speak to call: {e}", exc_info=True)
        return False


async def _play_to_call(assistant: 'SIPAIAssistant', pcm: bytes, call_id: Optional[str] = None) -> bool:
    """
    Play already-decoded PCM audio into an active call.

    Returns True if audio was queued, False if no active call. Raises
    AmbiguousActiveCall when several calls are active and no call_id picks one.
    """
    current_call = _resolve_active_call(assistant, call_id)
    if not current_call:
        return False
    try:
        await assistant.sip_handler.send_audio(current_call, pcm)

        log_event(logger, logging.INFO,
                  f"Queued {len(pcm)} bytes of uploaded audio to call",
                  event="api_play_success")
        return True

    except Exception as e:
        logger.error(f"Failed to play audio to call: {e}", exc_info=True)
        return False