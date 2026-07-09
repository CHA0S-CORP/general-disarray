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

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel, Field, model_validator

from telemetry import create_span, Metrics
from logging_utils import log_event
from retry_utils import retry_async, RetryError

if TYPE_CHECKING:
    from main import SIPAIAssistant
    from call_queue import CallQueue

logger = logging.getLogger(__name__)


# ============================================================================
# Request validation / security helpers
# ============================================================================

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


class CallStatus(str, Enum):
    """Status of an outbound call."""
    QUEUED = "queued"
    RINGING = "ringing"
    ANSWERED = "answered"
    COMPLETED = "completed"
    NO_ANSWER = "no_answer"
    FAILED = "failed"
    BUSY = "busy"


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
                call_info = await self.assistant.sip_handler.make_call(extension)
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
                if self.assistant.config.amd_enabled:
                    machine_answered = await self._detect_answering_machine(call_info)
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
                        call_id, call_info, request.choice, choice_audio
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

    async def _detect_answering_machine(self, call_info) -> bool:
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

            if chunk and self.assistant.audio_pipeline.has_speech(chunk):
                continuous_ms += cfg.chunk_duration_ms
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
        choice_audio: bytes
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
                timeout=choice.timeout_seconds
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
        
    async def _listen_for_response(self, call_info, timeout: float) -> Optional[tuple]:
        """Listen for a spoken response or a DTMF keypress.

        Returns ("speech", transcription) or ("dtmf", digit), or None on
        timeout/hangup. DTMF wins whenever a digit is buffered — it's an
        unambiguous signal, unlike STT.
        """
        start_time = asyncio.get_event_loop().time()
        get_dtmf = getattr(self.assistant.sip_handler, 'get_dtmf_digit', None)

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
                    transcription = await self.assistant.audio_pipeline.process_audio(audio_chunk)
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
                spoken = await _speak_to_call(assistant, result.message, request.call_id)
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
        spoken = await _speak_to_call(assistant, message, call_id)
        
        if spoken:
            return {"success": True, "message": "Message spoken to call"}
        else:
            raise HTTPException(status_code=404, detail="No active call to speak to")
    
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
        # execute_at is a datetime, so compare against wall-clock now (not the
        # event-loop monotonic clock) and format it directly.
        now = datetime.now()

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

        now = datetime.now()
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
    
    return app


async def _speak_to_call(assistant: 'SIPAIAssistant', message: str, call_id: Optional[str] = None) -> bool:
    """
    Speak a message to an active call.
    
    Returns True if message was spoken, False if no active call.
    """
    try:
        # Check if there's an active call
        current_call = getattr(assistant, 'current_call', None)
        
        if not current_call:
            logger.debug("No current_call attribute on assistant")
            return False
        
        # If call_id specified, verify it matches
        if call_id:
            current_call_id = getattr(current_call, 'call_id', None) or getattr(current_call, 'id', None)
            if current_call_id != call_id:
                logger.debug(f"Call ID mismatch: {current_call_id} != {call_id}")
                return False
        
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