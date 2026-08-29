"""
Verify Tool Plugin
==================
Prove the caller's identity mid-call with a static PIN and/or a rolling TOTP
code, entered over the phone keypad (DTMF) — never spoken, so the code never
lands in the STT transcript.

The LLM invokes this when a caller needs to authenticate before a sensitive
action (see config.verify_required_tools, enforced in tool_manager). On success
the per-call flag session.verified flips to True, which is injected into the
system prompt and read by tool-gating.

    User: "I need to transfer money."  (a gated action)
    LLM:  [TOOL:VERIFY]
    -> tool prompts "Please enter your code, then press pound", collects DTMF,
       checks it against the caller's PIN/TOTP, and reports success/failure.

Credential resolution (per-caller then global) and the actual checks live in
identity_verification.IdentityVerifier; this tool only drives the call-side
interaction. Fail-open on infrastructure errors, but a failed check leaves the
caller unverified (fail closed on the security decision).
"""

import logging
from typing import Any, Dict, Optional

from tool_plugins import BaseTool, ToolResult, ToolStatus
from caller_memory import caller_id_from_uri
from dtmf_collect import collect_dtmf_code
from logging_utils import log_event

logger = logging.getLogger(__name__)


class VerifyTool(BaseTool):
    """Collect a PIN/OTP over DTMF and verify the caller's identity."""

    name = "VERIFY"
    description = (
        "Verify the caller's identity when they need to authenticate before a "
        "sensitive action. Prompts them to key in their PIN or one-time code on "
        "the phone keypad and checks it. Call this with no arguments to accept "
        "either factor; the digits are entered by keypad, so do not ask the "
        "caller to say their code aloud.")
    enabled = True
    speak_result = True  # the success/failure result is spoken to the caller

    parameters = {
        "method": {
            "type": "string",
            "description": ("Which factor to require: 'pin', 'otp', or 'auto' "
                            "(default — accepts either)."),
            "enum": ["pin", "otp", "auto"],
            "required": False,
            "default": "auto",
        },
    }

    def __init__(self, assistant):
        super().__init__(assistant)
        # Self-gate: honored by tool_manager alongside the explicit config check.
        if self.config and not getattr(self.config, "enable_verify_tool", True):
            self.enabled = False

    def _session(self):
        return getattr(self.assistant, "session", None) if self.assistant else None

    def _verifier(self):
        return getattr(self.assistant, "verifier", None) if self.assistant else None

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        session = self._session()
        call_info = getattr(session, "call_info", None) if session else None
        if session is None or call_info is None:
            return ToolResult(status=ToolStatus.FAILED,
                              message="I can only verify your identity during a call.")

        verifier = self._verifier()
        if verifier is None:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Identity verification isn't available right now.")

        if session.verified:
            return ToolResult(status=ToolStatus.SUCCESS,
                              message="You're already verified — go ahead.")

        caller_id = caller_id_from_uri(getattr(call_info, "remote_uri", "") or "")
        if not caller_id or not verifier.can_verify(caller_id):
            # No PIN/secret for this caller and no global factor: the feature is
            # effectively off. Say so benignly rather than implying a failure.
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I don't have identity verification set up for this number.")

        max_attempts = int(getattr(self.config, "verify_max_attempts", 3))
        if session.verify_attempts >= max_attempts:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="That's too many attempts. I can't verify you on this call.")

        method = str(params.get("method") or "auto").lower()
        if method not in ("pin", "otp", "auto"):
            method = "auto"

        code = await self._collect_digits(session, call_info)
        if not code:
            # A timeout / empty entry isn't a wrong code — don't burn an attempt.
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I didn't get a code. Let me know when you'd like to try again.")

        ok, used = await verifier.averify(caller_id, code, method=method)
        # NEVER log or return the entered code — keep it out of transcripts/logs.
        if ok:
            session.verified = True
            log_event(logger, logging.INFO, "Caller verified",
                      event="verify", outcome="ok", caller=caller_id, method=used)
            return ToolResult(status=ToolStatus.SUCCESS,
                              message="Thank you — your identity is verified.",
                              data={"verified": True, "method": used})

        session.verify_attempts += 1
        remaining = max(0, max_attempts - session.verify_attempts)
        log_event(logger, logging.INFO, "Caller verification failed",
                  event="verify", outcome="failed", caller=caller_id,
                  attempts=session.verify_attempts)
        if remaining:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="That code wasn't right. You can try again when you're ready.",
                data={"verified": False})
        return ToolResult(
            status=ToolStatus.FAILED,
            message="That code wasn't right, and that was the last attempt.",
            data={"verified": False})

    async def _collect_digits(self, session, call_info) -> Optional[str]:
        """Prompt and collect a keypad code (shared loop in dtmf_collect).

        Uses the configured VERIFY_CALL_PROMPT (pre-cached at startup, so the
        prompt plays instantly and isn't re-synthesized per attempt). While
        collecting, ``session.dtmf_collecting`` suppresses barge-in in the audio
        loop and pauses the agentic engine's wall clock, so the caller's own
        keypad tones (or an "okay") can't cancel the turn mid-entry.
        """
        sip = self.assistant.sip_handler
        prompt_audio = None
        try:
            prompt_audio = await self.assistant.audio_pipeline.synthesize(
                self.config.verify_call_prompt)
        except Exception as e:
            logger.debug(f"verify prompt synthesis failed: {e}")

        timeout = float(getattr(self.config, "verify_dtmf_timeout_s", 20.0))
        interdigit = float(getattr(self.config, "verify_dtmf_interdigit_s", 3.0))
        session.dtmf_collecting = True
        try:
            return await collect_dtmf_code(
                sip, call_info, timeout=timeout, interdigit=interdigit,
                prompt_audio=prompt_audio)
        finally:
            session.dtmf_collecting = False
