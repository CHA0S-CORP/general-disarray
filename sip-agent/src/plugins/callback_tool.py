"""
Callback Tool Plugin
====================
Schedule a callback call to the current caller or a specified number.

Usage in conversation:
User: "Call me back in 5 minutes"
LLM: [TOOL:CALLBACK:delay=300]

User: "Schedule a callback to 555-1234 in 1 hour"
LLM: [TOOL:CALLBACK:delay=3600,destination=5551234]

All policy lives HERE (not in the tool manager) so every entry point — the
LLM path, REST ``/tools/CALLBACK/execute`` and ``/webhook/call`` — gets it:

- no destination / ``CALLER_NUMBER`` / the caller's own number -> the live
  caller's own URI (always allowed);
- any other destination must pass ``check_voice_dial_allowed`` (REST dial
  policy + VOICE_DIAL_ALLOW/DENY_PATTERN) — toll-fraud protection;
- ``delay`` must lie in [0, CALLBACK_MAX_DELAY_S];
- at most CALLBACK_MAX_PER_CALL callbacks per live call.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event, format_duration
from plugins.helpers import check_voice_dial_allowed, normalize_dial_target

logger = logging.getLogger(__name__)

# session.tool_state key: callbacks scheduled during this call.
_COUNT_KEY = "callbacks_scheduled"


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _caller_number(caller_uri: Optional[str]) -> str:
    """Digits of the caller URI's user part ('"Bob" <sip:+1555@h>' -> '1555')."""
    m = re.search(r"sips?:([^@;>\s]+)", caller_uri or "")
    return _digits(m.group(1) if m else (caller_uri or ""))


def is_callers_own_number(requested: str, caller_uri: Optional[str]) -> bool:
    """Whether a requested destination names the live caller's own number.

    A raw SIP URI only counts when it is exactly the caller's URI (a URI at
    another domain stays subject to policy). A bare number counts when its
    digits equal the caller's, allowing a dropped country/trunk prefix
    ("555-123-4567" for sip:+15551234567@...). A match always dials the
    caller's own URI, never the requested string, so a loose match can only
    ever call the caller back.
    """
    if not requested or not caller_uri:
        return False
    if requested.strip() == caller_uri.strip():
        return True
    requested = normalize_dial_target(requested)
    if requested.lower().startswith(("sip:", "sips:")) or "@" in requested:
        return False
    want, own = _digits(requested), _caller_number(caller_uri)
    if not want or not own:
        return False
    return want == own or (len(want) >= 7 and own.endswith(want))


class CallbackTool(BaseTool):
    """Schedule a callback call."""

    name = "CALLBACK"
    description = "Schedule a callback call. If no destination specified, calls back the current caller."
    enabled = True

    parameters = {
        "delay": {
            "type": "integer",
            "description": "Delay in seconds before callback",
            "required": False,
            "default": 60
        },
        "message": {
            "type": "string",
            "description": "Message to speak on callback",
            "required": False,
            "default": "This is your scheduled callback"
        },
        "destination": {
            "type": "string",
            "description": "Phone number/extension to call (optional, defaults to caller)",
            "required": False
        }
    }

    def _parse_delay(self, raw: Any) -> Tuple[Optional[int], Optional[str]]:
        try:
            delay = int(float(raw))
        except (TypeError, ValueError):
            return None, "I couldn't understand when to call you back."
        if delay < 0:
            return None, "I can't schedule a callback in the past."
        max_delay = int(getattr(self.config, "callback_max_delay_s", 86400))
        if delay > max_delay:
            return None, (f"I can only schedule callbacks up to "
                          f"{format_duration(max_delay)} ahead.")
        return delay, None

    def _callbacks_this_call(self, session, owner_call_id: Optional[str]) -> int:
        state = getattr(session, "tool_state", None)
        if isinstance(state, dict):
            return int(state.get(_COUNT_KEY, 0))
        if owner_call_id:
            return sum(1 for t in self.assistant.tool_manager.scheduled_tasks.values()
                       if t.task_type == "callback"
                       and getattr(t, "owner_call_id", None) == owner_call_id)
        return 0

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        delay, error = self._parse_delay(params.get("delay", 60))
        if error:
            return ToolResult(status=ToolStatus.FAILED, message=error)
        message = str(params.get("message") or "This is your scheduled callback")

        requested = params.get("uri") or params.get("destination")
        requested = str(requested).strip() if requested else ""

        call = getattr(self.assistant, "current_call", None)
        caller_uri = getattr(call, "remote_uri", None) if call else None

        if not requested or requested.upper() == "CALLER_NUMBER" \
                or is_callers_own_number(requested, caller_uri):
            target = caller_uri
            if not target:
                return ToolResult(
                    status=ToolStatus.FAILED,
                    message="No callback number available - please specify a number")
            if requested:
                logger.info(f"Callback destination is the caller's own number: {target}")
        else:
            target = normalize_dial_target(requested)
            policy_error = check_voice_dial_allowed(target, self.config)
            if policy_error:
                log_event(logger, logging.WARNING,
                          f"CALLBACK destination rejected by policy: {policy_error}",
                          event="callback_destination_blocked",
                          destination=requested, reason=policy_error)
                return ToolResult(
                    status=ToolStatus.FAILED,
                    message=("I can only call you back at your own number."
                             if caller_uri else "I can't call that number."))

        tm = self.assistant.tool_manager
        owner_call_id, owner_caller = tm.current_owner()
        session = getattr(self.assistant, "session", None)

        # Per-call cap: only meaningful when a live call is asking.
        if call is not None:
            limit = int(getattr(self.config, "callback_max_per_call", 3))
            if self._callbacks_this_call(session, owner_call_id) >= limit:
                log_event(logger, logging.WARNING,
                          f"CALLBACK refused: per-call limit of {limit} reached",
                          event="callback_limit_reached", limit=limit)
                return ToolResult(
                    status=ToolStatus.FAILED,
                    message=f"I can only schedule {limit} callbacks per call.")

        task_id = await tm.schedule_task(
            task_type="callback",
            delay_seconds=delay,
            message=message,
            target_uri=target,
            owner_call_id=owner_call_id,
            owner_caller=owner_caller,
        )
        state = getattr(session, "tool_state", None)
        if call is not None and isinstance(state, dict):
            state[_COUNT_KEY] = int(state.get(_COUNT_KEY, 0)) + 1

        log_event(logger, logging.INFO, f"Callback scheduled: {delay}s to {target}",
                  event="callback_scheduled", delay=delay, uri=target, task_id=task_id)

        when = "shortly" if delay == 0 else f"in {format_duration(delay)}"
        spoken = (f"I'll call you back {when}" if target == caller_uri
                  else f"I'll call {requested} {when}")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=spoken,
            data={"task_id": task_id, "delay": delay, "uri": target}
        )
