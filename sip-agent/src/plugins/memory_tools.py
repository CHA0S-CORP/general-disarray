"""
Memory Tools Plugin
===================
Let the caller explicitly manage their own cross-call memory (the same
CallerMemoryStore that auto-extracts facts after each call). REMEMBER saves
one fact; FORGET removes any saved facts matching a phrase. Both refresh the
live system-prompt block so the change is visible for the rest of this call.

Usage in conversation:
User: "Remember that I prefer morning callbacks."
LLM: [TOOL:REMEMBER:fact=Prefers morning callbacks]
User: "Forget what you know about my callbacks."
LLM: [TOOL:FORGET:what=callbacks]
"""

import logging
from typing import Any, Dict, Optional, Tuple

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event
from caller_memory import caller_id_from_uri

logger = logging.getLogger(__name__)


class _CallerMemoryToolBase(BaseTool):
    """Shared guards for the caller-memory tools."""

    def __init__(self, assistant):
        super().__init__(assistant)
        if self.config and not self.config.caller_memory_enabled:
            self.enabled = False
            logger.info(f"{self.name} tool disabled - caller memory is disabled")

    def _resolve_caller(self) -> Tuple[Optional[str], Optional[Any],
                                       Optional[ToolResult]]:
        """Resolve (caller_id, store) for the live call, or a FAILED result."""
        session = getattr(self.assistant, "session", None) if self.assistant else None
        if session is None:
            return None, None, ToolResult(
                status=ToolStatus.FAILED,
                message="I can only do that during a call.")
        store = getattr(self.assistant, "caller_memory", None)
        if store is None or not (self.config and self.config.caller_memory_enabled):
            return None, None, ToolResult(
                status=ToolStatus.FAILED,
                message="Memory is not enabled.")
        remote_uri = getattr(session.call_info, "remote_uri", "")
        caller_id = caller_id_from_uri(remote_uri)
        if caller_id is None:
            return None, None, ToolResult(
                status=ToolStatus.FAILED,
                message="I do not know who I am speaking with.")
        return caller_id, store, None

    def _refresh_prompt(self, store: Any, caller_id: str) -> None:
        """Re-render the caller-memory prompt block so THIS call sees the
        change immediately (it is normally loaded once at call start)."""
        try:
            self.assistant.session.caller_memory_prompt = \
                store.format_for_prompt(caller_id)
        except Exception as e:
            logger.warning(f"Could not refresh caller memory prompt: {e}")


class RememberTool(_CallerMemoryToolBase):
    """Save one fact to the caller's cross-call memory."""

    name = "REMEMBER"
    description = ("Save a fact about the caller to long-term memory so it is "
                   "remembered on future calls")
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "fact": {
            "type": "string",
            "description": "The fact to remember, as one short sentence",
            "required": True,
        }
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        caller_id, store, failure = self._resolve_caller()
        if failure is not None:
            return failure

        fact = str(params.get("fact") or "").strip()
        if not fact:
            return ToolResult(status=ToolStatus.FAILED,
                              message="I need something to remember.")

        try:
            ok = store.add_fact(caller_id, fact)
        except Exception as e:
            logger.error(f"REMEMBER failed for {caller_id}: {e}", exc_info=True)
            ok = False
        if not ok:
            return ToolResult(status=ToolStatus.FAILED,
                              message="I could not save that.",
                              data={"caller": caller_id, "fact": fact})

        self._refresh_prompt(store, caller_id)
        log_event(logger, logging.INFO,
                  f"Remembered fact for {caller_id}",
                  event="caller_memory_tool", outcome="remembered",
                  caller=caller_id)
        return ToolResult(status=ToolStatus.SUCCESS,
                          message="Got it. I will remember that.",
                          data={"caller": caller_id, "fact": fact})


class ForgetTool(_CallerMemoryToolBase):
    """Remove matching facts from the caller's cross-call memory."""

    name = "FORGET"
    description = ("Erase remembered facts about the caller that mention a "
                   "given word or phrase")
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "what": {
            "type": "string",
            "description": "A word or phrase identifying the facts to forget",
            "required": True,
        }
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        caller_id, store, failure = self._resolve_caller()
        if failure is not None:
            return failure

        what = str(params.get("what") or "").strip()
        if not what:
            return ToolResult(status=ToolStatus.FAILED,
                              message="I need to know what to forget.")

        try:
            removed = store.remove_facts(caller_id, what)
        except Exception as e:
            logger.error(f"FORGET failed for {caller_id}: {e}", exc_info=True)
            return ToolResult(status=ToolStatus.FAILED,
                              message="I could not update my memory.",
                              data={"caller": caller_id, "what": what})

        if removed == 0:
            # Nothing matched — the memory is already in the requested state,
            # so this is a success, not a failure.
            return ToolResult(status=ToolStatus.SUCCESS,
                              message="I did not have anything about that.",
                              data={"caller": caller_id, "what": what,
                                    "removed": 0})

        self._refresh_prompt(store, caller_id)
        log_event(logger, logging.INFO,
                  f"Forgot {removed} fact(s) for {caller_id}",
                  event="caller_memory_tool", outcome="forgotten",
                  caller=caller_id, removed=removed)
        if removed == 1:
            message = "Done - I have forgotten that."
        else:
            message = f"Done - I have forgotten {removed} things about that."
        return ToolResult(status=ToolStatus.SUCCESS,
                          message=message,
                          data={"caller": caller_id, "what": what,
                                "removed": removed})
