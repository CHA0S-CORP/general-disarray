"""
Hangup Tool Plugin
==================
End the current call gracefully.

Usage in conversation:
User: "Goodbye"
LLM: Goodbye! [TOOL:HANGUP]
"""

import asyncio
import logging
from typing import Any, Dict, Set

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import HANGUP_DELAY_SECONDS

logger = logging.getLogger(__name__)


class HangupTool(BaseTool):
    """End the current call."""

    name = "HANGUP"
    description = "End the current call gracefully"
    enabled = True

    parameters = {}  # No parameters needed

    # Strong references to in-flight delayed-hangup tasks so they aren't
    # garbage-collected mid-flight (which would silently swallow exceptions).
    _pending_tasks: Set["asyncio.Task[None]"] = set()

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        # Bind the call we were asked to end. The delayed task must NOT
        # re-read current_call later: this call can end during the delay and
        # a different caller can already be live, and hangup_call() drops
        # whatever CallInfo it is handed.
        target_call = self.assistant.current_call
        if target_call:
            try:
                # Schedule hangup after a short delay to allow goodbye message to play
                async def delayed_hangup():
                    try:
                        await asyncio.sleep(HANGUP_DELAY_SECONDS)
                        if self.assistant.current_call is target_call:
                            await self.assistant.sip_handler.hangup_call(target_call)
                            logger.info("Call ended via HANGUP tool")
                        else:
                            logger.info(
                                "Skipping delayed hangup: the call it was "
                                "scheduled for is no longer active")
                    except Exception as e:
                        # A failure here happens after execute() has already
                        # returned, so it must be logged here or it is lost.
                        logger.error(f"Delayed hangup failed: {e}")

                task = asyncio.create_task(delayed_hangup())
                # Keep a strong reference until the task completes so the
                # event loop doesn't drop it (and its exceptions) early.
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)

                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    message="Ending call"
                )
            except Exception as e:
                logger.error(f"Hangup error: {e}")
                return ToolResult(
                    status=ToolStatus.FAILED,
                    message=f"Failed to end call: {e}"
                )
                
        return ToolResult(
            status=ToolStatus.FAILED,
            message="No active call to end"
        )
