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
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import HANGUP_DELAY_SECONDS

logger = logging.getLogger(__name__)


class HangupTool(BaseTool):
    """End the current call."""
    
    name = "HANGUP"
    description = "End the current call gracefully"
    enabled = True
    
    parameters = {}  # No parameters needed
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        if self.assistant.current_call:
            try:
                # Schedule hangup after a short delay to allow goodbye message to play
                async def delayed_hangup():
                    await asyncio.sleep(HANGUP_DELAY_SECONDS)
                    if self.assistant.current_call:
                        await self.assistant.sip_handler.hangup_call(self.assistant.current_call)
                        logger.info("Call ended via HANGUP tool")
                        
                asyncio.create_task(delayed_hangup())
                
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
