"""
Callback Tool Plugin
====================
Schedule a callback call to the current caller or a specified number.

Usage in conversation:
User: "Call me back in 5 minutes"
LLM: [TOOL:CALLBACK:delay=300]

User: "Schedule a callback to 555-1234 in 1 hour"
LLM: [TOOL:CALLBACK:delay=3600,destination=sip:5551234@example.com]
"""

import logging
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)


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
            "description": "SIP URI or phone number to call (optional, defaults to caller)",
            "required": False
        }
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        delay = int(params.get('delay', 60))
        message = params.get('message', 'This is your scheduled callback')
        uri = params.get('uri')
        destination = params.get('destination')
        
        # Use provided URI/destination, or fall back to caller's number
        target = uri or destination
        
        # If no target specified, use the current caller's number
        if not target and self.assistant.current_call:
            target = getattr(self.assistant.current_call, 'remote_uri', None)
            logger.info(f"No callback number specified, using caller: {target}")
            
        if not target:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="No callback number available"
            )
            
        # Schedule callback
        task_id = await self.assistant.tool_manager.schedule_task(
            task_type="callback",
            delay_seconds=delay,
            message=message,
            target_uri=target
        )
        
        log_event(logger, logging.INFO, f"Callback scheduled: {delay}s to {target}",
                 event="callback_scheduled", delay=delay, uri=target, task_id=task_id)
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"I'll call you back in {delay} seconds",
            data={"task_id": task_id, "delay": delay, "uri": target}
        )
