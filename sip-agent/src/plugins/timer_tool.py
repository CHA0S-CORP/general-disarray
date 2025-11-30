"""
Timer Tool Plugin
=================
Set timers and reminders that trigger during or after the call.

Usage in conversation:
User: "Set a timer for 5 minutes"
LLM: [TOOL:SET_TIMER:duration=300,message=Your timer is complete]
"""

import logging
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event, format_duration

logger = logging.getLogger(__name__)


class TimerTool(BaseTool):
    """Set timers and reminders."""
    
    name = "SET_TIMER"
    description = "Set a timer or reminder for a specified duration"
    enabled = True
    
    parameters = {
        "duration": {
            "type": "integer",
            "description": "Duration in seconds",
            "required": True
        },
        "message": {
            "type": "string",
            "description": "Message to speak when timer completes",
            "required": False,
            "default": "Your timer is complete"
        }
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        duration = int(params.get('duration', 300))  # Default 5 minutes
        message = params.get('message', 'Your timer is complete')
        
        # Validate duration
        max_duration = self.config.max_timer_duration_hours * 3600
        if duration > max_duration:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Timer duration exceeds maximum of {self.config.max_timer_duration_hours} hours"
            )
            
        if duration < 1:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Timer duration must be at least 1 second"
            )
            
        # Schedule the timer
        task_id = await self.assistant.tool_manager.schedule_task(
            task_type="timer",
            delay_seconds=duration,
            message=message,
            target_uri=None  # Timer plays on current call
        )
        
        log_event(logger, logging.INFO, f"Timer set: {duration}s",
                 event="timer_set", duration=duration, message=message, task_id=task_id)
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"Timer set for {format_duration(duration)}",
            data={"task_id": task_id, "duration": duration}
        )
