"""
Cancel Tool Plugin
==================
Cancel the current caller's own pending timers and callbacks.

Usage in conversation:
User: "Cancel my timer"
LLM: [TOOL:CANCEL:task_type=timer]

User: "Cancel everything"
LLM: [TOOL:CANCEL:task_type=all]
"""

from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus


class CancelTool(BaseTool):
    """Cancel scheduled tasks."""
    
    name = "CANCEL"
    description = "Cancel pending timers or scheduled callbacks"
    enabled = True
    
    parameters = {
        "task_type": {
            "type": "string",
            "description": "Type of task to cancel: 'timer', 'callback', or 'all'",
            "required": False,
            "default": "all"
        }
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        task_type = str(params.get('task_type') or 'all').strip().lower()
        if task_type not in ('timer', 'callback', 'all'):
            task_type = 'all'

        # Scoped to the current caller's own timers/callbacks: a caller can
        # never cancel another caller's tasks or REST-scheduled calls.
        cancelled = await self.assistant.tool_manager.cancel_tasks(
            task_type, owned_only=True)
        
        if cancelled == 0:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message="No tasks to cancel"
            )
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"Cancelled {cancelled} {'task' if cancelled == 1 else 'tasks'}",
            data={"cancelled_count": cancelled}
        )
