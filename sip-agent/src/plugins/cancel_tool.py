"""
Cancel Tool Plugin
==================
Cancel pending timers and callbacks.

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
        task_type = params.get('task_type', 'all')
        
        cancelled = await self.assistant.tool_manager.cancel_tasks(task_type)
        
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
