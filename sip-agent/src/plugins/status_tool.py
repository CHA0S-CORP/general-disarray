"""
Status Tool Plugin
==================
Check status of pending timers and callbacks.

Usage in conversation:
User: "Do I have any timers running?"
LLM: [TOOL:STATUS]
"""

from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus


class StatusTool(BaseTool):
    """Get status of scheduled tasks."""
    
    name = "STATUS"
    description = "Check status of pending timers and scheduled callbacks"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode
    
    parameters = {}  # No parameters needed
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        pending = self.assistant.tool_manager.get_pending_tasks()
        
        if not pending:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message="You have no pending timers or callbacks"
            )
            
        messages = []
        for task in pending:
            # execute_at is on the scheduler's LOCAL_TIMEZONE wall clock.
            remaining = (task.execute_at - self.config.local_now()).total_seconds()
            if remaining > 0:
                # Format remaining time nicely
                if remaining < 60:
                    time_str = f"{int(remaining)} seconds"
                elif remaining < 3600:
                    mins = int(remaining / 60)
                    time_str = f"{mins} minute{'s' if mins != 1 else ''}"
                else:
                    hours = int(remaining / 3600)
                    mins = int((remaining % 3600) / 60)
                    time_str = f"{hours} hour{'s' if hours != 1 else ''}"
                    if mins > 0:
                        time_str += f" and {mins} minute{'s' if mins != 1 else ''}"
                
                if task.task_type == "timer":
                    messages.append(f"Timer in {time_str}")
                else:
                    messages.append(f"Callback in {time_str}")
                    
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=". ".join(messages) if messages else "No pending tasks",
            data={"pending_count": len(pending)}
        )
