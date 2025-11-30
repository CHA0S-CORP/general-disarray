"""
Tool Manager
============
Manages callable tools for the AI assistant.
All tools are loaded as plugins from the plugins/ directory.

To add a new tool, simply drop a Python file in the plugins/ directory.
See plugins/README.md for documentation.
"""

import json
import time
import uuid
import asyncio
import logging

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from main import SIPAIAssistant

from config import Config
from telemetry import create_span, Metrics
from logging_utils import log_event, format_duration, HANGUP_DELAY_SECONDS

# Import plugin system base classes
from tool_plugins import (
    BaseTool as PluginBaseTool,
    ToolResult as PluginToolResult,
    ToolStatus as PluginToolStatus,
)

# Alias for backwards compatibility and internal use
BaseTool = PluginBaseTool

logger = logging.getLogger(__name__)


# Re-export for backwards compatibility and internal use
class ToolStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    status: ToolStatus
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class ScheduledTask:
    """A scheduled task (timer or callback)."""
    id: str
    task_type: str
    execute_at: datetime
    message: str
    target_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False


class ToolManager:
    """
    Manages all tools and scheduled tasks.
    
    Tools are imported directly from the plugins directory.
    """
    
    def __init__(self, assistant: 'SIPAIAssistant'):
        self.assistant = assistant
        self.config = assistant.config
        self.tools: Dict[str, Any] = {}  # name -> PluginToolWrapper
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self._task_runner: Optional[asyncio.Task] = None
        
        # Load tools
        self._load_tools()
        
    def _load_tools(self):
        """Load all tool plugins."""
        # Import tool classes directly
        from plugins.timer_tool import TimerTool
        from plugins.callback_tool import CallbackTool
        from plugins.hangup_tool import HangupTool
        from plugins.weather_tool import WeatherTool
        from plugins.status_tool import StatusTool
        from plugins.cancel_tool import CancelTool
        from plugins.joke_tool import JokeTool
        from plugins.datetime_tool import DateTimeTool
        from plugins.calc_tool import CalculatorTool
        
        # All available tool classes
        tool_classes = [
            TimerTool,
            CallbackTool,
            HangupTool,
            WeatherTool,
            StatusTool,
            CancelTool,
            JokeTool,
            DateTimeTool,
            CalculatorTool,
        ]
        
        for tool_class in tool_classes:
            try:
                name = tool_class.name
                wrapper = self._create_plugin_wrapper(tool_class)
                
                # Check if tool should be enabled based on config
                if not self._should_enable_tool(name, wrapper):
                    logger.info(f"Skipping disabled tool: {name}")
                    continue
                
                self.tools[name] = wrapper
                logger.info(f"Loaded tool: {name}")
                
            except Exception as e:
                logger.error(f"Failed to load tool {tool_class}: {e}", exc_info=True)
                
        logger.info(f"Loaded {len(self.tools)} tools")
            
    def _should_enable_tool(self, name: str, wrapper) -> bool:
        """Check if a tool should be enabled based on configuration."""
        # Check tool-specific config flags
        if name == "SET_TIMER" and not self.config.enable_timer_tool:
            return False
        if name == "CALLBACK" and not self.config.enable_callback_tool:
            return False
        if name == "WEATHER" and not self.config.enable_weather_tool:
            return False
            
        # Check if tool disabled itself (e.g., missing API keys)
        if hasattr(wrapper, '_plugin_instance'):
            if not getattr(wrapper._plugin_instance, 'enabled', True):
                return False
                
        return True
            
    def _create_plugin_wrapper(self, plugin_class):
        """Create a wrapper that adapts a plugin tool to the local interface."""
        assistant = self.assistant
        
        class PluginToolWrapper:
            """Wrapper for plugin-based tools."""
            
            def __init__(wrapper_self):
                wrapper_self._plugin_instance = plugin_class(assistant)
                wrapper_self.name = plugin_class.name
                wrapper_self.description = plugin_class.description
                wrapper_self.enabled = getattr(plugin_class, 'enabled', True)
                wrapper_self.parameters = getattr(plugin_class, 'parameters', {})
                
            async def execute(wrapper_self, params: Dict[str, Any]) -> ToolResult:
                # Validate params if the plugin has validation
                if hasattr(wrapper_self._plugin_instance, 'validate_params'):
                    error = wrapper_self._plugin_instance.validate_params(params)
                    if error:
                        return ToolResult(
                            status=ToolStatus.FAILED,
                            message=error
                        )
                
                # Execute the plugin
                result = await wrapper_self._plugin_instance.execute(params)
                
                # Convert plugin result to local ToolResult
                status_map = {
                    PluginToolStatus.SUCCESS: ToolStatus.SUCCESS,
                    PluginToolStatus.FAILED: ToolStatus.FAILED,
                    PluginToolStatus.PENDING: ToolStatus.PENDING,
                }
                
                return ToolResult(
                    status=status_map.get(result.status, ToolStatus.FAILED),
                    message=result.message,
                    data=result.data
                )
                
            def validate_params(wrapper_self, params: Dict[str, Any]) -> Optional[str]:
                if hasattr(wrapper_self._plugin_instance, 'validate_params'):
                    return wrapper_self._plugin_instance.validate_params(params)
                return None
                
            def get_prompt_description(wrapper_self) -> str:
                """Get description for the system prompt."""
                if hasattr(wrapper_self._plugin_instance, 'get_prompt_description'):
                    return wrapper_self._plugin_instance.get_prompt_description()
                    
                # Generate description from parameters
                name = wrapper_self.name
                desc = wrapper_self.description
                
                if not wrapper_self.parameters:
                    return f"- {name}: [TOOL:{name}] - {desc}"
                    
                # Build parameter examples
                param_examples = []
                for param_name, param_spec in wrapper_self.parameters.items():
                    required = param_spec.get("required", False)
                    param_type = param_spec.get("type", "string")
                    default = param_spec.get("default", "")
                    
                    if param_type == "integer":
                        example = "NUMBER"
                    elif param_type == "number":
                        example = "NUMBER"
                    elif param_type == "boolean":
                        example = "true/false"
                    else:
                        example = "TEXT"
                    
                    if required:
                        param_examples.append(f"{param_name}={example}")
                    else:
                        param_examples.append(f"{param_name}={example} (optional)")
                        
                params_str = ",".join(param_examples)
                return f"- {name}: [TOOL:{name}:{params_str}] - {desc}"
        
        return PluginToolWrapper()
        
    def reload_plugins(self) -> int:
        """
        Reload all plugin tools from the plugins directory.
        
        This allows adding new tools without restarting the service.
        Returns the number of plugins loaded.
        """
        # Clear all tools
        old_count = len(self.tools)
        self.tools.clear()
        logger.info(f"Cleared {old_count} existing tools")
        
        # Reload plugins
        self._load_plugins()
        
        return len(self.tools)
        
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their descriptions."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "enabled": getattr(tool, 'enabled', True),
                "parameters": getattr(tool, 'parameters', {})
            }
            for tool in self.tools.values()
        ]
        
    def get_tools_prompt(self) -> str:
        """
        Generate the tools section for the system prompt.
        
        This is dynamically generated based on loaded plugins.
        """
        if not self.tools:
            logger.warning("No tools loaded - tools prompt will be empty")
            return ""
            
        lines = [
            "TOOLS:",
            "You can use tools by including them in your response. Format: [TOOL:NAME] or [TOOL:NAME:param=value,param2=value2]",
            ""
        ]
        
        # Sort tools by name for consistent ordering
        for name in sorted(self.tools.keys()):
            tool = self.tools[name]
            if hasattr(tool, 'get_prompt_description'):
                lines.append(tool.get_prompt_description())
            else:
                lines.append(f"- {name}: {tool.description}")
                
        lines.append("")
        lines.append("Use tools when helpful. Speak the result to the user naturally.")
        
        prompt = "\n".join(lines)
        logger.debug(f"Generated tools prompt with {len(self.tools)} tools")
        return prompt
        
    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        return self.tools.get(name.upper())
        
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name.upper() in self.tools
            
    async def start(self):
        """Start the task runner."""
        self._task_runner = asyncio.create_task(self._run_scheduler())
        logger.info("Tool manager started")
        
    async def stop(self):
        """Stop the task runner."""
        if self._task_runner:
            self._task_runner.cancel()
            try:
                await self._task_runner
            except asyncio.CancelledError:
                pass
        logger.info("Tool manager stopped")
        
    async def execute_tool(self, tool_call) -> ToolResult:
        """Execute a tool call with interception."""
        tool_name = tool_call.name.upper()
        start_time = time.time()
        
        # Log tool invocation (convert params to simple dict for JSON)
        try:
            params_dict = {k: str(v) for k, v in tool_call.params.items()}
        except:
            params_dict = {}
        log_event(logger, logging.INFO, f"Tool called: {tool_name}",
                 event="tool_call", tool=tool_name, params=params_dict)
        
        # Record tool call metric
        Metrics.record_tool_call(tool_name)
        
        if tool_name not in self.tools:
            Metrics.record_tool_error(tool_name, "unknown_tool")
            return ToolResult(status=ToolStatus.FAILED, message=f"Unknown tool: {tool_name}")
            
        tool = self.tools[tool_name]
        if not tool.enabled:
            Metrics.record_tool_error(tool_name, "disabled")
            return ToolResult(status=ToolStatus.FAILED, message=f"Tool {tool_name} disabled")

        # Validate base params (delay, message)
        error = tool.validate_params(tool_call.params)
        if error:
            Metrics.record_tool_error(tool_name, "validation_error")
            return ToolResult(status=ToolStatus.FAILED, message=error)
            
        with create_span(f"tool.{tool_name.lower()}", {
            "tool.name": tool_name,
            "tool.params": str(params_dict)
        }) as span:
            try:
                # --- INTERCEPT CALLBACK TOOL ---
                # Handle callback manually to ensure caller's number is used by default
                if tool_name == "CALLBACK":
                    delay = int(tool_call.params.get("delay", 60))
                    message = tool_call.params.get("message", "This is your scheduled callback")
                    destination = tool_call.params.get("destination") or tool_call.params.get("uri")
                    
                    # Sanitize destination
                    if destination:
                        destination = str(destination).strip()
                    
                    # Use caller's number if not specified or if explicitly "CALLER_NUMBER"
                    if not destination or destination.upper() == "CALLER_NUMBER":
                        if self.assistant.current_call:
                            destination = getattr(self.assistant.current_call, 'remote_uri', None)
                            logger.info(f"Using caller's number for callback: {destination}")
                        if not destination:
                            latency_ms = (time.time() - start_time) * 1000
                            Metrics.record_tool_latency(latency_ms, tool_name)
                            Metrics.record_tool_error(tool_name, "no_callback_number")
                            span.set_attribute("tool.error", "no_callback_number")
                            return ToolResult(
                                status=ToolStatus.FAILED,
                                message="No callback number available - please specify a number"
                            )
                    
                    logger.debug(f"Processing CALLBACK: delay={delay}, dest={destination}")
                    
                    # Schedule the callback
                    await self.assistant.schedule_callback(delay, message, destination)
                    
                    latency_ms = (time.time() - start_time) * 1000
                    Metrics.record_tool_latency(latency_ms, tool_name)
                    span.set_attribute("tool.latency_ms", latency_ms)
                    span.set_attribute("tool.status", "success")
                    
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        message=f"I'll call you back in {format_duration(delay)}"
                    )
                # -------------------------------

                # For other tools, run normally
                result = await tool.execute(tool_call.params)
                
                latency_ms = (time.time() - start_time) * 1000
                Metrics.record_tool_latency(latency_ms, tool_name)
                span.set_attribute("tool.latency_ms", latency_ms)
                span.set_attribute("tool.status", result.status.value)
                
                if result.status == ToolStatus.FAILED:
                    Metrics.record_tool_error(tool_name, "execution_failed")
                
                return result
                
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                import traceback
                traceback.print_exc()
                latency_ms = (time.time() - start_time) * 1000
                Metrics.record_tool_latency(latency_ms, tool_name)
                Metrics.record_tool_error(tool_name, type(e).__name__)
                span.record_exception(e)
                return ToolResult(status=ToolStatus.FAILED, message=str(e))
            
    async def schedule_task(
        self,
        task_type: str,
        delay_seconds: int,
        message: str,
        target_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Schedule a task for later execution."""
        task_id = str(uuid.uuid4())[:8]
        
        task = ScheduledTask(
            id=task_id,
            task_type=task_type,
            execute_at=datetime.now() + timedelta(seconds=delay_seconds),
            message=message,
            target_uri=target_uri,
            metadata=metadata or {}
        )
        
        self.scheduled_tasks[task_id] = task
        log_event(logger, logging.INFO, f"Task scheduled: {task_type} in {delay_seconds}s",
                 event="task_scheduled", task_id=task_id, task_type=task_type, 
                 delay=delay_seconds, target=str(target_uri) if target_uri else None)
        
        return task_id
        
    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all pending (not completed) tasks."""
        now = datetime.now()
        return [
            task for task in self.scheduled_tasks.values()
            if not task.completed and task.execute_at > now
        ]
        
    async def cancel_tasks(self, task_type: str = 'all') -> int:
        """Cancel tasks by type. Returns number cancelled."""
        cancelled = 0
        to_remove = []
        
        for task_id, task in self.scheduled_tasks.items():
            if not task.completed:
                if task_type == 'all' or task.task_type == task_type:
                    to_remove.append(task_id)
                    cancelled += 1
                    
        for task_id in to_remove:
            del self.scheduled_tasks[task_id]
            
        return cancelled
        
    async def _run_scheduler(self):
        """Background task that executes scheduled tasks."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                now = datetime.now()
                
                for task_id, task in list(self.scheduled_tasks.items()):
                    if task.completed:
                        continue
                        
                    if now >= task.execute_at:
                        await self._execute_scheduled_task(task)
                        task.completed = True
                        
                # Cleanup old tasks
                self._cleanup_old_tasks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                
    async def _execute_scheduled_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        log_event(logger, logging.INFO, f"Executing task: {task.id} ({task.task_type})",
                 event="task_execute", task_id=task.id, task_type=task.task_type)
        
        try:
            if task.task_type == "timer":
                await self._execute_timer(task)
            elif task.task_type == "callback":
                await self._execute_callback(task)
            elif task.task_type == "scheduled_call":
                await self._execute_scheduled_call(task)
            else:
                logger.warning(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            
    async def _execute_timer(self, task: ScheduledTask):
        """Execute a timer - speak the message on current call."""
        log_event(logger, logging.INFO, f"Timer fired: {task.message}",
                 event="timer_fired", task_id=task.id, message=task.message)
        if self.assistant.current_call and self.assistant.current_call.is_active:
            # Use streaming if available for consistent voice
            if hasattr(self.assistant, '_stream_response'):
                await self.assistant._stream_response(self.assistant.current_call, task.message)
            else:
                await self.assistant._speak(task.message)
        else:
            logger.warning(f"Timer {task.id} expired but no active call")
            
    async def _execute_callback(self, task: ScheduledTask):
        """Execute a callback - make outbound call."""
        if not task.target_uri:
            logger.error(f"Callback {task.id} has no target URI")
            return
            
        log_event(logger, logging.INFO, f"Executing callback to {task.target_uri}",
                 event="callback_execute", task_id=task.id, uri=task.target_uri)
        
        # Make the call
        for attempt in range(self.config.callback_retry_attempts):
            try:
                await self.assistant.make_outbound_call(
                    task.target_uri,
                    task.message
                )
                log_event(logger, logging.INFO, f"Callback completed: {task.id}",
                         event="callback_complete", task_id=task.id)
                return
            except Exception as e:
                logger.warning(f"Callback attempt {attempt + 1} failed: {e}")
                if attempt < self.config.callback_retry_attempts - 1:
                    await asyncio.sleep(self.config.callback_retry_delay_s)
                    
        logger.error(f"Callback {task.id} failed after {self.config.callback_retry_attempts} attempts")

    async def _execute_scheduled_call(self, task: ScheduledTask):
        """Execute a scheduled call - optionally run a tool, then make outbound call."""
        metadata = task.metadata or {}
        extension = metadata.get("extension") or task.target_uri
        
        if not extension:
            logger.error(f"Scheduled call {task.id} has no extension")
            return
        
        log_event(logger, logging.INFO, f"Executing scheduled call to {extension}",
                 event="scheduled_call_execute", task_id=task.id, extension=extension)
        
        # Build the message
        message_parts = []
        
        # Add prefix
        if metadata.get("prefix"):
            message_parts.append(metadata["prefix"])
        
        # Execute tool if specified
        tool_name = metadata.get("tool")
        if tool_name:
            tool = self.get_tool(tool_name)
            if tool:
                try:
                    tool_params = metadata.get("tool_params", {})
                    result = await tool.execute(tool_params)
                    
                    if result.status.value == "success" if hasattr(result.status, 'value') else str(result.status).lower() == "success":
                        message_parts.append(result.message)
                        log_event(logger, logging.INFO, f"Tool {tool_name} executed for scheduled call",
                                 event="scheduled_call_tool_success", tool=tool_name)
                    else:
                        logger.warning(f"Tool {tool_name} failed: {result.message}")
                        message_parts.append(f"I was unable to get the {tool_name.lower()} information.")
                except Exception as e:
                    logger.error(f"Tool {tool_name} error: {e}")
                    message_parts.append(f"I encountered an error getting the {tool_name.lower()} information.")
            else:
                logger.warning(f"Tool {tool_name} not found for scheduled call")
                message_parts.append(metadata.get("message", "This is your scheduled call."))
        elif metadata.get("message"):
            message_parts.append(metadata["message"])
        
        # Add suffix
        if metadata.get("suffix"):
            message_parts.append(metadata["suffix"])
        
        # Combine message
        full_message = " ".join(message_parts) if message_parts else "This is your scheduled call."
        
        # Make the call
        for attempt in range(self.config.callback_retry_attempts):
            try:
                await self.assistant.make_outbound_call(extension, full_message)
                
                log_event(logger, logging.INFO, f"Scheduled call completed: {task.id}",
                         event="scheduled_call_complete", task_id=task.id, extension=extension)
                
                # Handle recurring
                if metadata.get("recurring"):
                    await self._reschedule_recurring_call(task, metadata)
                
                # Send callback webhook if specified
                if metadata.get("callback_url"):
                    await self._send_scheduled_call_webhook(task, metadata, "completed")
                
                return
                
            except Exception as e:
                logger.warning(f"Scheduled call attempt {attempt + 1} failed: {e}")
                if attempt < self.config.callback_retry_attempts - 1:
                    await asyncio.sleep(self.config.callback_retry_delay_s)
        
        logger.error(f"Scheduled call {task.id} failed after {self.config.callback_retry_attempts} attempts")
        
        if metadata.get("callback_url"):
            await self._send_scheduled_call_webhook(task, metadata, "failed")
    
    async def _reschedule_recurring_call(self, task: ScheduledTask, metadata: dict):
        """Reschedule a recurring call."""
        import pytz
        
        recurring = metadata.get("recurring")
        if not recurring:
            return
        
        tz = pytz.timezone(metadata.get("timezone", "America/Los_Angeles"))
        now = datetime.now(tz)
        next_time = None
        
        if recurring == "daily":
            # Same time tomorrow
            next_time = now + timedelta(days=1)
        elif recurring == "weekdays":
            # Next weekday (Mon-Fri)
            next_time = now + timedelta(days=1)
            while next_time.weekday() >= 5:  # Saturday=5, Sunday=6
                next_time += timedelta(days=1)
        elif recurring == "weekends":
            # Next weekend day
            next_time = now + timedelta(days=1)
            while next_time.weekday() < 5:
                next_time += timedelta(days=1)
        else:
            # TODO: Support cron expressions
            logger.warning(f"Unsupported recurring pattern: {recurring}")
            return
        
        # If at_time was specified, use that time on the next day
        at_time = metadata.get("at_time")
        if at_time and ':' in at_time and 'T' not in at_time:
            hour, minute = map(int, at_time.split(':'))
            next_time = next_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        delay_seconds = int((next_time - now).total_seconds())
        
        # Schedule next occurrence
        new_task_id = await self.schedule_task(
            task_type="scheduled_call",
            delay_seconds=delay_seconds,
            message=task.message,
            target_uri=metadata.get("extension"),
            metadata=metadata
        )
        
        log_event(logger, logging.INFO, f"Rescheduled recurring call: {new_task_id}",
                 event="scheduled_call_rescheduled", 
                 task_id=new_task_id, 
                 recurring=recurring,
                 next_time=next_time.isoformat())
    
    async def _send_scheduled_call_webhook(self, task: ScheduledTask, metadata: dict, status: str):
        """Send webhook for scheduled call completion."""
        import httpx
        
        url = metadata.get("callback_url")
        if not url:
            return
        
        payload = {
            "schedule_id": task.id,
            "status": status,
            "extension": metadata.get("extension"),
            "tool": metadata.get("tool"),
            "recurring": metadata.get("recurring"),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                logger.info(f"Scheduled call webhook sent: {url}")
        except Exception as e:
            logger.error(f"Scheduled call webhook failed: {e}")

    async def schedule_callback(self, delay_seconds: int, message: str, target_uri: str) -> str:
        """
        Bridge method required by main.py to schedule callbacks.
        """
        # The internal scheduler expects (task_type, delay, message, TARGET_URI)
        return await self.schedule_task(
            task_type="callback",
            delay_seconds=delay_seconds,
            message=message,
            target_uri=target_uri # Pass URI correctly here
        )
        
    def _cleanup_old_tasks(self):
        """Remove completed tasks older than 1 hour."""
        cutoff = datetime.now() - timedelta(hours=1)
        to_remove = [
            task_id for task_id, task in self.scheduled_tasks.items()
            if task.completed and task.execute_at < cutoff
        ]
        for task_id in to_remove:
            del self.scheduled_tasks[task_id]


# Convenience function for creating custom tools
def create_custom_tool(
    name: str,
    description: str,
    handler: Callable,
    assistant: 'SIPAIAssistant'
) -> BaseTool:
    """Factory function to create custom tools."""
    
    class CustomTool(BaseTool):
        def __init__(self, name: str, description: str, handler: Callable, assistant):
            super().__init__(assistant)
            self.name = name
            self.description = description
            self._handler = handler
            
        async def execute(self, params: Dict[str, Any]) -> ToolResult:
            return await self._handler(params)
            
    return CustomTool(name, description, handler, assistant)