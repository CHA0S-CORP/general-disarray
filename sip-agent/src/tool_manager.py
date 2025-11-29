"""
Tool Manager
============
Manages callable tools for the AI assistant.
Includes timer, callback, and extensible tool framework.
"""

import json
import uuid
import asyncio
import logging
import httpx

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from main import SIPAIAssistant

from config import Config

logger = logging.getLogger(__name__)


def log_event(log, level, msg, event=None, **data):
    """Helper to log structured events."""
    extra = {}
    if event:
        extra['event_type'] = event
    if data:
        extra['event_data'] = data
    log.log(level, msg, extra=extra)


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


class BaseTool(ABC):
    """Base class for all tools."""
    
    name: str = "base_tool"
    description: str = "Base tool"
    enabled: bool = True
    
    def __init__(self, assistant: 'SIPAIAssistant'):
        self.assistant = assistant
        
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
        
    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters. Return error message if invalid."""
        return None


class TimerTool(BaseTool):
    """
    Set timers and reminders.
    """
    
    name = "SET_TIMER"
    description = "Set a timer or reminder"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        duration = int(params.get('duration', 300))  # Default 5 minutes
        message = params.get('message', 'Your timer is complete')
        
        # Validate duration
        max_duration = self.assistant.config.max_timer_duration_hours * 3600
        if duration > max_duration:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Timer duration exceeds maximum of {self.assistant.config.max_timer_duration_hours} hours"
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
            message=f"Timer set for {self._format_duration(duration)}",
            data={"task_id": task_id, "duration": duration}
        )
        
    def _format_duration(self, seconds: int) -> str:
        """Format duration for speech."""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
            if minutes:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            return " and ".join(parts)


class CallbackTool(BaseTool):
    """
    Schedule a callback call.
    If no number is specified, calls back the current caller.
    """
    
    name = "CALLBACK"
    description = "Schedule a callback call. If no destination specified, calls back the current caller."
    
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


class HangupTool(BaseTool):
    """End the current call."""
    
    name = "HANGUP"
    description = "End the current call"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        # Actually hang up the call
        if self.assistant.current_call:
            try:
                # Schedule hangup after a short delay to allow goodbye message to play
                async def delayed_hangup():
                    await asyncio.sleep(3)  # Wait for TTS to finish
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


class WeatherTool(BaseTool):
    """Get current weather from Tempest weather station."""
    
    name = "WEATHER"
    description = "Get current weather conditions from the local weather station"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        config = self.assistant.config
        station_id = config.tempest_station_id
        api_token = config.tempest_api_token
        
        logger.info(f"WeatherTool executing - station_id={station_id}, token={'set' if api_token else 'not set'}")
        
        if not station_id or not api_token:
            logger.warning("Weather station not configured")
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Weather station not configured"
            )
        
        try:
            # Tempest API - use token as query parameter
            url = f"https://swd.weatherflow.com/swd/rest/observations/station/{station_id}"
            params_dict = {"token": api_token}
            
            logger.info(f"Fetching weather from: {url}")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params_dict)
                logger.info(f"Weather API response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Weather API error: {response.status_code} - {response.text}")
                    return ToolResult(
                        status=ToolStatus.FAILED,
                        message="Couldn't reach the weather station"
                    )
                
                data = response.json()
            
            # Parse the observation data
            station = data.get("station_name", {})
            obs_list = data.get("obs", [])
            if not obs_list:
                logger.warning(f"No observations in response: {data}")
                return ToolResult(
                    status=ToolStatus.FAILED,
                    message="No weather data available"
                )
            
            obs = obs_list[0] if isinstance(obs_list, list) else obs_list
            logger.debug(f"Weather observation: {obs}")
            
            # Extract key weather values
            # Tempest API returns values in metric, we'll convert to imperial for speech
            temp_c = obs.get("air_temperature")
            humidity = obs.get("relative_humidity")
            wind_avg = obs.get("wind_avg")  # m/s
            wind_gust = obs.get("wind_gust")  # m/s
            wind_dir = obs.get("wind_direction")
            precip_rate = obs.get("precip")  # mm/min for last minute
            precip_accum_local_day = obs.get("precip_accum_local_day")  # mm/min for last minute
            precip_accum_last_1hr = obs.get("precip_accum_last_1hr")  # mm/min for last minute
            uv = obs.get("uv")
            feels_like = obs.get("feels_like")
            lightning_strike_count_last_1hr = obs.get("lightning_strike_count_last_1hr")
            lightning_strike_last_distance = obs.get("lightning_strike_last_distance")
            pressure_trend = obs.get("pressure_trend")
            barometric_pressure = obs.get("barometric_pressure")
            solar_radiation = obs.get("solar_radiation")
            
            
            # Convert to imperial
            temp_f = round(temp_c * 9/5 + 32) if temp_c is not None else None
            feels_like_f = round(feels_like * 9/5 + 32) if feels_like is not None else None
            wind_mph = round(wind_avg * 2.237) if wind_avg is not None else None
            wind_gust_mph = round(wind_gust * 2.237) if wind_gust is not None else None
            lightning_strike_last_distance_miles = round(lightning_strike_last_distance * 0.621371) if lightning_strike_last_distance is not None else None
            
            # Wind direction to cardinal
            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            wind_cardinal = directions[int((wind_dir + 11.25) / 22.5) % 16] if wind_dir is not None else None
            
            # Build natural language response
            parts = []

            if solar_radiation > 0:
                sun = "sunny" if solar_radiation > 800 else "partly sunny" if solar_radiation > 400 else "cloudy"
            else:
                sun = "dark"

            prefix = f"The current weather at {station} is: {sun} with a temperature of"
            
            if temp_f is not None:
                if feels_like_f == temp_f:
                    parts.append(f"{prefix} {temp_f} degrees Ferenheit")
                else:
                    parts.append(f"{prefix} {temp_f} degrees Ferenheit, and feels like {feels_like_f}")
        
            if humidity is not None:
                parts.append(f"the humidity is {round(humidity)}%")
            
            if wind_mph is not None:
                wind_desc = f"wind from the {wind_cardinal} at {wind_mph} miles per hour"
                if wind_gust_mph >0:
                    wind_desc += f", gusting to {wind_gust_mph} miles per hour"
                if wind_mph == 0:
                    wind_desc = f"Wind is currently calm"
                if wind_mph >= 15:
                    wind_desc += ", be advised it is quite windy"
                parts.append(wind_desc)
            
            if precip_rate and precip_rate > 0:
                parts.append(f"it is currently raining, with a total of {round(precip_accum_local_day * 0.03937, 2)} inches today")
            elif precip_accum_last_1hr and precip_accum_last_1hr > 0:
                parts.append(f"It is not currently raining, however it has rained {round(precip_accum_last_1hr * 0.03937, 2)} inches in the past hour")

            if lightning_strike_count_last_1hr and lightning_strike_count_last_1hr > 0:
                parts.append(f"Be advised there have been {lightning_strike_count_last_1hr} lightning strikes in the past hour {lightning_strike_last_distance_miles} miles away")
            
            if uv is not None and uv >= 6:
                parts.append(f"UV index is currently high at {round(uv)}")

            if pressure_trend:
                parts.append(f"barometric pressure is {pressure_trend} at {round(barometric_pressure, 2)} millibars")

           
            
            weather_summary = ". ".join(parts) + "." if parts else "Weather data unavailable."
            
            log_event(logger, logging.INFO, f"Weather: {weather_summary}",
                     event="weather_fetch", temp_f=temp_f, humidity=humidity, wind_mph=wind_mph)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=weather_summary,
                data={
                    "temp_f": temp_f,
                    "feels_like_f": feels_like_f,
                    "humidity": humidity,
                    "wind_mph": wind_mph,
                    "wind_gust_mph": wind_gust_mph,
                    "wind_direction": wind_cardinal,
                    "uv": uv,
                    "precip_rate": precip_rate,
                    "precip_accum_local_day": precip_accum_local_day,
                    "lightning_strike_count_last_1hr": lightning_strike_count_last_1hr,
                    "lightning_strike_last_distance": lightning_strike_last_distance,
                    "barometric_pressure": barometric_pressure,
                    "pressure_trend": pressure_trend,
                }
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Weather API HTTP error: {e}")
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Couldn't reach the weather station"
            )
        except Exception as e:
            logger.error(f"Weather fetch error: {e}", exc_info=True)
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Error getting weather data"
            )


class StatusTool(BaseTool):
    """Get status of scheduled tasks."""
    
    name = "STATUS"
    description = "Check status of timers and callbacks"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        pending = self.assistant.tool_manager.get_pending_tasks()
        
        if not pending:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message="You have no pending timers or callbacks"
            )
            
        messages = []
        for task in pending:
            remaining = (task.execute_at - datetime.now()).total_seconds()
            if remaining > 0:
                if task.task_type == "timer":
                    messages.append(f"Timer in {int(remaining)} seconds")
                else:
                    messages.append(f"Callback in {int(remaining)} seconds")
                    
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=". ".join(messages) if messages else "No pending tasks",
            data={"pending_count": len(pending)}
        )


class CancelTool(BaseTool):
    """Cancel scheduled tasks."""
    
    name = "CANCEL"
    description = "Cancel timers or callbacks"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        task_type = params.get('task_type', 'all')
        
        cancelled = await self.assistant.tool_manager.cancel_tasks(task_type)
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"Cancelled {cancelled} {'task' if cancelled == 1 else 'tasks'}",
            data={"cancelled_count": cancelled}
        )


class ToolManager:
    """Manages all tools and scheduled tasks."""
    
    def __init__(self, assistant: 'SIPAIAssistant'):
        self.assistant = assistant
        self.config = assistant.config
        self.tools: Dict[str, BaseTool] = {}
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self._task_runner: Optional[asyncio.Task] = None
        
        # Register default tools
        self._register_default_tools()
        
    def _register_default_tools(self):
        """Register built-in tools."""
        if self.config.enable_timer_tool:
            self.register_tool(TimerTool(self.assistant))
            
        if self.config.enable_callback_tool:
            self.register_tool(CallbackTool(self.assistant))
        
        # Weather tool (requires Tempest API config)
        if self.config.enable_weather_tool:
            if self.config.tempest_station_id and self.config.tempest_api_token:
                self.register_tool(WeatherTool(self.assistant))
                logger.info(f"Weather tool registered with station_id={self.config.tempest_station_id}")
            else:
                logger.info("Weather tool enabled but Tempest API not configured (missing TEMPEST_STATION_ID or TEMPEST_API_TOKEN)")
            
        # Always available
        self.register_tool(HangupTool(self.assistant))
        self.register_tool(StatusTool(self.assistant))
        self.register_tool(CancelTool(self.assistant))
        
    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
        
    def unregister_tool(self, name: str):
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
            
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
        
        # Log tool invocation (convert params to simple dict for JSON)
        try:
            params_dict = {k: str(v) for k, v in tool_call.params.items()}
        except:
            params_dict = {}
        log_event(logger, logging.INFO, f"Tool called: {tool_name}",
                 event="tool_call", tool=tool_name, params=params_dict)
        
        if tool_name not in self.tools:
            return ToolResult(status=ToolStatus.FAILED, message=f"Unknown tool: {tool_name}")
            
        tool = self.tools[tool_name]
        if not tool.enabled:
            return ToolResult(status=ToolStatus.FAILED, message=f"Tool {tool_name} disabled")

        # Validate base params (delay, message)
        error = tool.validate_params(tool_call.params)
        if error:
            return ToolResult(status=ToolStatus.FAILED, message=error)
            
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
                        return ToolResult(
                            status=ToolStatus.FAILED,
                            message="No callback number available - please specify a number"
                        )
                
                logger.debug(f"Processing CALLBACK: delay={delay}, dest={destination}")
                
                # Schedule the callback
                await self.assistant.schedule_callback(delay, message, destination)
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    message=f"I'll call you back in {self._format_delay(delay)}"
                )
            # -------------------------------

            # For other tools, run normally
            result = await tool.execute(tool_call.params)
            return result
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            import traceback
            traceback.print_exc()
            return ToolResult(status=ToolStatus.FAILED, message=str(e))
            
    async def schedule_task(
        self,
        task_type: str,
        delay_seconds: int,
        message: str,
        target_uri: Optional[str] = None
    ) -> str:
        """Schedule a task for later execution."""
        task_id = str(uuid.uuid4())[:8]
        
        task = ScheduledTask(
            id=task_id,
            task_type=task_type,
            execute_at=datetime.now() + timedelta(seconds=delay_seconds),
            message=message,
            target_uri=target_uri
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
            
    def _format_delay(self, seconds: int) -> str:
        """Format delay for natural speech."""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining = seconds % 60
            if remaining == 0:
                return f"{minutes} minute{'s' if minutes != 1 else ''}"
            return f"{minutes} minute{'s' if minutes != 1 else ''} and {remaining} seconds"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
            if minutes:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            return " and ".join(parts)


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