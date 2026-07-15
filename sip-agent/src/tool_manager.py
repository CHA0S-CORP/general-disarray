"""
Tool Manager
============
Manages callable tools for the AI assistant.
All tools are loaded as plugins from the plugins/ directory.

To add a new tool, simply drop a Python file in the plugins/ directory.
See plugins/README.md for documentation.
"""

import json
import os
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
from call_session import get_current_session, set_current_session
from telemetry import create_span, Metrics
from logging_utils import log_event, format_duration, HANGUP_DELAY_SECONDS

# Import plugin system base classes
from tool_plugins import (
    BaseTool as PluginBaseTool,
    ToolResult as PluginToolResult,
    ToolStatus as PluginToolStatus,
)

# Shared request-security helpers (dial-target policy + webhook SSRF pinning)
# live in api.py and are reused here so the voice CALLBACK path and the
# scheduler enforce exactly the same rules as the REST endpoints. api.py imports
# neither tool_manager nor main at module load, so this is not an import cycle.
from api import check_extension_allowed, deliver_webhook, tool_result_success

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
    # The CallSession the task was scheduled from (timers only; captured via
    # the current-session contextvar at schedule time). A timer belongs to
    # the call that set it: with several concurrent calls the scheduler's
    # context is unbound, so firing must target THIS session — and if it has
    # ended, the announcement expires instead of leaking into another
    # caller's call. Never persisted (to_dict omits it; timers are in-memory).
    session: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "execute_at": self.execute_at.isoformat(),
            # execute_at is naive LOCAL_TIMEZONE wall-clock; entries without
            # this marker predate it (container clock) and get migrated on load.
            "clock": "local",
            "message": self.message,
            "target_uri": self.target_uri,
            "metadata": self.metadata,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledTask':
        data = dict(data)
        data.pop("clock", None)
        data["execute_at"] = datetime.fromisoformat(data["execute_at"])
        return cls(**data)


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
        # Track in-flight dispatched task executions so a slow/retrying
        # task does not block other due tasks in the scheduler loop.
        self._running_tasks: set = set()
        # Scheduler-driven outbound-call tasks (callbacks, scheduled calls)
        # run one at a time even though they are dispatched concurrently.
        # This is POLICY, not correctness: the assistant can host several
        # concurrent call sessions (MAX_CONCURRENT_CALLS), but batch-firing
        # every due scheduled call at once would stack STT/TTS/LLM load —
        # so machine-initiated calls stay serialized. Timers are not
        # serialized by this lock.
        self._outbound_call_lock = asyncio.Lock()
        
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
        from plugins.simon_says_tool import SimonSaysTool
        from plugins.knowledge_tool import KnowledgeTool
        from plugins.random_tools import DiceTool, CoinTool
        from plugins.trivia_tool import TriviaTool
        from plugins.story_tool import StoryTool
        from plugins.web_search_tool import WebSearchTool
        from plugins.nws_weather_tool import NWSForecastTool
        from plugins.space_weather_tool import KpIndexTool
        from plugins.quake_tool import EarthquakeTool
        from plugins.memory_tools import RememberTool, ForgetTool
        from plugins.workflow_tool import TriggerWorkflowTool
        from plugins.gpu_status_tool import GpuStatusTool
        from plugins.alerts_tool import AlertsTool
        from plugins.container_tool import ContainerControlTool
        from plugins.transfer_tool import TransferTool
        from plugins.drink_tool import DrinkRecipeTool

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
            SimonSaysTool,
            KnowledgeTool,
            # Fun
            DiceTool,
            CoinTool,
            TriviaTool,
            StoryTool,
            DrinkRecipeTool,
            # Information
            WebSearchTool,
            NWSForecastTool,
            KpIndexTool,
            EarthquakeTool,
            # Memory + automation
            RememberTool,
            ForgetTool,
            TriggerWorkflowTool,
            # Ops (self-gated: need the observability stack / docker socket)
            GpuStatusTool,
            AlertsTool,
            ContainerControlTool,
            # Telephony
            TransferTool,
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

        # Auto-discover additional (non-builtin) plugins so the documented
        # "drop a file in plugins/" path actually works. Builtins stay
        # explicitly registered above and are never overridden by discovery.
        if self.config.enable_plugin_autodiscovery:
            self._discover_extra_plugins()

        logger.info(f"Loaded {len(self.tools)} tools")

    def _discover_extra_plugins(self):
        """Load non-builtin tools found by PluginLoader.

        Scans the source plugins/ directories plus data/plugins (the mounted
        data volume), so deployments can add tools without rebuilding the
        image. Tools whose names are already registered are skipped.
        """
        try:
            from tool_plugins import PluginLoader
            loader = PluginLoader()
            loader.plugin_dirs.append(self.config.data_dir / "plugins")
            discovered = loader.discover_plugins()
        except Exception as e:
            logger.error(f"Plugin auto-discovery failed: {e}")
            return

        for name, tool_class in discovered.items():
            key = name.upper()
            if key in self.tools:
                continue
            try:
                wrapper = self._create_plugin_wrapper(tool_class)
                if not self._should_enable_tool(key, wrapper):
                    logger.info(f"Skipping disabled tool: {key}")
                    continue
                self.tools[key] = wrapper
                logger.info(f"Loaded discovered plugin tool: {key}")
            except Exception as e:
                logger.error(f"Failed to load discovered tool {name}: {e}")
            
    def _should_enable_tool(self, name: str, wrapper) -> bool:
        """Check if a tool should be enabled based on configuration."""
        # Check tool-specific config flags
        if name == "SET_TIMER" and not self.config.enable_timer_tool:
            return False
        if name == "CALLBACK" and not self.config.enable_callback_tool:
            return False
        if name == "WEATHER" and not self.config.enable_weather_tool:
            return False
        if name == "KNOWLEDGE":
            # Needs the knowledge base (enabled + deps + documents present).
            kb = getattr(self.assistant, "knowledge_base", None)
            if kb is None or not kb.available:
                return False
        if name == "TRANSFER" and not self.config.enable_transfer_tool:
            return False
        if name in ("REMEMBER", "FORGET") and not self.config.caller_memory_enabled:
            return False
        if name == "DRINK_RECIPE" and not self.config.enable_drink_tool:
            return False
        # WEB_SEARCH, FORECAST and CONTAINER_CTL self-disable in __init__ when
        # their required config (SearxNG URL / coordinates / allowlist+socket)
        # is missing; the generic `enabled` check below catches them.


        # Check if tool disabled itself (e.g., missing API keys)
        if hasattr(wrapper, '_plugin_instance'):
            if not getattr(wrapper._plugin_instance, 'enabled', True):
                return False
                
        return True
            
    def _create_plugin_wrapper(self, plugin_class):
        """Create a wrapper that adapts a plugin tool to the local interface."""
        return self._wrap_tool_instance(plugin_class(self.assistant))

    def _wrap_tool_instance(self, instance):
        """Wrap an already-constructed BaseTool instance (plugins, MCP tools)."""

        class PluginToolWrapper:
            """Wrapper for plugin-based tools."""

            def __init__(wrapper_self):
                wrapper_self._plugin_instance = instance
                wrapper_self.name = instance.name
                wrapper_self.description = instance.description
                wrapper_self.enabled = getattr(instance, 'enabled', True)
                wrapper_self.parameters = getattr(instance, 'parameters', {})
                # Informational tools: result message is spoken in marker mode.
                wrapper_self.speak_result = getattr(instance, 'speak_result', False)
                # Full JSON schema (e.g. an MCP inputSchema): used verbatim by
                # _build_native_tools instead of synthesizing from `parameters`.
                wrapper_self.json_schema = getattr(instance, 'json_schema', None)
                
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
            
    # Task types worth surviving a restart. Timers reference the live call
    # (they speak into it), so they are meaningless after the process dies.
    PERSISTED_TASK_TYPES = ("callback", "scheduled_call")
    # A missed one-shot task fires immediately if it is at most this late;
    # older ones are dropped (calling someone hours late is worse than not).
    MISSED_TASK_GRACE_S = 300

    @property
    def _tasks_file(self):
        return self.config.data_dir / "scheduled_tasks.json"

    def _persist_tasks(self):
        """Atomically write persistable pending tasks to data/scheduled_tasks.json."""
        try:
            tasks = [
                task.to_dict() for task in self.scheduled_tasks.values()
                if task.task_type in self.PERSISTED_TASK_TYPES and not task.completed
            ]
            tmp_path = self._tasks_file.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(tasks, indent=2))
            os.replace(tmp_path, self._tasks_file)
        except Exception as e:
            logger.error(f"Failed to persist scheduled tasks: {e}")

    def _local_now(self) -> datetime:
        """Naive LOCAL_TIMEZONE wall-clock (see Config.local_now).

        execute_at values are stored naive and mean local wall-clock time
        (users schedule "07:00" in their timezone); comparing them against a
        naive UTC now() — which is what a TZ-less container gives — makes
        recurring schedules fire hours off.
        """
        return self.config.local_now()

    def _migrate_legacy_execute_at(self, dt: datetime) -> datetime:
        """Convert a pre-'clock' persisted execute_at (written with the naive
        container clock by older versions) to the LOCAL_TIMEZONE wall clock
        the scheduler now runs on. Assumes the container timezone did not
        change across the upgrade."""
        try:
            from zoneinfo import ZoneInfo
            system_tz = datetime.now().astimezone().tzinfo
            return dt.replace(tzinfo=system_tz).astimezone(
                ZoneInfo(self.config.local_timezone)).replace(tzinfo=None)
        except Exception:
            return dt

    def _advance_recurring(self, task: ScheduledTask) -> bool:
        """Move a past-due recurring task to its next future occurrence.

        Returns False for unsupported patterns (task should be dropped).
        """
        pattern = (task.metadata or {}).get("recurring")
        if pattern not in ("daily", "weekdays", "weekends"):
            return False
        while task.execute_at <= self._local_now():
            nxt = task.execute_at + timedelta(days=1)
            if pattern == "weekdays":
                while nxt.weekday() >= 5:
                    nxt += timedelta(days=1)
            elif pattern == "weekends":
                while nxt.weekday() < 5:
                    nxt += timedelta(days=1)
            task.execute_at = nxt
        return True

    def _load_persisted_tasks(self):
        """Reload persisted tasks at startup, applying the missed-task policy:
        recurring tasks advance to their next occurrence, one-shots missed by
        less than MISSED_TASK_GRACE_S fire shortly, older ones are dropped.
        """
        if not self._tasks_file.exists():
            return
        try:
            entries = json.loads(self._tasks_file.read_text())
        except Exception as e:
            logger.error(f"Failed to read persisted scheduled tasks: {e}")
            return

        now = self._local_now()
        restored = dropped = migrated_count = 0
        for entry in entries:
            try:
                task = ScheduledTask.from_dict(entry)
            except Exception as e:
                logger.warning(f"Skipping malformed persisted task: {e}")
                continue
            if entry.get("clock") != "local":
                migrated = self._migrate_legacy_execute_at(task.execute_at)
                if migrated != task.execute_at:
                    log_event(logger, logging.INFO,
                             f"Migrated legacy task {task.id} to local clock: "
                             f"{task.execute_at.isoformat()} -> {migrated.isoformat()}",
                             event="task_clock_migrated", task_id=task.id)
                    task.execute_at = migrated
                migrated_count += 1
            if task.execute_at <= now:
                if (task.metadata or {}).get("recurring"):
                    if not self._advance_recurring(task):
                        dropped += 1
                        continue
                elif (now - task.execute_at).total_seconds() <= self.MISSED_TASK_GRACE_S:
                    # Slightly late: fire soon rather than exactly on time.
                    task.execute_at = now + timedelta(seconds=5)
                else:
                    log_event(logger, logging.WARNING,
                             f"Dropping task {task.id} missed by more than "
                             f"{self.MISSED_TASK_GRACE_S}s while the agent was down",
                             event="task_missed_dropped", task_id=task.id,
                             task_type=task.task_type)
                    dropped += 1
                    continue
            self.scheduled_tasks[task.id] = task
            restored += 1

        if restored or dropped:
            log_event(logger, logging.INFO,
                     f"Restored {restored} scheduled task(s), dropped {dropped}",
                     event="tasks_restored", restored=restored, dropped=dropped)
        if dropped or migrated_count:
            # Re-persist so legacy entries carry the clock marker from now on
            # (migration must not re-run against a changed container TZ).
            self._persist_tasks()

    def register_tool_instance(self, instance) -> bool:
        """Register an already-constructed BaseTool instance (e.g. an MCP tool
        wrapper) through the same wrapper path as plugins.

        Never overrides an already-registered tool (same rule as plugin
        autodiscovery). Returns True when the tool was registered.
        """
        key = str(instance.name).upper()
        if key in self.tools:
            logger.warning(
                f"Tool name collision: {key} is already registered — "
                "skipping (never overriding built-ins)")
            return False
        try:
            self.tools[key] = self._wrap_tool_instance(instance)
        except Exception as e:
            logger.error(f"Failed to register tool instance {key}: {e}")
            return False
        logger.info(f"Loaded tool: {key}")
        return True

    async def _start_mcp_tools(self):
        """Connect the assistant's MCPManager (if any) and register its tools.

        Runs inside start() so MCP tools are final before the first call:
        they then appear in /tools, the system-prompt tools list, and native
        tool schemas exactly like plugins. Fail-open: any error just means no
        MCP tools.
        """
        manager = getattr(self.assistant, "mcp_manager", None)
        if manager is None:
            return
        try:
            await manager.start()
        except Exception as e:
            logger.error(f"MCP manager failed to start: {e}")
            return
        registered = 0
        for wrapper in getattr(manager, "tool_wrappers", []):
            if self.register_tool_instance(wrapper):
                registered += 1
        if registered > 15:
            logger.warning(
                f"{registered} MCP tools registered — this bloats the system "
                "prompt in text-marker mode; consider trimming the expose "
                "allowlists")
        if registered:
            logger.info(f"Registered {registered} MCP tool(s)")

    async def start(self):
        """Start the task runner."""
        await self._start_mcp_tools()
        self._load_persisted_tasks()
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
        result = await self._execute_tool_inner(tool_call)
        # Mirror the outcome onto the admin event bus (name + success only —
        # params can carry private data). Must never raise into the call path.
        try:
            bus = getattr(self.assistant, "events", None)
            if bus is not None:
                session = getattr(self.assistant, "session", None)
                status = getattr(result.status, "value", result.status)
                bus.publish(
                    "tool_call",
                    session.transcript_id if session else "-",
                    {"tool": tool_call.name.upper(),
                     "success": str(status).lower() == "success"})
        except Exception as e:
            logger.debug(f"Admin tool_call event publish failed: {e}")
        return result

    async def _execute_tool_inner(self, tool_call) -> ToolResult:
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
                    else:
                        # An explicit destination was supplied by the (untrusted)
                        # caller via the LLM. Enforce the same dial-target policy as
                        # the REST path so the voice path can't be abused to dial
                        # arbitrary numbers / SIP domains (toll fraud). On a policy
                        # violation, fall back to the verified current caller rather
                        # than the attacker-chosen target.
                        policy_error = check_extension_allowed(destination, self.config)
                        if policy_error:
                            caller_uri = getattr(self.assistant.current_call, 'remote_uri', None) \
                                if self.assistant.current_call else None
                            log_event(logger, logging.WARNING,
                                     f"CALLBACK destination rejected by policy: {policy_error}",
                                     event="callback_destination_blocked",
                                     destination=destination, reason=policy_error,
                                     fallback=caller_uri)
                            Metrics.record_tool_error(tool_name, "destination_blocked")
                            span.set_attribute("tool.error", "destination_blocked")
                            if not caller_uri:
                                return ToolResult(
                                    status=ToolStatus.FAILED,
                                    message="I can only call you back at your own number."
                                )
                            destination = caller_uri

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

        # Timers announce into the call that set them: capture the scheduling
        # task's session (bound by the turn/audio-loop task body) so the
        # scheduler — whose own context is unbound — can route the
        # announcement to the right call even with several calls live.
        session = get_current_session() if task_type == "timer" else None

        task = ScheduledTask(
            id=task_id,
            task_type=task_type,
            execute_at=self._local_now() + timedelta(seconds=delay_seconds),
            message=message,
            target_uri=target_uri,
            metadata=metadata or {},
            session=session,
        )
        
        self.scheduled_tasks[task_id] = task
        if task_type in self.PERSISTED_TASK_TYPES:
            self._persist_tasks()
        log_event(logger, logging.INFO, f"Task scheduled: {task_type} in {delay_seconds}s",
                 event="task_scheduled", task_id=task_id, task_type=task_type,
                 delay=delay_seconds, target=str(target_uri) if target_uri else None)

        return task_id
        
    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all pending (not completed) tasks."""
        now = self._local_now()
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

        if cancelled:
            self._persist_tasks()
        return cancelled

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a single task by id. Returns True if it existed."""
        task = self.scheduled_tasks.pop(task_id, None)
        if task is None:
            return False
        if task.task_type in self.PERSISTED_TASK_TYPES:
            self._persist_tasks()
        return True
        
    async def _run_scheduler(self):
        """Background task that executes scheduled tasks."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                now = self._local_now()
                
                for task_id, task in list(self.scheduled_tasks.items()):
                    if task.completed:
                        continue
                        
                    if now >= task.execute_at:
                        # Mark completed immediately so the task is not
                        # re-dispatched on the next tick, then run it
                        # concurrently so a slow/retrying task (e.g. a
                        # callback retrying for ~120s) does not block other
                        # due tasks such as timers.
                        task.completed = True
                        if task.task_type in self.PERSISTED_TASK_TYPES:
                            # Drop it from the on-disk set now so a crash
                            # mid-execution doesn't re-fire the call on restart.
                            self._persist_tasks()
                        runner = asyncio.create_task(self._execute_scheduled_task(task))
                        self._running_tasks.add(runner)
                        runner.add_done_callback(self._running_tasks.discard)
                        
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
                async with self._outbound_call_lock:
                    await self._execute_callback(task)
            elif task.task_type == "scheduled_call":
                async with self._outbound_call_lock:
                    await self._execute_scheduled_call(task)
            else:
                logger.warning(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            
    async def _execute_timer(self, task: ScheduledTask):
        """Execute a timer - speak the message on the call that set it."""
        log_event(logger, logging.INFO, f"Timer fired: {task.message}",
                 event="timer_fired", task_id=task.id, message=task.message)

        if task.session is not None:
            # The timer was set from inside a call, so it belongs to THAT
            # call. Bind the session in this (scheduler-spawned, otherwise
            # unbound) task so everything downstream — assistant.session /
            # current_call, playback — resolves the originating call even
            # with several calls live.
            set_current_session(task.session)
            registered = any(
                s is task.session
                for s in getattr(self.assistant, "sessions", {}).values())
            call = task.session.call_info if registered else None
            if call is not None and getattr(call, "is_active", False):
                if hasattr(self.assistant, '_stream_response'):
                    await self.assistant._stream_response(call, task.message)
                else:
                    await self.assistant._speak(task.message)
            else:
                # The originating call ended: the announcement expires.
                # Never fall back to "the" current call — with another call
                # live that would speak this caller's reminder into someone
                # else's conversation.
                logger.warning(
                    f"Timer {task.id} expired but its originating call has "
                    "ended; dropping announcement")
            return

        # No originating session recorded (e.g. scheduled via the REST API
        # outside any call): fall back to the sole active call, if any.
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
                    
                    if tool_result_success(result):
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

        # Opt-in LLM rewrite into spoken form. Done here (not at schedule
        # creation) because tool-driven schedules produce their text at fire
        # time; reformat_for_speech falls back to the original on any failure.
        if metadata.get("reformat_for_speech"):
            engine = getattr(self.assistant, "llm_engine", None)
            if engine is not None:
                full_message = await engine.reformat_for_speech(
                    full_message, self.config.message_reformat_timeout_s)

        # Make the call
        call_succeeded = False
        for attempt in range(self.config.callback_retry_attempts):
            try:
                await self.assistant.make_outbound_call(extension, full_message)

                log_event(logger, logging.INFO, f"Scheduled call completed: {task.id}",
                         event="scheduled_call_complete", task_id=task.id, extension=extension)

                call_succeeded = True
                break

            except Exception as e:
                logger.warning(f"Scheduled call attempt {attempt + 1} failed: {e}")
                if attempt < self.config.callback_retry_attempts - 1:
                    await asyncio.sleep(self.config.callback_retry_delay_s)

        if not call_succeeded:
            logger.error(f"Scheduled call {task.id} failed after {self.config.callback_retry_attempts} attempts")

            if metadata.get("callback_url"):
                await self._send_scheduled_call_webhook(task, metadata, "failed")
            return

        # Post-success work runs in a SEPARATE try so that a failure here
        # (e.g. a bad timezone while rescheduling, or a webhook error) does
        # NOT re-trigger the already-placed outbound call, which would cause
        # duplicate calls and lose the recurrence.
        try:
            # Handle recurring
            if metadata.get("recurring"):
                await self._reschedule_recurring_call(task, metadata)

            # Send callback webhook if specified
            if metadata.get("callback_url"):
                await self._send_scheduled_call_webhook(task, metadata, "completed")
        except Exception as e:
            logger.error(f"Scheduled call {task.id} post-call handling failed: {e}")
    
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
        """Send webhook for scheduled call completion.

        Delegates to deliver_webhook, which re-validates and IP-pins the target
        at send time (SSRF / DNS-rebinding defense), signs the payload, and
        retries transient failures — the same protection as the REST callback
        path. Pinning at fire time matters most for recurring schedules, whose
        URL could be repointed to an internal address after it was accepted.
        """
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

        if await deliver_webhook(url, payload, self.config,
                                 api_name="scheduled_call_webhook"):
            logger.info(f"Scheduled call webhook sent: {url}")

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
        cutoff = self._local_now() - timedelta(hours=1)
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