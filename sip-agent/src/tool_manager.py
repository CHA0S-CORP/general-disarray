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
from dataclasses import fields as dataclass_fields
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from main import SIPAIAssistant

from config import Config
from call_session import get_current_session, set_current_session
from telemetry import create_span, Metrics
from logging_utils import log_event, HANGUP_DELAY_SECONDS
from caller_memory import caller_id_from_uri

# Import plugin system base classes
from tool_plugins import (
    BaseTool as PluginBaseTool,
    ToolResult as PluginToolResult,
    ToolStatus as PluginToolStatus,
)

# Shared request-security helpers (webhook SSRF pinning) live in api.py and are
# reused here so the scheduler enforces exactly the same rules as the REST
# endpoints. The voice dial-target policy lives in plugins.helpers
# (check_voice_dial_allowed, used by CALLBACK and TRANSFER). api.py imports
# neither tool_manager nor main at module load, so this is not an import cycle.
from api import deliver_webhook, tool_result_success

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
    # What the CALLER hears (see tool_plugins.ToolResult.spoken_message):
    # carried through from the plugin result so llm_engine's to_speech()
    # path speaks e.g. WEB_SEARCH's summary instead of raw scraped results.
    spoken_message: str = ""

    def to_speech(self) -> str:
        """Speech-friendly text: spoken_message, falling back to message."""
        return self.spoken_message or self.message


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
    # Who scheduled it, for the voice CANCEL/STATUS scope: the originating
    # call (CallSession.transcript_id) and caller (SIP URI user part). None
    # for tasks created via the REST API (and for records predating these
    # fields) — those are never visible to, or cancellable by, a caller.
    owner_call_id: Optional[str] = None
    owner_caller: Optional[str] = None

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
            "owner_call_id": self.owner_call_id,
            "owner_caller": self.owner_caller,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledTask':
        # Keep only known persisted fields: old records lack the owner keys
        # (dataclass defaults apply) and unknown/future keys are ignored.
        known = {f.name for f in dataclass_fields(cls)} - {"session"}
        data = {k: v for k, v in dict(data).items() if k in known}
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
        from plugins.persona_tool import PersonaTool
        from plugins.workflow_tool import TriggerWorkflowTool
        from plugins.gpu_status_tool import GpuStatusTool
        from plugins.alerts_tool import AlertsTool
        from plugins.container_tool import ContainerControlTool
        from plugins.transfer_tool import TransferTool
        from plugins.drink_tool import DrinkRecipeTool
        from plugins.map_tool import MapTool
        from plugins.verify_tool import VerifyTool

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
            MapTool,
            # Memory + automation
            RememberTool,
            ForgetTool,
            PersonaTool,
            TriggerWorkflowTool,
            # Ops (self-gated: need the observability stack / docker socket)
            GpuStatusTool,
            AlertsTool,
            ContainerControlTool,
            # Telephony
            TransferTool,
            # Identity verification
            VerifyTool,
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
        if name == "PERSONA" and not self.config.enable_persona_tool:
            return False
        if name == "DRINK_RECIPE" and not self.config.enable_drink_tool:
            return False
        if name == "MAP" and not self.config.enable_map_tool:
            return False
        if name == "VERIFY" and not self.config.enable_verify_tool:
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
                    data=result.data,
                    spoken_message=getattr(result, "spoken_message", "") or "",
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
        Returns the number of tools loaded. The new set is built aside and
        swapped in only on success, so a failed reload keeps the old tools.
        Externally registered instances (MCP tools) are carried over.
        """
        old_tools = self.tools
        self.tools = {}
        try:
            self._load_tools()
        except Exception as e:
            logger.error(f"Plugin reload failed, keeping existing tools: {e}",
                         exc_info=True)
            self.tools = old_tools
            return len(self.tools)
        if not self.tools:
            logger.error("Plugin reload produced no tools, keeping existing tools")
            self.tools = old_tools
            return len(self.tools)
        for key, wrapper in old_tools.items():
            if key not in self.tools and getattr(wrapper, "_external", False):
                self.tools[key] = wrapper
        logger.info(f"Reloaded tools: {len(old_tools)} -> {len(self.tools)}")
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
        next_time = self._next_occurrence(task, self._local_now())
        if next_time is None:
            logger.warning(f"Unsupported recurring pattern "
                           f"{(task.metadata or {}).get('recurring')!r} for task {task.id}")
            return False
        task.execute_at = next_time
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
            wrapper = self._wrap_tool_instance(instance)
            # Not reproducible by _load_tools: reload_plugins carries it over.
            wrapper._external = True
            self.tools[key] = wrapper
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

    def verification_block(self, tool_name: str, session) -> Optional[ToolResult]:
        """The VERIFY_REQUIRED_TOOLS gate, shared by LLM-driven and REST execution.

        Returns a FAILED ToolResult when ``tool_name`` is gated and ``session``
        is not a verified call session (fail closed, including no session at
        all); None when the tool may run. VERIFY itself is never gated.
        """
        tool_name = (tool_name or "").upper()
        gated_tools = getattr(self.config, "verify_required_tools_set", None) or set()
        if tool_name not in gated_tools or tool_name == "VERIFY":
            return None
        if getattr(session, "verified", False):
            return None
        return ToolResult(
            status=ToolStatus.FAILED,
            message=("You'll need to verify your identity first — "
                     "say 'verify me' to begin."))

    async def _execute_tool_inner(self, tool_call) -> ToolResult:
        tool_name = tool_call.name.upper()
        start_time = time.time()
        
        # Log tool invocation (convert params to simple dict for JSON)
        try:
            params_dict = {k: str(v) for k, v in tool_call.params.items()}
        except Exception:
            params_dict = {}
        log_event(logger, logging.INFO, f"Tool called: {tool_name}",
                 event="tool_call", tool=tool_name, params=params_dict)
        
        if tool_name not in self.tools:
            # Hallucinated names must not mint new Prometheus series: the
            # tool_name label is bounded to registered tools + "unknown".
            Metrics.record_tool_call("unknown")
            Metrics.record_tool_error("unknown", "unknown_tool")
            return ToolResult(status=ToolStatus.FAILED, message=f"Unknown tool: {tool_name}")

        # Record tool call metric (validated name only)
        Metrics.record_tool_call(tool_name)

        tool = self.tools[tool_name]
        if not tool.enabled:
            Metrics.record_tool_error(tool_name, "disabled")
            return ToolResult(status=ToolStatus.FAILED, message=f"Tool {tool_name} disabled")

        # Validate base params (delay, message)
        error = tool.validate_params(tool_call.params)
        if error:
            Metrics.record_tool_error(tool_name, "validation_error")
            return ToolResult(status=ToolStatus.FAILED, message=error)

        # --- IDENTITY VERIFICATION GATE ---
        # Tools named in config.verify_required_tools require a caller who has
        # passed the VERIFY flow this call. Fail CLOSED: if verification can't be
        # confirmed, refuse. VERIFY itself is never gated (that would deadlock).
        blocked = self.verification_block(
            tool_name, getattr(self.assistant, "session", None))
        if blocked is not None:
            Metrics.record_tool_error(tool_name, "verification_required")
            log_event(logger, logging.INFO,
                      f"Tool {tool_name} blocked: caller not verified",
                      event="verify_gate", tool=tool_name, outcome="blocked")
            return blocked

        with create_span(f"tool.{tool_name.lower()}", {
            "tool.name": tool_name,
            "tool.params": str(params_dict)
        }) as span:
            try:
                # CALLBACK needs no interception: CallbackTool itself defaults
                # to the current caller's number and enforces the voice-dial
                # policy, delay bounds and per-call cap — so the REST paths
                # (/tools/CALLBACK/execute, /webhook/call) get them too.
                result = await tool.execute(tool_call.params)
                
                latency_ms = (time.time() - start_time) * 1000
                Metrics.record_tool_latency(latency_ms, tool_name)
                span.set_attribute("tool.latency_ms", latency_ms)
                span.set_attribute("tool.status", result.status.value)
                
                if result.status == ToolStatus.FAILED:
                    Metrics.record_tool_error(tool_name, "execution_failed")
                
                return result
                
            except Exception as e:
                # Details go to the log/trace only: str(e) can carry internal
                # hosts, paths or stack details that must never be spoken.
                logger.error(f"Tool execution error in {tool_name}: {e}", exc_info=True)
                latency_ms = (time.time() - start_time) * 1000
                Metrics.record_tool_latency(latency_ms, tool_name)
                Metrics.record_tool_error(tool_name, type(e).__name__)
                span.record_exception(e)
                return ToolResult(
                    status=ToolStatus.FAILED,
                    message=f"Sorry, the {tool_name.lower().replace('_', ' ')} "
                            "tool ran into a problem.")
            
    async def schedule_task(
        self,
        task_type: str,
        delay_seconds: int,
        message: str,
        target_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        owner_call_id: Optional[str] = None,
        owner_caller: Optional[str] = None,
    ) -> str:
        """Schedule a task for later execution.

        Timers and callbacks are owned by the call scheduling them (taken
        from the live session when no owner is passed) so voice CANCEL/STATUS
        only ever see the caller's own tasks. scheduled_call tasks (REST
        /schedule) are never auto-owned.
        """
        task_id = str(uuid.uuid4())[:8]

        # Timers announce into the call that set them: capture the scheduling
        # task's session (bound by the turn/audio-loop task body) so the
        # scheduler — whose own context is unbound — can route the
        # announcement to the right call even with several calls live.
        session = get_current_session() if task_type == "timer" else None

        if (owner_call_id is None and owner_caller is None
                and task_type in self.VOICE_TASK_TYPES):
            owner_call_id, owner_caller = self.current_owner()

        task = ScheduledTask(
            id=task_id,
            task_type=task_type,
            execute_at=self._local_now() + timedelta(seconds=delay_seconds),
            message=message,
            target_uri=target_uri,
            metadata=metadata or {},
            session=session,
            owner_call_id=owner_call_id,
            owner_caller=owner_caller,
        )
        
        self.scheduled_tasks[task_id] = task
        if task_type in self.PERSISTED_TASK_TYPES:
            self._persist_tasks()
        log_event(logger, logging.INFO, f"Task scheduled: {task_type} in {delay_seconds}s",
                 event="task_scheduled", task_id=task_id, task_type=task_type,
                 delay=delay_seconds, target=str(target_uri) if target_uri else None)

        return task_id
        
    # Task types a caller can create by voice (and therefore see/cancel).
    VOICE_TASK_TYPES = ("timer", "callback")

    def current_owner(self) -> Tuple[Optional[str], Optional[str]]:
        """(call id, caller id) of the call the current task acts for.

        Resolved through assistant.session (the bound session, else the sole
        live one); falls back to assistant.current_call for minimal
        assistants. (None, None) when no call is live.
        """
        session = getattr(self.assistant, "session", None)
        if session is not None:
            call = getattr(session, "call_info", None)
        else:
            call = getattr(self.assistant, "current_call", None)
        call_id = getattr(session, "transcript_id", None) or getattr(call, "call_id", None)
        remote = getattr(call, "remote_uri", None)
        caller = getattr(session, "caller_id", "") or (
            caller_id_from_uri(remote) if remote else None)
        return (str(call_id) if call_id else None), (caller or None)

    def _owned_by(self, task: ScheduledTask, owner: Tuple[Optional[str], Optional[str]]) -> bool:
        """Whether a caller may see/cancel ``task`` by voice. Only voice-type
        tasks, and only those owned by this call or this caller — never REST
        /schedule calls or unowned (REST / legacy) tasks. With no live call
        (REST/operator use) every timer/callback is visible."""
        if task.task_type not in self.VOICE_TASK_TYPES:
            return False
        call_id, caller = owner
        if not call_id and not caller:
            # No live call: an (authenticated) REST/operator invocation, not
            # a caller — it sees every timer/callback, still never /schedule
            # calls (those are cancelled by id via DELETE /schedule).
            return True
        if call_id and task.owner_call_id == call_id:
            return True
        return bool(caller) and task.owner_caller == caller

    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all pending (not completed) tasks (admin/REST view)."""
        now = self._local_now()
        return [
            task for task in self.scheduled_tasks.values()
            if not task.completed and task.execute_at > now
        ]

    def get_owned_pending_tasks(self) -> List[ScheduledTask]:
        """Pending tasks the current caller owns (the voice STATUS view)."""
        owner = self.current_owner()
        return [t for t in self.get_pending_tasks() if self._owned_by(t, owner)]
        
    async def cancel_tasks(self, task_type: str = 'all', owned_only: bool = False) -> int:
        """Cancel tasks by type. Returns number cancelled.

        ``owned_only`` (the voice CANCEL path) restricts it to timers and
        callbacks owned by the current call/caller; REST-scheduled calls are
        never touched that way (cancel those by id via DELETE /schedule).
        """
        owner = self.current_owner() if owned_only else None
        to_remove = []

        for task_id, task in self.scheduled_tasks.items():
            if task.completed:
                continue
            if task_type != 'all' and task.task_type != task_type:
                continue
            if owner is not None and not self._owned_by(task, owner):
                continue
            to_remove.append(task_id)

        for task_id in to_remove:
            del self.scheduled_tasks[task_id]

        if to_remove:
            self._persist_tasks()
        return len(to_remove)

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
        """Execute a scheduled call - optionally run a tool, then make outbound call.

        A recurring schedule is rescheduled (same id) whatever this
        occurrence's outcome — failure, exception or cancellation — so one
        unanswered morning call never silently ends the series.
        """
        metadata = task.metadata or {}
        extension = metadata.get("extension") or task.target_uri
        
        if not extension:
            logger.error(f"Scheduled call {task.id} has no extension")
            return

        call_succeeded = False
        try:
            call_succeeded = await self._place_scheduled_call(task, metadata, extension)
        except Exception as e:
            logger.error(f"Scheduled call {task.id} failed: {e}", exc_info=True)
        finally:
            if metadata.get("recurring"):
                try:
                    await self._reschedule_recurring_call(task, metadata)
                except Exception as e:
                    logger.error(f"Scheduled call {task.id} reschedule failed: {e}",
                                 exc_info=True)

        if metadata.get("callback_url"):
            try:
                await self._send_scheduled_call_webhook(
                    task, metadata, "completed" if call_succeeded else "failed")
            except Exception as e:
                logger.error(f"Scheduled call {task.id} webhook failed: {e}")

    async def _place_scheduled_call(self, task: ScheduledTask, metadata: dict,
                                    extension: str) -> bool:
        """Build the scheduled call's message and dial it (with retries).
        Returns True when the call was placed."""
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

        # Make the call. make_outbound_call raises on dial failure / no
        # answer / busy, so every exception counts as a failed attempt.
        for attempt in range(self.config.callback_retry_attempts):
            try:
                await self.assistant.make_outbound_call(extension, full_message)

                log_event(logger, logging.INFO, f"Scheduled call completed: {task.id}",
                         event="scheduled_call_complete", task_id=task.id, extension=extension)
                return True

            except Exception as e:
                logger.warning(f"Scheduled call attempt {attempt + 1} failed: {e}")
                if attempt < self.config.callback_retry_attempts - 1:
                    await asyncio.sleep(self.config.callback_retry_delay_s)

        logger.error(f"Scheduled call {task.id} failed after {self.config.callback_retry_attempts} attempts")
        return False

    # Recurrence patterns the scheduler understands (cron is not supported).
    RECURRING_PATTERNS = ("daily", "weekdays", "weekends")
    # The API's default timezone for at_time / recurring schedules.
    DEFAULT_SCHEDULE_TZ = "America/Los_Angeles"

    def _zone(self, name: Optional[str], fallback: str) -> ZoneInfo:
        for candidate in (name, fallback, "UTC"):
            if not candidate:
                continue
            try:
                return ZoneInfo(candidate)
            except Exception:
                logger.warning(f"Unknown timezone {candidate!r}; falling back")
        return ZoneInfo("UTC")

    def _recurrence_anchor(self, task: ScheduledTask, tz: ZoneInfo, local_tz: ZoneInfo) -> dt_time:
        """The schedule's wall-clock time of day in its own timezone.

        From ``at_time`` (HH:MM, or an ISO datetime) when present, else from
        the first occurrence's execute_at. Cached in metadata["anchor_time"]
        so every later occurrence lands on the same wall-clock time — no
        drift by call duration, stable across DST changes.
        """
        metadata = task.metadata
        anchor = metadata.get("anchor_time")
        if anchor:
            try:
                return dt_time.fromisoformat(anchor)
            except ValueError:
                pass
        result: Optional[dt_time] = None
        at_time = str(metadata.get("at_time") or "").strip()
        if at_time:
            try:
                if "T" in at_time or "-" in at_time:
                    parsed = datetime.fromisoformat(at_time.replace("Z", "+00:00"))
                    if parsed.tzinfo is not None:
                        parsed = parsed.astimezone(tz)
                    result = parsed.time().replace(microsecond=0, tzinfo=None)
                else:
                    hour, minute = map(int, at_time.split(":")[:2])
                    result = dt_time(hour, minute)
            except (ValueError, TypeError):
                result = None
        if result is None:
            result = (task.execute_at.replace(tzinfo=local_tz).astimezone(tz)
                      .time().replace(microsecond=0, tzinfo=None))
        metadata["anchor_time"] = result.isoformat()
        return result

    def _next_occurrence(self, task: ScheduledTask, after: datetime) -> Optional[datetime]:
        """Next run of a recurring task strictly after ``after`` (naive
        LOCAL_TIMEZONE wall clock, like execute_at), or None for unsupported
        patterns.

        Computed from the schedule's original wall-clock time in ITS timezone
        and localised properly (zoneinfo), so a 07:00 America/New_York call
        stays at 07:00 local across DST and never drifts by call duration.
        """
        metadata = task.metadata if task.metadata is not None else {}
        task.metadata = metadata
        pattern = metadata.get("recurring")
        if pattern not in self.RECURRING_PATTERNS:
            return None
        local_tz = self._zone(self.config.local_timezone, "UTC")
        tz = self._zone(metadata.get("timezone"), self.DEFAULT_SCHEDULE_TZ)
        anchor = self._recurrence_anchor(task, tz, local_tz)

        after_aware = after.replace(tzinfo=local_tz)
        day: date = task.execute_at.replace(tzinfo=local_tz).astimezone(tz).date()
        for _ in range(3660):  # bounded: ~10 years of days
            day += timedelta(days=1)
            if pattern == "weekdays" and day.weekday() >= 5:
                continue
            if pattern == "weekends" and day.weekday() < 5:
                continue
            candidate = datetime.combine(day, anchor, tzinfo=tz)
            # Round-trip through UTC so a non-existent wall time (inside a
            # spring-forward gap) resolves to the real instant.
            candidate = candidate.astimezone(timezone.utc).astimezone(tz)
            if candidate > after_aware:
                return candidate.astimezone(local_tz).replace(tzinfo=None)
        return None

    async def _reschedule_recurring_call(self, task: ScheduledTask, metadata: dict):
        """Re-arm a recurring call for its next occurrence IN PLACE.

        Reuses task.id so DELETE /schedule/{id} keeps working for the whole
        series. A task cancelled while its call was running (removed from
        scheduled_tasks) is not resurrected.
        """
        recurring = metadata.get("recurring")
        if not recurring:
            return
        if self.scheduled_tasks.get(task.id) is not task:
            log_event(logger, logging.INFO,
                      f"Recurring call {task.id} was cancelled; not rescheduling",
                      event="scheduled_call_series_cancelled", task_id=task.id)
            return

        next_time = self._next_occurrence(task, self._local_now())
        if next_time is None:
            logger.warning(f"Unsupported recurring pattern {recurring!r} for "
                           f"scheduled call {task.id}; not rescheduling")
            return

        task.execute_at = next_time
        task.completed = False
        self._persist_tasks()

        log_event(logger, logging.INFO, f"Rescheduled recurring call: {task.id}",
                 event="scheduled_call_rescheduled",
                 task_id=task.id,
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
            "timestamp": datetime.now(timezone.utc).isoformat()
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