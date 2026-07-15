"""
MCP client support (Model Context Protocol)
===========================================
Lets the agent consume tools from external MCP servers alongside the
built-ins and filesystem plugins.

- ``MCPManager`` connects to each configured server (stdio or streamable
  HTTP), lists its tools, and builds ``MCPToolWrapper`` instances for the
  tools the operator explicitly exposed. ``ToolManager.start()`` registers
  those wrappers through the same wrapper/registration path as plugins.
- Everything is fail-open: a missing/invalid servers file, a missing ``mcp``
  package, or an unreachable server logs a warning and registers nothing.

Gated by ``MCP_ENABLED`` (default false), configured via ``MCP_SERVERS_FILE``
(default ``<data_dir>/mcp_servers.json``) and ``MCP_TOOL_TIMEOUT_S``.

Asyncio lifecycle note
----------------------
The mcp SDK's transports and ``ClientSession`` are async context managers
backed by anyio task groups / cancel scopes, which MUST be entered and exited
by the same asyncio task. Each server connection therefore runs on its own
dedicated task (``_MCPConnection._run``) that opens the transport + session,
signals readiness, parks on a stop event, and unwinds everything itself.
Other tasks only ever *use* the session object for requests, which is safe.
"""

import asyncio
import copy
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tool_plugins import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Guarded import (same pattern as langchain_engine.py): the mcp package is
# optional at runtime. MCP_ENABLED=true without it installed -> warn + no-op.
try:
    from mcp import ClientSession, McpError, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

    class McpError(Exception):  # type: ignore[no-redef]
        """Placeholder when the mcp package is absent (never raised by the SDK,
        kept so `except McpError` clauses stay valid)."""


# =============================================================================
# Pure helpers (unit-testable without the mcp package)
# =============================================================================

_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]")

# Marker mode parses [TOOL:(\w+)...], so registered names must be \w only.
def sanitize_tool_name(server: str, tool: str) -> str:
    """Build the registered tool name: SERVER_TOOL, uppercased, \\w-safe."""
    return _NAME_SANITIZE_RE.sub("_", f"{server}_{tool}").upper()


_FLAT_TYPES = ("string", "integer", "number", "boolean")


def flatten_input_schema(schema: Any) -> Dict[str, Dict[str, Any]]:
    """Map an MCP inputSchema's top-level properties to the flat
    ``BaseTool.parameters`` dict used by marker mode and /tools.

    Scalar JSON types pass through; anything nested (object/array/anyOf/…)
    becomes type "string" described as a JSON object passed as a string.
    """
    params: Dict[str, Dict[str, Any]] = {}
    if not isinstance(schema, dict):
        return params
    props = schema.get("properties")
    if not isinstance(props, dict):
        return params
    required = schema.get("required")
    required_set = set(required) if isinstance(required, list) else set()
    for pname, spec in props.items():
        if not isinstance(spec, dict):
            spec = {}
        desc = spec.get("description") or ""
        ptype = spec.get("type")
        out: Dict[str, Any] = {}
        if ptype in _FLAT_TYPES:
            out["type"] = ptype
            if desc:
                out["description"] = desc
        else:
            out["type"] = "string"
            note = "JSON object as a string"
            out["description"] = f"{desc} ({note})" if desc else note
        if pname in required_set:
            out["required"] = True
        params[pname] = out
    return params


@dataclass
class MCPServerEntry:
    """One entry from the MCP servers config file."""
    name: str
    transport: str                       # "stdio" | "http"
    url: str = ""                        # http transport
    command: str = ""                    # stdio transport
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    expose: List[str] = field(default_factory=list)   # REQUIRED allowlist
    confirm: List[str] = field(default_factory=list)  # tools needing confirm=true
    timeout_s: Optional[float] = None    # per-server tool-call timeout


def load_servers_file(path: Path) -> List[MCPServerEntry]:
    """Load and validate the MCP servers config file.

    File format — a JSON array of server objects::

        [
          {
            "name": "homeassistant",          // required, unique
            "transport": "http",              // "http" (streamable HTTP) or "stdio"
            "url": "http://ha:8123/mcp",      // required for http
            "expose": ["turn_on_light"],      // REQUIRED allowlist of tool names;
                                              // a server with no non-empty expose
                                              // list registers zero tools
            "confirm": ["turn_on_light"],     // tools requiring confirm=true
            "timeout_s": 10,                  // per-call timeout (default MCP_TOOL_TIMEOUT_S)
            "env": {}                         // stdio only: extra env vars
          },
          {
            "name": "files",
            "transport": "stdio",
            "command": "mcp-server-files",    // required for stdio
            "args": ["--root", "/data"],
            "expose": ["read_file"]
          }
        ]

    ``expose`` is a mandatory allowlist because the caller is untrusted voice
    input — servers are never auto-exposed wholesale. Invalid entries (and a
    missing/unparsable file) are skipped with a warning; nothing raises.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"MCP servers file not found: {path} — no MCP tools will load")
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"MCP servers file {path} is not valid JSON ({e}) — ignoring it")
        return []
    if not isinstance(raw, list):
        logger.warning(f"MCP servers file {path} must be a JSON array — ignoring it")
        return []

    entries: List[MCPServerEntry] = []
    seen = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.warning(f"MCP servers file: entry {i} is not an object — skipped")
            continue
        name = str(item.get("name") or "").strip()
        transport = str(item.get("transport") or "").strip().lower()
        if transport in ("streamable_http", "streamable-http"):
            transport = "http"
        if not name:
            logger.warning(f"MCP servers file: entry {i} has no name — skipped")
            continue
        if name in seen:
            logger.warning(f"MCP servers file: duplicate server name '{name}' — keeping first")
            continue
        if transport not in ("stdio", "http"):
            logger.warning(f"MCP server '{name}': unknown transport '{transport}' — skipped")
            continue
        expose = item.get("expose")
        if not isinstance(expose, list) or not expose:
            # Mandatory allowlist: never auto-expose a server's whole tool set
            # to untrusted voice input.
            logger.warning(
                f"MCP server '{name}' has no non-empty 'expose' allowlist — "
                "it will register zero tools (expose is required)")
            continue
        if transport == "stdio" and not item.get("command"):
            logger.warning(f"MCP server '{name}': stdio transport requires 'command' — skipped")
            continue
        if transport == "http" and not item.get("url"):
            logger.warning(f"MCP server '{name}': http transport requires 'url' — skipped")
            continue
        timeout_s = item.get("timeout_s")
        try:
            timeout_s = float(timeout_s) if timeout_s is not None else None
        except (TypeError, ValueError):
            timeout_s = None
        entries.append(MCPServerEntry(
            name=name,
            transport=transport,
            url=str(item.get("url") or ""),
            command=str(item.get("command") or ""),
            args=[str(a) for a in item.get("args") or []],
            env={str(k): str(v) for k, v in (item.get("env") or {}).items()},
            expose=[str(t) for t in expose],
            confirm=[str(t) for t in item.get("confirm") or []],
            timeout_s=timeout_s,
        ))
        seen.add(name)
    return entries


# =============================================================================
# Errors
# =============================================================================

class MCPToolError(Exception):
    """Spoken-friendly MCP failure."""


class MCPTimeoutError(MCPToolError):
    """The MCP tool call exceeded its timeout."""


# =============================================================================
# Connection: one persistent session per server, owned by a dedicated task
# =============================================================================

class _MCPConnection:
    """Holds one persistent ClientSession to a single MCP server.

    All transport/session context managers are entered and exited inside
    ``_run`` (a dedicated task) so anyio cancel scopes never cross tasks.
    """

    CONNECT_TIMEOUT_S = 15.0

    def __init__(self, entry: MCPServerEntry):
        self.entry = entry
        self.session = None                      # type: Optional[Any]
        self.tools: List[Any] = []               # mcp.types.Tool
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._reapers: set = set()

    async def start(self) -> bool:
        """Connect + initialize + list tools. Returns True on success."""
        await self.stop()
        self._ready = asyncio.Event()
        self._stop_event = asyncio.Event()
        self.session = None
        self.error = None
        self._task = asyncio.create_task(
            self._run(self._stop_event), name=f"mcp-conn-{self.entry.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), self.CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError:
            self.error = "connection timed out"
            await self.stop()
            return False
        if self.session is None:
            await self.stop()
            return False
        return True

    async def stop(self):
        """Signal the connection task to unwind its contexts and wait for it."""
        self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None:
            try:
                await asyncio.wait_for(task, 10)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception:
                pass
        self.session = None

    def abandon(self):
        """Detach the current session/task and unwind it in the background.

        Used on the speaking path, where awaiting a graceful ``stop()`` (up to
        10s for a stuck task) would be dead air for the phone caller. The
        session is nulled synchronously; the old task is reaped by a detached
        background task, so a subsequent ``start()`` can proceed immediately
        without the reaper ever touching the new session.
        """
        self.session = None
        task = self._task
        self._task = None
        self._stop_event.set()
        if task is None:
            return

        async def _reap():
            try:
                await asyncio.wait_for(task, 10)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception:
                pass

        reaper = asyncio.create_task(
            _reap(), name=f"mcp-abandon-{self.entry.name}")
        self._reapers.add(reaper)
        reaper.add_done_callback(self._reapers.discard)

    async def _run(self, stop_event: asyncio.Event):
        """Owns the transport + session lifecycles start to finish.

        ``stop_event`` is passed in (rather than read off ``self``) so a task
        abandoned by ``abandon()`` still unwinds on its own event even after
        ``start()`` has installed fresh events for a new task.
        """
        try:
            if self.entry.transport == "stdio":
                params = StdioServerParameters(
                    command=self.entry.command,
                    args=self.entry.args,
                    env=self.entry.env or None,
                )
                async with stdio_client(params) as (read, write):
                    await self._session_loop(read, write, stop_event)
            else:
                async with streamablehttp_client(self.entry.url) as (read, write, _):
                    await self._session_loop(read, write, stop_event)
        except Exception as e:
            self.error = str(e)
            logger.warning(f"MCP server '{self.entry.name}' connection ended: {e}")
        finally:
            self._ready.set()

    async def _session_loop(self, read, write, stop_event: asyncio.Event):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            self.tools = list(result.tools)
            self.session = session
            self._ready.set()
            try:
                await stop_event.wait()
            finally:
                # Only clear our own session: an abandoned task must never
                # clobber a newer session installed by a later start().
                if self.session is session:
                    self.session = None


# =============================================================================
# Manager
# =============================================================================

class MCPManager:
    """Connects to configured MCP servers and exposes their tools as wrappers.

    Fail-open by design: any failure logs a warning and yields zero tools for
    the affected server; other servers still load.
    """

    def __init__(self, config):
        self.config = config
        self.connections: Dict[str, _MCPConnection] = {}
        self.tool_wrappers: List["MCPToolWrapper"] = []
        self._started = False

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "mcp_enabled", False))

    async def start(self):
        """Load the servers file, connect servers, build tool wrappers."""
        if self._started:
            return
        self._started = True
        if not self.enabled:
            return
        if not MCP_AVAILABLE:
            logger.warning(
                "MCP_ENABLED=true but the 'mcp' package is not installed — "
                "no MCP tools will be registered (pip install 'mcp>=1.0')")
            return

        entries = load_servers_file(Path(self.config.mcp_servers_file))
        for entry in entries:
            conn = _MCPConnection(entry)
            try:
                ok = await conn.start()
            except Exception as e:
                ok = False
                conn.error = str(e)
            if not ok:
                logger.warning(
                    f"MCP server '{entry.name}' failed to connect "
                    f"({conn.error or 'unknown error'}) — skipping it")
                await conn.stop()
                continue
            self.connections[entry.name] = conn
            self.tool_wrappers.extend(
                self.build_wrappers_for_entry(entry, conn.tools))

        if self.tool_wrappers:
            logger.info(
                f"MCP: {len(self.tool_wrappers)} tool(s) from "
                f"{len(self.connections)} server(s) ready for registration")

    def build_wrappers_for_entry(
            self, entry: MCPServerEntry, tools: List[Any]) -> List["MCPToolWrapper"]:
        """Build wrappers for the exposed subset of a server's tool list.

        Only tools on the ``expose`` allowlist are wrapped; duplicates against
        already-built MCP wrappers are skipped with a warning.
        """
        available = {getattr(t, "name", None): t for t in tools}
        taken = {w.name for w in self.tool_wrappers}
        wrappers: List[MCPToolWrapper] = []
        for tool_name in entry.expose:
            tool = available.get(tool_name)
            if tool is None:
                logger.warning(
                    f"MCP server '{entry.name}' does not provide exposed tool "
                    f"'{tool_name}' — skipped")
                continue
            wrapper = MCPToolWrapper(self, entry, tool)
            if wrapper.name in taken:
                logger.warning(
                    f"MCP tool name collision: {wrapper.name} already built — "
                    f"skipping duplicate from server '{entry.name}'")
                continue
            taken.add(wrapper.name)
            wrappers.append(wrapper)
        return wrappers

    async def call_tool(self, server: str, tool: str,
                        params: Optional[Dict[str, Any]] = None):
        """Call a tool on a connected server, bounded by the configured timeout.

        The per-call timeout covers the WHOLE operation — reconnect included —
        so a phone caller never sits through more than ~timeout_s of silence.

        - Session found dead BEFORE sending: ONE reconnect attempt (within the
          same time budget), then the call is sent once.
        - Per-request JSON-RPC errors (``McpError``, e.g. invalid params or an
          unknown tool): the server answered, so the session is healthy — the
          error message is surfaced for the model to correct its arguments;
          the connection is NOT torn down and nothing is retried.
        - Transport failure mid-flight: the request may already have reached
          the server, so it is never replayed (a non-idempotent tool could run
          twice); the dead connection is unwound in the background.
        """
        conn = self.connections.get(server)
        if conn is None:
            raise MCPToolError(f"the {server} service is not connected")
        timeout_s = conn.entry.timeout_s or float(
            getattr(self.config, "mcp_tool_timeout_s", 10.0))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s

        session = conn.session
        if session is None:
            # ONE reconnect attempt, bounded by the same per-call budget.
            logger.info(f"MCP server '{server}': reconnecting")
            try:
                ok = await asyncio.wait_for(conn.start(), timeout_s)
            except asyncio.TimeoutError:
                conn.abandon()
                raise MCPTimeoutError(
                    f"the {server} service took too long to respond")
            except Exception as e:
                conn.abandon()
                raise MCPToolError(
                    f"the {server} service is unavailable "
                    f"({str(e) or type(e).__name__})")
            session = conn.session
            if not ok or session is None:
                raise MCPToolError(
                    f"the {server} service is unavailable "
                    f"({conn.error or 'reconnect failed'})")

        remaining = deadline - loop.time()
        if remaining <= 0:
            raise MCPTimeoutError(
                f"the {server} service took too long to respond")
        try:
            return await asyncio.wait_for(
                session.call_tool(tool, arguments=params or None),
                remaining)
        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f"the {server} service took too long to respond")
        except MCPToolError:
            raise
        except McpError as e:
            # The server responded with a request-level JSON-RPC error (bad
            # arguments, unknown tool, ...). The shared session is healthy —
            # other in-flight calls must not be disturbed — and retrying the
            # same request would fail identically. Surface the message so the
            # model can fix its arguments.
            msg = str(e) or "request rejected"
            logger.warning(f"MCP call {server}/{tool} rejected: {msg}")
            raise MCPToolError(f"the {tool} tool reported an error: {msg}")
        except Exception as e:
            # Transport-level failure. The request may have reached the
            # server, so do NOT replay it — unwind the dead connection in the
            # background and fail fast with a spoken-friendly message.
            last_error = str(e) or type(e).__name__
            logger.warning(f"MCP call {server}/{tool} failed: {last_error}")
            conn.abandon()
            raise MCPToolError(
                f"the {server} service is unavailable ({last_error})")

    async def stop(self):
        """Close all sessions cleanly."""
        for conn in list(self.connections.values()):
            try:
                await conn.stop()
            except Exception as e:
                logger.debug(f"MCP connection stop error: {e}")
        self.connections.clear()
        self.tool_wrappers = []
        self._started = False


# =============================================================================
# Tool wrapper
# =============================================================================

class MCPToolWrapper(BaseTool):
    """Adapts one exposed MCP server tool to the BaseTool contract."""

    speak_result = True  # informational: result message is spoken in marker mode

    MAX_DESCRIPTION_CHARS = 200
    # ToolResult.message is spoken verbatim to the caller and stored in the
    # conversation history — an MCP server's output must be bounded on the
    # TTS path (built-in informational tools self-cap the same way).
    MAX_RESULT_CHARS = 1500
    _CONFIRM_PROPERTY = {
        "type": "boolean",
        "description": "Set true only after the caller has explicitly confirmed.",
    }

    def __init__(self, manager: MCPManager, entry: MCPServerEntry, tool: Any):
        super().__init__(None)  # MCP tools need no assistant back-reference
        self.manager = manager
        self.server_name = entry.name
        self.tool_name = getattr(tool, "name", "")
        self.name = sanitize_tool_name(entry.name, self.tool_name)

        desc = (getattr(tool, "description", None)
                or f"Tool {self.tool_name} from MCP server {entry.name}").strip()
        if len(desc) > self.MAX_DESCRIPTION_CHARS:
            desc = desc[:self.MAX_DESCRIPTION_CHARS - 3].rstrip() + "..."
        self.description = desc

        input_schema = getattr(tool, "inputSchema", None)
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}
        props = input_schema.get("properties")
        self._server_accepts_confirm = isinstance(props, dict) and "confirm" in props

        # Params flatten_input_schema advertises as "JSON object as a string":
        # the model sends them as JSON text, so execute() must json.loads them
        # back into real objects before the server validates its inputSchema.
        self._json_string_params = set()
        if isinstance(props, dict):
            for pname, spec in props.items():
                ptype = spec.get("type") if isinstance(spec, dict) else None
                if ptype not in _FLAT_TYPES:
                    self._json_string_params.add(pname)

        self.requires_confirm = self.tool_name in (entry.confirm or [])
        # Flat parameters for marker mode / prompt / /tools.
        self.parameters = flatten_input_schema(input_schema)
        # Full JSON schema for native/langgraph modes (_build_native_tools
        # uses it verbatim when present).
        self.json_schema = copy.deepcopy(input_schema)
        if self.requires_confirm:
            self.parameters.setdefault("confirm", {
                "type": "boolean",
                "description": self._CONFIRM_PROPERTY["description"],
                "required": False,
                "default": False,
            })
            schema_props = self.json_schema.setdefault("properties", {})
            if isinstance(schema_props, dict):
                schema_props.setdefault("confirm", dict(self._CONFIRM_PROPERTY))

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "yes", "1")

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        params = dict(params or {})

        # Confirmation gate (same pattern as CONTAINER_CTL restart).
        if self.requires_confirm and not self._truthy(params.get("confirm", False)):
            return ToolResult(
                status=ToolStatus.FAILED,
                message=(f"{self.tool_name} needs confirmation — ask the "
                         "caller and retry with confirm=true"),
            )
        if not self._server_accepts_confirm:
            params.pop("confirm", None)

        # Re-inflate params that were flattened to "JSON object as a string":
        # the server's inputSchema expects a real object/array there, not the
        # JSON text the model was told to send.
        for pname in self._json_string_params:
            value = params.get(pname)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text:
                continue
            try:
                params[pname] = json.loads(text)
            except ValueError:
                if text[0] in "{[":
                    # Looked like JSON but didn't parse (often a value the
                    # marker regex truncated). Tell the model so it can retry
                    # with a corrected argument instead of failing server-side.
                    return ToolResult(
                        status=ToolStatus.FAILED,
                        message=(f"The {pname} argument wasn't valid JSON — "
                                 "send it again as one complete JSON value."))
                # A plain string can be legitimate (e.g. anyOf including
                # string) — forward it as-is and let the server validate.

        try:
            result = await self.manager.call_tool(
                self.server_name, self.tool_name, params)
        except MCPTimeoutError:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=(f"Sorry, the {self.server_name} service took too "
                         "long to respond."))
        except MCPToolError as e:
            return ToolResult(status=ToolStatus.FAILED, message=f"Sorry, {e}.")
        except Exception as e:
            logger.error(f"MCP tool {self.name} failed: {e}")
            return ToolResult(
                status=ToolStatus.FAILED,
                message=(f"Sorry, I couldn't reach the {self.server_name} "
                         "service right now."))

        # Concatenate text content blocks into the spoken message, bounded by
        # MAX_RESULT_CHARS: the message is spoken verbatim to the caller and
        # kept in conversation history, so server output must not be unbounded.
        texts = []
        total = 0
        for block in (getattr(result, "content", None) or []):
            if getattr(block, "type", "") == "text":
                text = (getattr(block, "text", "") or "").strip()
                if text:
                    texts.append(text)
                    total += len(text) + 1
                    if total > self.MAX_RESULT_CHARS:
                        break
        message = " ".join(texts)
        if len(message) > self.MAX_RESULT_CHARS:
            logger.warning(
                f"MCP tool {self.name} returned an oversized result "
                f"({len(message)}+ chars) — truncating to "
                f"{self.MAX_RESULT_CHARS} for the spoken path")
            message = message[:self.MAX_RESULT_CHARS].rstrip() + "..."

        structured = getattr(result, "structuredContent", None)
        data = structured if isinstance(structured, dict) else {}

        if getattr(result, "isError", False):
            return ToolResult(
                status=ToolStatus.FAILED,
                message=message or f"The {self.tool_name} tool reported an error.",
                data=data)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message or "Done.",
            data=data)
