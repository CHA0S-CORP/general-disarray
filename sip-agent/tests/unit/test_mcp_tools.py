"""Unit tests for mcp_tools: name sanitization, schema flattening, the
confirm gate, servers-file parsing, wrapper build/collision rules, the
manager start() gates (disabled / package missing), and the call_tool
reconnect/error-classification logic.

Pure — no MCP server is spawned here (the real stdio round trip lives in
tests/component/test_mcp_client.py).
"""
import asyncio
import json
import re
from types import SimpleNamespace

import pytest

import mcp_tools
from mcp_tools import (
    MCP_AVAILABLE,
    MCPManager,
    MCPServerEntry,
    MCPToolError,
    MCPTimeoutError,
    MCPToolWrapper,
    McpError,
    flatten_input_schema,
    load_servers_file,
    sanitize_tool_name,
)
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


# --- name sanitization -------------------------------------------------------

def test_sanitize_basic():
    assert sanitize_tool_name("files", "read_file") == "FILES_READ_FILE"


def test_sanitize_replaces_non_word_chars():
    assert (sanitize_tool_name("home-assistant", "turn.on light")
            == "HOME_ASSISTANT_TURN_ON_LIGHT")


def test_sanitized_names_are_marker_parser_safe():
    # The marker parser regex is \[TOOL:(\w+)...] — names must be \w only.
    ugly = sanitize_tool_name("srv!@#", "tool/with:stuff")
    assert re.fullmatch(r"\w+", ugly)


# --- schema flattening -------------------------------------------------------

def test_flatten_scalar_types_pass_through():
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer"},
            "temp": {"type": "number"},
            "metric": {"type": "boolean"},
        },
        "required": ["city"],
    }
    params = flatten_input_schema(schema)
    assert params["city"] == {"type": "string", "description": "City name",
                              "required": True}
    assert params["days"]["type"] == "integer"
    assert params["temp"]["type"] == "number"
    assert params["metric"]["type"] == "boolean"
    assert "required" not in params["days"]


def test_flatten_nested_becomes_string():
    schema = {
        "type": "object",
        "properties": {
            "filters": {"type": "object", "properties": {"a": {"type": "string"}},
                        "description": "Search filters"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "anything": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        },
    }
    params = flatten_input_schema(schema)
    assert params["filters"]["type"] == "string"
    assert "JSON object as a string" in params["filters"]["description"]
    assert "Search filters" in params["filters"]["description"]
    assert params["tags"]["type"] == "string"
    assert params["anything"]["type"] == "string"
    assert params["anything"]["description"] == "JSON object as a string"


def test_flatten_handles_garbage():
    assert flatten_input_schema(None) == {}
    assert flatten_input_schema("nope") == {}
    assert flatten_input_schema({"type": "object"}) == {}
    assert flatten_input_schema({"properties": "not-a-dict"}) == {}


# --- servers-file parsing ----------------------------------------------------

def test_load_missing_file(tmp_path):
    assert load_servers_file(tmp_path / "nope.json") == []


def test_load_bad_json(tmp_path):
    path = tmp_path / "mcp_servers.json"
    path.write_text("{not json")
    assert load_servers_file(path) == []


def test_load_non_list(tmp_path):
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps({"name": "x"}))
    assert load_servers_file(path) == []


def test_entry_without_expose_registers_zero_tools(tmp_path, caplog):
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps([
        {"name": "noexpose", "transport": "stdio", "command": "srv"},
        {"name": "emptyexpose", "transport": "stdio", "command": "srv",
         "expose": []},
    ]))
    with caplog.at_level("WARNING"):
        assert load_servers_file(path) == []
    assert "expose" in caplog.text


def test_load_valid_entries(tmp_path):
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps([
        {"name": "files", "transport": "stdio", "command": "mcp-server-files",
         "args": ["--root", "/data"], "expose": ["read_file"],
         "confirm": ["read_file"], "timeout_s": 3, "env": {"K": "V"}},
        {"name": "ha", "transport": "streamable_http",
         "url": "http://ha:8123/mcp", "expose": ["get_state"]},
        # invalid: stdio without command
        {"name": "broken", "transport": "stdio", "expose": ["x"]},
        # invalid: http without url
        {"name": "nourl", "transport": "http", "expose": ["x"]},
        # invalid: unknown transport
        {"name": "weird", "transport": "carrier-pigeon", "expose": ["x"]},
        # duplicate name: keep first
        {"name": "files", "transport": "stdio", "command": "other",
         "expose": ["y"]},
    ]))
    entries = load_servers_file(path)
    assert [e.name for e in entries] == ["files", "ha"]
    files = entries[0]
    assert files.command == "mcp-server-files"
    assert files.args == ["--root", "/data"]
    assert files.expose == ["read_file"]
    assert files.confirm == ["read_file"]
    assert files.timeout_s == 3.0
    assert files.env == {"K": "V"}
    ha = entries[1]
    assert ha.transport == "http"  # streamable_http normalized
    assert ha.url == "http://ha:8123/mcp"
    assert ha.timeout_s is None


# --- wrapper build + confirm gate --------------------------------------------

class FakeManager:
    """Stands in for MCPManager in wrapper tests."""

    def __init__(self, result=None, exc=None):
        self.calls = []
        self.result = result
        self.exc = exc

    async def call_tool(self, server, tool, params=None):
        self.calls.append((server, tool, dict(params or {})))
        if self.exc is not None:
            raise self.exc
        return self.result


def _tool(name="turn_on", description="Turn on a light", schema=None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema if schema is not None else {
            "type": "object",
            "properties": {"entity": {"type": "string"}},
            "required": ["entity"],
        },
    )


def _entry(**kw):
    defaults = dict(name="ha", transport="http", url="http://ha/mcp",
                    expose=["turn_on"], confirm=[])
    defaults.update(kw)
    return MCPServerEntry(**defaults)


def _ok_result(*texts, structured=None, is_error=False):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=t) for t in texts],
        structuredContent=structured,
        isError=is_error,
    )


def test_wrapper_attributes():
    wrapper = MCPToolWrapper(FakeManager(), _entry(), _tool())
    assert wrapper.name == "HA_TURN_ON"
    assert wrapper.speak_result is True
    assert wrapper.parameters["entity"] == {"type": "string", "required": True}
    assert wrapper.json_schema["properties"]["entity"] == {"type": "string"}


def test_wrapper_description_truncated():
    tool = _tool(description="x" * 500)
    wrapper = MCPToolWrapper(FakeManager(), _entry(), tool)
    assert len(wrapper.description) <= 200
    assert wrapper.description.endswith("...")


async def test_confirm_gate_refuses_without_confirm():
    manager = FakeManager(result=_ok_result("on"))
    wrapper = MCPToolWrapper(manager, _entry(confirm=["turn_on"]), _tool())
    result = await wrapper.execute({"entity": "light.porch"})
    assert result.status == ToolStatus.FAILED
    assert "confirm=true" in result.message
    assert manager.calls == []  # never reached the server


async def test_confirm_gate_passes_with_confirm():
    manager = FakeManager(result=_ok_result("light is on"))
    wrapper = MCPToolWrapper(manager, _entry(confirm=["turn_on"]), _tool())
    result = await wrapper.execute({"entity": "light.porch", "confirm": "true"})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "light is on"
    # confirm is a gate param, not a server param — stripped before the call.
    assert manager.calls == [("ha", "turn_on", {"entity": "light.porch"})]


def test_confirm_gated_tool_advertises_confirm_param():
    wrapper = MCPToolWrapper(FakeManager(), _entry(confirm=["turn_on"]), _tool())
    assert wrapper.parameters["confirm"]["type"] == "boolean"
    assert "confirm" in wrapper.json_schema["properties"]


async def test_execute_concatenates_text_and_structured_content():
    manager = FakeManager(result=_ok_result("part one", "part two",
                                            structured={"n": 2}))
    wrapper = MCPToolWrapper(manager, _entry(), _tool())
    result = await wrapper.execute({"entity": "e"})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "part one part two"
    assert result.data == {"n": 2}


async def test_execute_is_error_maps_to_failed():
    manager = FakeManager(result=_ok_result("kaboom", is_error=True))
    wrapper = MCPToolWrapper(manager, _entry(), _tool())
    result = await wrapper.execute({"entity": "e"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "kaboom"


async def test_execute_timeout_is_spoken_friendly():
    manager = FakeManager(exc=MCPTimeoutError("the ha service took too long"))
    wrapper = MCPToolWrapper(manager, _entry(), _tool())
    result = await wrapper.execute({"entity": "e"})
    assert result.status == ToolStatus.FAILED
    assert "too long" in result.message


async def test_execute_mcp_error_is_spoken_friendly():
    manager = FakeManager(exc=MCPToolError("the ha service is unavailable"))
    wrapper = MCPToolWrapper(manager, _entry(), _tool())
    result = await wrapper.execute({"entity": "e"})
    assert result.status == ToolStatus.FAILED
    assert "unavailable" in result.message


# --- wrapper-set build rules (expose + collisions) ----------------------------

def _manager():
    return MCPManager(SimpleNamespace(mcp_enabled=True, mcp_tool_timeout_s=10))


def test_build_wrappers_respects_expose_allowlist(caplog):
    manager = _manager()
    entry = _entry(expose=["turn_on", "ghost_tool"])
    tools = [_tool("turn_on"), _tool("secret")]
    with caplog.at_level("WARNING"):
        wrappers = manager.build_wrappers_for_entry(entry, tools)
    assert [w.name for w in wrappers] == ["HA_TURN_ON"]  # secret not exposed
    assert "ghost_tool" in caplog.text  # exposed-but-missing warned


def test_build_wrappers_skips_duplicate_names():
    manager = _manager()
    entry = _entry(expose=["turn_on"])
    first = manager.build_wrappers_for_entry(entry, [_tool("turn_on")])
    manager.tool_wrappers.extend(first)
    # A second server whose sanitized name collides is skipped.
    entry2 = _entry(name="ha", expose=["turn.on"])
    dupes = manager.build_wrappers_for_entry(entry2, [_tool("turn.on")])
    assert dupes == []


# --- JSON-string param re-inflation (marker mode nested params) ---------------

def _nested_tool():
    return SimpleNamespace(
        name="set_scene",
        description="Set a scene",
        inputSchema={
            "type": "object",
            "properties": {
                "options": {"type": "object",
                            "properties": {"a": {"type": "integer"}}},
                "label": {"type": "string"},
                "anyval": {"anyOf": [{"type": "string"},
                                     {"type": "integer"}]},
            },
        },
    )


async def test_json_string_params_decoded_before_call():
    manager = FakeManager(result=_ok_result("ok"))
    wrapper = MCPToolWrapper(manager, _entry(expose=["set_scene"]),
                             _nested_tool())
    result = await wrapper.execute({"options": '{"a": 1}',
                                    "label": '{"not": "decoded"}'})
    assert result.status == ToolStatus.SUCCESS
    _, _, params = manager.calls[0]
    # The nested param (advertised as "JSON object as a string") is decoded
    # back into a real object before it reaches the server…
    assert params["options"] == {"a": 1}
    # …while a genuinely string-typed param is forwarded verbatim.
    assert params["label"] == '{"not": "decoded"}'


async def test_truncated_json_param_fails_with_guidance():
    manager = FakeManager(result=_ok_result("ok"))
    wrapper = MCPToolWrapper(manager, _entry(expose=["set_scene"]),
                             _nested_tool())
    result = await wrapper.execute({"options": '{"a": [1, 2'})
    assert result.status == ToolStatus.FAILED
    assert "JSON" in result.message
    assert manager.calls == []  # never sent a knowingly-broken argument


async def test_plain_string_for_anyof_param_forwarded_as_is():
    manager = FakeManager(result=_ok_result("ok"))
    wrapper = MCPToolWrapper(manager, _entry(expose=["set_scene"]),
                             _nested_tool())
    result = await wrapper.execute({"anyval": "hello"})
    assert result.status == ToolStatus.SUCCESS
    assert manager.calls[0][2]["anyval"] == "hello"


# --- result-size cap on the spoken path ----------------------------------------

async def test_result_message_capped_for_spoken_path():
    huge = "word " * 5000  # ~25k chars from a verbose/malicious server
    manager = FakeManager(result=_ok_result(huge.strip()))
    wrapper = MCPToolWrapper(manager, _entry(), _tool())
    result = await wrapper.execute({"entity": "e"})
    assert result.status == ToolStatus.SUCCESS
    assert len(result.message) <= MCPToolWrapper.MAX_RESULT_CHARS + 3
    assert result.message.endswith("...")


async def test_short_result_message_not_truncated():
    manager = FakeManager(result=_ok_result("all good"))
    wrapper = MCPToolWrapper(manager, _entry(), _tool())
    result = await wrapper.execute({"entity": "e"})
    assert result.message == "all good"


# --- manager start() gates ------------------------------------------------------

async def test_start_disabled_is_noop_without_touching_servers_file(monkeypatch):
    loads = []
    monkeypatch.setattr(mcp_tools, "load_servers_file",
                        lambda p: loads.append(p) or [])
    # The config deliberately has NO mcp_servers_file attribute: any access
    # hoisted above the enabled check would raise AttributeError here.
    manager = MCPManager(SimpleNamespace(mcp_enabled=False))
    await manager.start()
    assert manager.connections == {}
    assert manager.tool_wrappers == []
    assert loads == []


async def test_start_without_mcp_package_warns_and_registers_nothing(
        monkeypatch, caplog):
    loads = []
    monkeypatch.setattr(mcp_tools, "load_servers_file",
                        lambda p: loads.append(p) or [])
    monkeypatch.setattr(mcp_tools, "MCP_AVAILABLE", False)
    manager = MCPManager(SimpleNamespace(mcp_enabled=True))
    with caplog.at_level("WARNING"):
        await manager.start()
    assert "not installed" in caplog.text  # the only diagnostic — keep it
    assert manager.connections == {}
    assert manager.tool_wrappers == []
    assert loads == []


# --- call_tool: timeout budget, reconnect, and error classification -------------

def _mcp_error(msg):
    """A real SDK McpError when mcp is installed, the placeholder otherwise."""
    if MCP_AVAILABLE:
        from mcp import types
        return McpError(types.ErrorData(code=types.INVALID_PARAMS, message=msg))
    return McpError(msg)


class FakeSession:
    def __init__(self, result="ok", exc=None, delay=0.0):
        self.result = result
        self.exc = exc
        self.delay = delay
        self.calls = []

    async def call_tool(self, tool, arguments=None):
        self.calls.append((tool, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeConn:
    def __init__(self, entry, session=None, start_ok=True, start_delay=0.0,
                 start_session=None):
        self.entry = entry
        self.session = session
        self.error = None
        self.start_ok = start_ok
        self.start_delay = start_delay
        self.start_session = start_session
        self.start_calls = 0
        self.abandon_calls = 0

    async def start(self):
        self.start_calls += 1
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        if self.start_ok:
            self.session = self.start_session or FakeSession()
            return True
        self.error = "spawn failed"
        return False

    def abandon(self):
        self.abandon_calls += 1
        self.session = None


def _mgr_with_conn(conn):
    manager = MCPManager(SimpleNamespace(mcp_enabled=True,
                                         mcp_tool_timeout_s=0.5))
    manager.connections[conn.entry.name] = conn
    return manager


async def test_call_tool_unknown_server():
    manager = MCPManager(SimpleNamespace(mcp_enabled=True,
                                         mcp_tool_timeout_s=0.5))
    with pytest.raises(MCPToolError, match="not connected"):
        await manager.call_tool("ghost", "t", {})


async def test_call_tool_happy_path_no_reconnect():
    sess = FakeSession(result="fine")
    conn = FakeConn(_entry(name="srv"), session=sess)
    manager = _mgr_with_conn(conn)
    assert await manager.call_tool("srv", "t", {"a": 1}) == "fine"
    assert conn.start_calls == 0
    assert conn.abandon_calls == 0


async def test_dead_session_gets_one_reconnect_then_call():
    conn = FakeConn(_entry(name="srv"), session=None,
                    start_session=FakeSession(result="back"))
    manager = _mgr_with_conn(conn)
    assert await manager.call_tool("srv", "t") == "back"
    assert conn.start_calls == 1


async def test_reconnect_failure_is_spoken_friendly_and_single():
    conn = FakeConn(_entry(name="srv"), session=None, start_ok=False)
    manager = _mgr_with_conn(conn)
    with pytest.raises(MCPToolError, match="unavailable"):
        await manager.call_tool("srv", "t")
    assert conn.start_calls == 1  # ONE attempt, not a retry loop


async def test_reconnect_is_bounded_by_the_call_timeout():
    # A hung reconnect must cost ~timeout_s of dead air, not CONNECT_TIMEOUT_S
    # (15s) plus a 10s stop wait.
    conn = FakeConn(_entry(name="srv", timeout_s=0.2), session=None,
                    start_delay=30.0)
    manager = _mgr_with_conn(conn)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(MCPTimeoutError):
        await manager.call_tool("srv", "t")
    assert loop.time() - t0 < 2.0
    assert conn.abandon_calls == 1  # unwound in the background, not awaited


async def test_request_level_mcp_error_keeps_session_alive():
    # A JSON-RPC request error (e.g. Invalid params from LLM-built arguments)
    # means the server ANSWERED: the shared session must not be torn down
    # (that would abort concurrent calls) and the message must surface so the
    # model can fix its arguments.
    sess = FakeSession(exc=_mcp_error("Invalid params"))
    conn = FakeConn(_entry(name="srv"), session=sess)
    manager = _mgr_with_conn(conn)
    with pytest.raises(MCPToolError, match="Invalid params"):
        await manager.call_tool("srv", "t", {"bad": "arg"})
    assert conn.session is sess  # session untouched
    assert conn.abandon_calls == 0
    assert conn.start_calls == 0
    assert len(sess.calls) == 1  # deterministic error: no pointless retry


async def test_transport_error_never_replays_the_request():
    # A mid-flight transport failure may have already delivered the request —
    # replaying it could execute a non-idempotent tool twice server-side.
    sess = FakeSession(exc=ConnectionResetError("connection reset"))
    conn = FakeConn(_entry(name="srv"), session=sess)
    manager = _mgr_with_conn(conn)
    with pytest.raises(MCPToolError, match="unavailable"):
        await manager.call_tool("srv", "t", {"confirm": True})
    assert len(sess.calls) == 1   # NOT re-sent
    assert conn.start_calls == 0  # no reconnect-and-replay
    assert conn.abandon_calls == 1


async def test_slow_call_times_out():
    sess = FakeSession(delay=30.0)
    conn = FakeConn(_entry(name="srv", timeout_s=0.2), session=sess)
    manager = _mgr_with_conn(conn)
    with pytest.raises(MCPTimeoutError):
        await manager.call_tool("srv", "t")
