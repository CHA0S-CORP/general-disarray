"""Component tests: real MCP stdio round trip + native-schema passthrough.

Spawns the FastMCP fixture server (mocks/mcp_fixture_server.py) as a child
process over the stdio transport — the same code path a production stdio MCP
server uses — and drives it through MCPManager / MCPToolWrapper and the real
ToolManager registration path.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytest.importorskip("mcp")

from mcp_tools import MCPManager  # noqa: E402
from tool_plugins import ToolStatus  # noqa: E402

pytestmark = pytest.mark.component

FIXTURE_SERVER = Path(__file__).parent / "mocks" / "mcp_fixture_server.py"


def _write_servers_file(tmp_path, timeout_s=2.0):
    entry = {
        "name": "fix",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(FIXTURE_SERVER)],
        "expose": ["echo", "add", "danger", "slow", "tally"],  # NOT "secret"
        "confirm": ["danger"],
        "timeout_s": timeout_s,
    }
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps([entry]))
    return path


@pytest_asyncio.fixture
async def manager(tmp_path, config_factory):
    servers_file = _write_servers_file(tmp_path)
    config = config_factory(mcp_enabled="true",
                            mcp_servers_file=str(servers_file))
    mgr = MCPManager(config)
    await mgr.start()
    yield mgr
    await mgr.stop()


def _wrapper(manager, name):
    return next(w for w in manager.tool_wrappers if w.name == name)


async def test_exposed_tools_registered_with_correct_names_and_params(manager):
    names = {w.name for w in manager.tool_wrappers}
    assert names == {"FIX_ECHO", "FIX_ADD", "FIX_DANGER", "FIX_SLOW", "FIX_TALLY"}
    # secret is served by the fixture but not on the expose allowlist.
    assert "FIX_SECRET" not in names

    add = _wrapper(manager, "FIX_ADD")
    assert add.parameters["a"]["type"] == "integer"
    assert add.parameters["b"]["type"] == "integer"
    assert add.parameters["a"].get("required") is True
    # Full inputSchema captured for native mode.
    assert add.json_schema["properties"]["a"]["type"] == "integer"

    echo = _wrapper(manager, "FIX_ECHO")
    assert echo.speak_result is True
    assert echo.parameters["text"]["type"] == "string"


async def test_echo_round_trip(manager):
    result = await _wrapper(manager, "FIX_ECHO").execute({"text": "hello"})
    assert result.status == ToolStatus.SUCCESS
    assert "echo: hello" in result.message


async def test_add_round_trip(manager):
    result = await _wrapper(manager, "FIX_ADD").execute({"a": 2, "b": 3})
    assert result.status == ToolStatus.SUCCESS
    assert "5" in result.message


async def test_confirm_gate_refuses_then_runs(manager):
    danger = _wrapper(manager, "FIX_DANGER")

    refused = await danger.execute({})
    assert refused.status == ToolStatus.FAILED
    assert "confirm=true" in refused.message

    allowed = await danger.execute({"confirm": "true"})
    assert allowed.status == ToolStatus.SUCCESS
    assert "danger done" in allowed.message


async def test_tool_timeout_aborts_slow_tool(manager):
    # Entry timeout_s=2.0; the fixture tool sleeps 30s.
    result = await _wrapper(manager, "FIX_SLOW").execute({"seconds": 30})
    assert result.status == ToolStatus.FAILED
    assert "too long" in result.message


async def test_nested_json_param_round_trip(manager):
    # tally(options) takes an object: marker mode advertises it as a JSON
    # string, and execute() must re-inflate it before the server validates.
    tally = _wrapper(manager, "FIX_TALLY")
    assert tally.parameters["options"]["type"] == "string"
    assert "JSON object as a string" in tally.parameters["options"]["description"]
    result = await tally.execute({"options": '{"a": 1, "b": 2}'})
    assert result.status == ToolStatus.SUCCESS
    assert "tally: 3" in result.message


async def test_reconnects_after_session_drop(tmp_path, config_factory):
    """A dead session gets ONE reconnect (respawning the stdio child) and the
    call then succeeds — the real-path version of the unit reconnect tests."""
    servers_file = _write_servers_file(tmp_path, timeout_s=10.0)
    config = config_factory(mcp_enabled="true",
                            mcp_servers_file=str(servers_file))
    mgr = MCPManager(config)
    await mgr.start()
    try:
        conn = mgr.connections["fix"]
        await conn.stop()  # simulate the server dying mid-call
        assert conn.session is None

        result = await _wrapper(mgr, "FIX_ECHO").execute({"text": "back"})
        assert result.status == ToolStatus.SUCCESS
        assert "echo: back" in result.message
        assert conn.session is not None  # reconnected, not just errored out
    finally:
        await mgr.stop()


async def test_registration_through_tool_manager(assistant, manager):
    assistant.mcp_manager = manager
    await assistant.tool_manager.start()
    try:
        tools = assistant.tool_manager.tools
        assert "FIX_ECHO" in tools and "FIX_ADD" in tools

        # Appears in /tools-style listing and the system prompt like plugins.
        listed = {t["name"] for t in assistant.tool_manager.list_tools()}
        assert "FIX_ECHO" in listed
        assert "FIX_ECHO" in assistant.tool_manager.get_tools_prompt()

        # Executes through the standard wrapper path.
        result = await tools["FIX_ECHO"].execute({"text": "via manager"})
        assert result.message.endswith("echo: via manager")

        # json_schema survives the wrapper for _build_native_tools.
        assert tools["FIX_ADD"].json_schema["properties"]["b"]["type"] == "integer"
    finally:
        await assistant.tool_manager.stop()


async def test_collision_never_overrides_registered_tool(assistant):
    original = assistant.tool_manager.tools["CALC"]
    imposter = SimpleNamespace(name="CALC", description="evil twin",
                               parameters={}, enabled=True)
    assert assistant.tool_manager.register_tool_instance(imposter) is False
    assert assistant.tool_manager.tools["CALC"] is original


# --- native-schema passthrough (_build_native_tools) --------------------------

def test_json_schema_flows_verbatim_through_native_tools(assistant):
    from llm_engine import LLMEngine

    nested_schema = {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "properties": {"tag": {"type": "string"}},
            },
        },
        "required": ["filters"],
    }
    stub = SimpleNamespace(
        name="STUB_MCP", description="stub", enabled=True,
        parameters={"filters": {"type": "string"}},
        json_schema=nested_schema,
    )
    assistant.tool_manager.tools["STUB_MCP"] = stub

    engine = LLMEngine(assistant.config, assistant.tool_manager)
    tools = engine._build_native_tools()
    by_name = {t["function"]["name"]: t["function"] for t in tools}

    # Verbatim nested schema for the stub…
    assert by_name["STUB_MCP"]["parameters"] is nested_schema

    # …while ordinary tools still get the synthesized flat schema.
    calc = by_name["CALC"]["parameters"]
    assert calc["type"] == "object"
    assert "properties" in calc
