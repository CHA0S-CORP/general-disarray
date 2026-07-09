"""Unit tests for the REMEMBER/FORGET caller-memory tools, exercised against
a real CallerMemoryStore over a temporary data directory."""
import pytest

from types import SimpleNamespace

from call_session import CallSession
from caller_memory import CallerMemoryStore
from plugins.memory_tools import RememberTool, ForgetTool
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def make_assistant(tmp_path, config_factory, remote_uri="sip:1001@pbx",
                   **config_overrides):
    """Assistant stub with a live session and a real store over tmp_path."""
    cfg = config_factory(data_dir=str(tmp_path), **config_overrides)
    store = CallerMemoryStore(cfg)
    session = CallSession(call_info=SimpleNamespace(remote_uri=remote_uri),
                          direction="inbound", transcript_id="t1")
    return SimpleNamespace(config=cfg, session=session, caller_memory=store)


# --- REMEMBER ----------------------------------------------------------------

async def test_remember_persists_fact(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory)
    tool = RememberTool(assistant)

    result = await tool.execute({"fact": "Prefers morning callbacks"})

    assert result.status == ToolStatus.SUCCESS
    assert result.message == "Got it. I will remember that."
    assert result.data == {"caller": "1001", "fact": "Prefers morning callbacks"}
    record = assistant.caller_memory.get("1001")
    assert record is not None
    assert "Prefers morning callbacks" in record["facts"]


async def test_remember_refreshes_live_prompt(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory)
    tool = RememberTool(assistant)
    assert assistant.session.caller_memory_prompt == ""

    await tool.execute({"fact": "Name is Bob"})

    assert "Name is Bob" in assistant.session.caller_memory_prompt


async def test_remember_empty_fact_fails(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory)
    tool = RememberTool(assistant)

    result = await tool.execute({"fact": "   "})

    assert result.status == ToolStatus.FAILED
    assert assistant.caller_memory.get("1001") is None


# --- FORGET ------------------------------------------------------------------

async def test_forget_removes_matching_facts_with_count(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory)
    store = assistant.caller_memory
    store.add_fact("1001", "Likes coffee black")
    store.add_fact("1001", "Drinks coffee every morning")
    store.add_fact("1001", "Name is Bob")
    tool = ForgetTool(assistant)

    result = await tool.execute({"what": "coffee"})

    assert result.status == ToolStatus.SUCCESS
    assert result.data["removed"] == 2
    assert "2" in result.message  # multiple removals mention the count
    assert store.get("1001")["facts"] == ["Name is Bob"]
    # The live prompt no longer mentions the forgotten facts.
    assert "coffee" not in assistant.session.caller_memory_prompt
    assert "Name is Bob" in assistant.session.caller_memory_prompt


async def test_forget_single_fact_message(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory)
    assistant.caller_memory.add_fact("1001", "Has a dog named Rex")
    tool = ForgetTool(assistant)

    result = await tool.execute({"what": "dog"})

    assert result.status == ToolStatus.SUCCESS
    assert result.message == "Done - I have forgotten that."
    assert result.data["removed"] == 1


async def test_forget_no_match_is_success(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory)
    assistant.caller_memory.add_fact("1001", "Name is Bob")
    tool = ForgetTool(assistant)

    result = await tool.execute({"what": "spaceships"})

    assert result.status == ToolStatus.SUCCESS
    assert "did not have anything" in result.message
    assert result.data["removed"] == 0
    assert assistant.caller_memory.get("1001")["facts"] == ["Name is Bob"]


# --- Guards ------------------------------------------------------------------

async def test_no_session_fails(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory)
    assistant.session = None  # between calls
    for tool in (RememberTool(assistant), ForgetTool(assistant)):
        result = await tool.execute({"fact": "x", "what": "x"})
        assert result.status == ToolStatus.FAILED
        assert result.message == "I can only do that during a call."


async def test_memory_disabled_fails(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory,
                               caller_memory_enabled="false")
    tool = RememberTool(assistant)

    assert tool.enabled is False  # self-disables at construction
    result = await tool.execute({"fact": "Name is Bob"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "Memory is not enabled."


async def test_unusable_remote_uri_fails(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory,
                               remote_uri="sip:../../etc/passwd@pbx")
    tool = RememberTool(assistant)

    result = await tool.execute({"fact": "Name is Bob"})

    assert result.status == ToolStatus.FAILED
    assert result.message == "I do not know who I am speaking with."
