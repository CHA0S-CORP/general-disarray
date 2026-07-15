"""Unit tests for the CONTAINER_CTL tool (allowlist gating, docker API paths)."""
import pytest

from types import SimpleNamespace

import plugins.container_tool as container_tool
from plugins.container_tool import ContainerControlTool, _allowlist
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def make_tool(monkeypatch, config_factory, allowlist="n8n,redis"):
    """Build an enabled tool: allowlist populated, docker socket 'present'."""
    monkeypatch.setattr(container_tool.os.path, "exists", lambda p: True)
    cfg = config_factory(container_ctl_allowlist=allowlist)
    assistant = SimpleNamespace(config=cfg, session=None)
    return ContainerControlTool(assistant)


def forbid_docker_request(monkeypatch):
    """Install a _docker_request that fails the test if ever invoked."""
    async def _fail(*args, **kwargs):
        pytest.fail("_docker_request must not be called")
    monkeypatch.setattr(container_tool, "_docker_request", _fail)


# --- _allowlist helper -------------------------------------------------------

def test_allowlist_strips_spaces_and_drops_empties():
    cfg = SimpleNamespace(container_ctl_allowlist=" n8n , redis ,, vllm ,")
    assert _allowlist(cfg) == ["n8n", "redis", "vllm"]


def test_allowlist_empty_or_missing_config():
    assert _allowlist(None) == []
    assert _allowlist(SimpleNamespace(container_ctl_allowlist="")) == []


# --- self-disable ------------------------------------------------------------

def test_disabled_when_allowlist_empty(monkeypatch, config_factory):
    monkeypatch.setattr(container_tool.os.path, "exists", lambda p: True)
    cfg = config_factory(container_ctl_allowlist="")
    tool = ContainerControlTool(SimpleNamespace(config=cfg, session=None))
    assert tool.enabled is False


def test_disabled_when_socket_missing(monkeypatch, config_factory):
    monkeypatch.setattr(container_tool.os.path, "exists", lambda p: False)
    cfg = config_factory(container_ctl_allowlist="n8n,redis")
    tool = ContainerControlTool(SimpleNamespace(config=cfg, session=None))
    assert tool.enabled is False


def test_enabled_when_configured(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)
    assert tool.enabled is True


# --- allowlist enforcement ---------------------------------------------------

async def test_name_not_in_allowlist_fails_without_io(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)
    forbid_docker_request(monkeypatch)
    result = await tool.execute({"action": "status", "name": "postgres"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I am not allowed to touch that container."
    # The spoken message must never leak the allowlist contents
    assert "n8n" not in result.message and "redis" not in result.message


async def test_allowlist_match_is_case_sensitive(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)
    forbid_docker_request(monkeypatch)
    result = await tool.execute({"action": "status", "name": "N8N"})
    assert result.status == ToolStatus.FAILED


# --- status ------------------------------------------------------------------

async def test_status_happy_path_with_health(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)
    calls = []

    async def fake_request(socket_path, method, path, params=None):
        calls.append((method, path))
        return 200, {"State": {"Status": "running",
                               "Health": {"Status": "healthy"}}}

    monkeypatch.setattr(container_tool, "_docker_request", fake_request)
    result = await tool.execute({"action": "status", "name": "n8n"})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "The n8n container is running. Health is healthy."
    assert result.data["status"] == "running"
    assert calls == [("GET", "/containers/n8n/json")]


async def test_status_without_health(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)

    async def fake_request(socket_path, method, path, params=None):
        return 200, {"State": {"Status": "exited"}}

    monkeypatch.setattr(container_tool, "_docker_request", fake_request)
    result = await tool.execute({"action": "status", "name": "redis"})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "The redis container is exited."


async def test_status_container_not_found(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)

    async def fake_request(socket_path, method, path, params=None):
        return 404, {"message": "No such container"}

    monkeypatch.setattr(container_tool, "_docker_request", fake_request)
    result = await tool.execute({"action": "status", "name": "n8n"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I do not see a container called n8n."


# --- restart -----------------------------------------------------------------

async def test_restart_without_confirm_makes_no_request(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)
    forbid_docker_request(monkeypatch)
    result = await tool.execute({"action": "restart", "name": "n8n"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "Tell me to confirm and I will restart n8n."
    assert result.data["confirmed"] is False


async def test_restart_with_confirm_succeeds(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)
    calls = []

    async def fake_request(socket_path, method, path, params=None):
        calls.append((method, path, params))
        return 204, None

    monkeypatch.setattr(container_tool, "_docker_request", fake_request)
    result = await tool.execute({"action": "restart", "name": "n8n",
                                 "confirm": True})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "Restarting n8n now."
    assert result.data["confirmed"] is True
    assert calls == [("POST", "/containers/n8n/restart", {"t": 10})]


# --- failure modes -----------------------------------------------------------

async def test_socket_unreachable_fails(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)

    async def fake_request(socket_path, method, path, params=None):
        return 0, None

    monkeypatch.setattr(container_tool, "_docker_request", fake_request)
    result = await tool.execute({"action": "status", "name": "n8n"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I cannot reach the docker daemon."


async def test_unknown_action_fails(monkeypatch, config_factory):
    tool = make_tool(monkeypatch, config_factory)
    forbid_docker_request(monkeypatch)
    result = await tool.execute({"action": "destroy", "name": "n8n"})
    assert result.status == ToolStatus.FAILED
