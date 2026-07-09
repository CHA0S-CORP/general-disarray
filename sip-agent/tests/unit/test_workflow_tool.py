"""Unit tests for the TRIGGER_WORKFLOW tool (plugins/workflow_tool.py)."""
import json
from types import SimpleNamespace

import pytest

from call_session import CallSession
from plugins.workflow_tool import TriggerWorkflowTool, _load_registry, _speakable
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


REGISTRY = {
    "lights_off": {"url": "http://n8n:5678/webhook/abc",
                   "description": "Turn off the living room lights"},
    "coffee": {"url": "http://n8n:5678/webhook/def",
               "description": "Start the coffee maker"},
}


def make_assistant(tmp_path, config_factory, registry=None, remote_uri=None):
    """Assistant stub whose data_dir points at tmp_path (holds workflows.json)."""
    if registry is not None:
        (tmp_path / "workflows.json").write_text(json.dumps(registry))
    cfg = config_factory(data_dir=str(tmp_path))
    call_info = SimpleNamespace(remote_uri=remote_uri) if remote_uri else None
    session = CallSession(call_info=call_info, direction="inbound",
                          transcript_id="t1")
    return SimpleNamespace(config=cfg, session=session)


# --- _load_registry ----------------------------------------------------------

def test_load_registry_missing_file(tmp_path):
    assert _load_registry(tmp_path / "workflows.json") == {}


def test_load_registry_invalid_json(tmp_path):
    path = tmp_path / "workflows.json"
    path.write_text("{not json")
    assert _load_registry(path) == {}


def test_load_registry_non_dict(tmp_path):
    path = tmp_path / "workflows.json"
    path.write_text(json.dumps(["lights_off"]))
    assert _load_registry(path) == {}


def test_load_registry_drops_malformed_entries(tmp_path):
    path = tmp_path / "workflows.json"
    path.write_text(json.dumps({
        "good": {"url": "http://n8n:5678/webhook/x"},
        "no_url": {"description": "missing url"},
        "not_a_dict": "http://n8n:5678/webhook/y",
    }))
    assert list(_load_registry(path)) == ["good"]


# --- prompt description ------------------------------------------------------

def test_prompt_description_lists_names(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory, registry=REGISTRY)
    tool = TriggerWorkflowTool(assistant)
    desc = tool.get_prompt_description()
    assert "lights_off (Turn off the living room lights)" in desc
    assert "coffee (Start the coffee maker)" in desc


def test_prompt_description_empty_registry_falls_back(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory, registry=None)
    tool = TriggerWorkflowTool(assistant)
    desc = tool.get_prompt_description()
    assert "TRIGGER_WORKFLOW" in desc
    assert TriggerWorkflowTool.description in desc
    assert "Available:" not in desc


# --- execute -----------------------------------------------------------------

async def test_unknown_name_lists_known(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory, registry=REGISTRY)
    tool = TriggerWorkflowTool(assistant)
    result = await tool.execute({"name": "garage"})
    assert result.status == ToolStatus.FAILED
    assert "garage" in result.message
    assert "lights off" in result.message  # spoken form of lights_off
    assert "coffee" in result.message
    assert result.data == {"workflow": "garage", "delivered": False}


async def test_happy_path_delivers_payload(tmp_path, config_factory, monkeypatch):
    assistant = make_assistant(tmp_path, config_factory, registry=REGISTRY,
                               remote_uri="sip:alice@example.com")
    tool = TriggerWorkflowTool(assistant)

    calls = []

    async def fake_deliver(url, payload, config, api_name="webhook"):
        calls.append({"url": url, "payload": payload, "api_name": api_name})
        return True

    # deliver_webhook is imported inside execute(), so patching the api
    # module attribute is enough.
    monkeypatch.setattr("api.deliver_webhook", fake_deliver)

    result = await tool.execute({"name": "lights_off", "message": "good night"})
    assert result.status == ToolStatus.SUCCESS
    assert "lights off" in result.message
    assert result.data == {"workflow": "lights_off", "delivered": True}
    assert "url" not in result.data

    assert len(calls) == 1
    assert calls[0]["url"] == "http://n8n:5678/webhook/abc"
    assert calls[0]["api_name"] == "workflow:lights_off"
    assert calls[0]["payload"] == {
        "workflow": "lights_off",
        "message": "good night",
        "caller": "sip:alice@example.com",
        "source": "sip-agent",
    }


async def test_delivery_failure_returns_failed(tmp_path, config_factory, monkeypatch):
    assistant = make_assistant(tmp_path, config_factory, registry=REGISTRY)
    tool = TriggerWorkflowTool(assistant)

    async def fake_deliver(url, payload, config, api_name="webhook"):
        return False

    monkeypatch.setattr("api.deliver_webhook", fake_deliver)

    result = await tool.execute({"name": "coffee"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I could not reach that automation."
    assert result.data == {"workflow": "coffee", "delivered": False}


async def test_missing_call_info_sends_empty_caller(tmp_path, config_factory,
                                                    monkeypatch):
    assistant = make_assistant(tmp_path, config_factory, registry=REGISTRY)
    tool = TriggerWorkflowTool(assistant)

    seen = {}

    async def fake_deliver(url, payload, config, api_name="webhook"):
        seen.update(payload)
        return True

    monkeypatch.setattr("api.deliver_webhook", fake_deliver)

    result = await tool.execute({"name": "coffee"})
    assert result.status == ToolStatus.SUCCESS
    assert seen["caller"] == ""


async def test_empty_name_fails(tmp_path, config_factory):
    assistant = make_assistant(tmp_path, config_factory, registry=REGISTRY)
    tool = TriggerWorkflowTool(assistant)
    result = await tool.execute({"name": "  "})
    assert result.status == ToolStatus.FAILED


# --- helpers -----------------------------------------------------------------

def test_speakable():
    assert _speakable("lights_off") == "lights off"
    assert _speakable("wake-up_call") == "wake up call"
