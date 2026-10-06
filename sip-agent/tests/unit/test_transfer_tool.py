"""Unit tests for the TRANSFER tool (blind transfer via SIP REFER)."""
import pytest
from types import SimpleNamespace

import api
from plugins.transfer_tool import TransferTool
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def make_assistant(cfg, transfer_result=True, with_handler=True, with_call=True):
    """Build a stub assistant with a recording transfer_call stub."""
    calls = []

    async def transfer_call(call_info, target):
        calls.append((call_info, target))
        return transfer_result

    if with_handler:
        handler = SimpleNamespace(transfer_call=transfer_call)
    else:
        handler = SimpleNamespace()  # no transfer_call attribute

    assistant = SimpleNamespace(
        config=cfg,
        current_call=SimpleNamespace(call_id="c1") if with_call else None,
        sip_handler=handler,
    )
    return assistant, calls


async def test_no_active_call_fails(config_factory):
    assistant, calls = make_assistant(config_factory(), with_call=False)
    tool = TransferTool(assistant)
    result = await tool.execute({"extension": "2001"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "There is no active call to transfer."
    assert calls == []


async def test_disallowed_extension_fails(config_factory, monkeypatch):
    monkeypatch.setattr(api, "check_extension_allowed",
                        lambda extension, config: "extension does not match the allowed pattern")
    assistant, calls = make_assistant(config_factory())
    tool = TransferTool(assistant)
    result = await tool.execute({"extension": "9999"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I cannot transfer to that number."
    assert calls == []


async def test_allowed_extension_builds_target_and_transfers(config_factory):
    cfg = config_factory(sip_domain="pbx.example.com")
    assistant, calls = make_assistant(cfg)
    tool = TransferTool(assistant)
    result = await tool.execute({"extension": "2001"})
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "Transferring you now."
    assert result.data == {
        "extension": "2001",
        "target": "sip:2001@pbx.example.com",
        "transferred": True,
    }
    assert len(calls) == 1
    call_info, target = calls[0]
    assert call_info is assistant.current_call
    assert target == "sip:2001@pbx.example.com"


async def test_raw_sip_uri_passthrough_when_policy_allows(config_factory):
    cfg = config_factory(outbound_allow_sip_uri="true", sip_domain="pbx.example.com")
    assistant, calls = make_assistant(cfg)
    tool = TransferTool(assistant)
    result = await tool.execute({"extension": "sip:support@other.example.com"})
    assert result.status == ToolStatus.SUCCESS
    assert calls[0][1] == "sip:support@other.example.com"
    assert result.data["target"] == "sip:support@other.example.com"


async def test_raw_sip_uri_rejected_by_default_policy(config_factory):
    # outbound_allow_sip_uri defaults off: the real policy rejects sip: URIs
    assistant, calls = make_assistant(config_factory())
    tool = TransferTool(assistant)
    result = await tool.execute({"extension": "sip:evil@attacker.example.com"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I cannot transfer to that number."
    assert calls == []


async def test_transfer_call_false_fails(config_factory):
    assistant, calls = make_assistant(config_factory(), transfer_result=False)
    tool = TransferTool(assistant)
    result = await tool.execute({"extension": "2001"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "I could not complete the transfer."
    assert len(calls) == 1


async def test_handler_without_transfer_call_fails(config_factory):
    assistant, _ = make_assistant(config_factory(), with_handler=False)
    tool = TransferTool(assistant)
    result = await tool.execute({"extension": "2001"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "Transfers are not available."


async def test_self_disables_when_config_flag_off(config_factory):
    cfg = config_factory(enable_transfer_tool="false")
    assistant, _ = make_assistant(cfg)
    tool = TransferTool(assistant)
    assert tool.enabled is False


async def test_international_target_rejected_by_voice_dial_policy(config_factory):
    """Regression (toll fraud): TRANSFER applies VOICE_DIAL_DENY_PATTERN."""
    assistant, calls = make_assistant(config_factory())
    tool = TransferTool(assistant)
    for target in ("011 44 20 7946 0958", "+44-20-7946-0958", "1-900-555-0100"):
        result = await tool.execute({"extension": target})
        assert result.status == ToolStatus.FAILED
        assert result.message == "I cannot transfer to that number."
    assert calls == []


async def test_transfer_normalizes_spoken_number(config_factory):
    cfg = config_factory(sip_domain="pbx.example.com")
    assistant, calls = make_assistant(cfg)
    result = await TransferTool(assistant).execute({"extension": "555-123-4567"})
    assert result.status == ToolStatus.SUCCESS
    assert calls[0][1] == "sip:5551234567@pbx.example.com"
