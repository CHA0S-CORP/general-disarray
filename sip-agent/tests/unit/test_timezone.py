"""Unit tests for the LOCAL_TIMEZONE plumbing and new reliability config."""
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_config_defaults(config_factory):
    cfg = config_factory()
    assert cfg.local_timezone == "America/Los_Angeles"
    assert cfg.sip_busy_reject is True
    assert cfg.grounding_retry_enabled is True
    assert cfg.grounding_retry_timeout_s == 15.0
    assert cfg.enable_drink_tool is True


def test_config_overrides(config_factory):
    cfg = config_factory(
        LOCAL_TIMEZONE="US/Eastern",
        SIP_BUSY_REJECT="false",
        GROUNDING_RETRY_ENABLED="false",
        GROUNDING_RETRY_TIMEOUT_S="7.5",
        ENABLE_DRINK_TOOL="false",
    )
    assert cfg.local_timezone == "US/Eastern"
    assert cfg.sip_busy_reject is False
    assert cfg.grounding_retry_enabled is False
    assert cfg.grounding_retry_timeout_s == 7.5
    assert cfg.enable_drink_tool is False


async def test_datetime_tool_defaults_to_config_tz(config_factory):
    from plugins.datetime_tool import DateTimeTool
    cfg = config_factory(LOCAL_TIMEZONE="UTC")
    tool = DateTimeTool(SimpleNamespace(config=cfg, session=None))

    assert tool.parameters["timezone"]["default"] == "UTC"
    result = await tool.execute({"format": "full"})
    assert result.data["timezone"] == "UTC"
    assert "UTC" in result.message


async def test_datetime_tool_bad_tz_falls_back(config_factory):
    from plugins.datetime_tool import DateTimeTool
    cfg = config_factory(LOCAL_TIMEZONE="Not/AZone")
    tool = DateTimeTool(SimpleNamespace(config=cfg, session=None))

    result = await tool.execute({"timezone": "Also/Bogus"})
    # Double fallback lands on US/Pacific rather than raising.
    assert result.data["timezone"] == "US/Pacific"


def test_scheduler_local_now_uses_config_tz(config_factory):
    """_local_now returns naive local wall-clock even on a UTC container."""
    from datetime import datetime, timezone as dt_tz
    from zoneinfo import ZoneInfo
    from tool_manager import ToolManager

    cfg = config_factory(LOCAL_TIMEZONE="America/Los_Angeles")
    assistant = SimpleNamespace(config=cfg, session=None)
    tm = ToolManager.__new__(ToolManager)  # skip tool loading
    tm.config = cfg

    local = tm._local_now()
    expected = datetime.now(ZoneInfo("America/Los_Angeles")).replace(tzinfo=None)
    assert abs((local - expected).total_seconds()) < 5
    assert local.tzinfo is None
