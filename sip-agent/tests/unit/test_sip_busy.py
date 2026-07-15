"""Unit tests for the busy-reject decision (SIPHandler._busy)."""
import pytest

from sip_handler import SIPHandler

pytestmark = pytest.mark.unit


def make_handler(config_factory, **overrides):
    cfg = config_factory(**overrides)
    return SIPHandler(cfg, on_call_callback=lambda *_: None)


def test_busy_when_call_active(config_factory):
    handler = make_handler(config_factory, SIP_BUSY_REJECT="true")
    assert handler._busy() is False
    handler.active_calls["some-call"] = object()
    assert handler._busy() is True


def test_not_busy_when_disabled(config_factory):
    handler = make_handler(config_factory, SIP_BUSY_REJECT="false")
    handler.active_calls["some-call"] = object()
    assert handler._busy() is False


def test_outbound_call_also_counts(config_factory):
    # _do_make_call populates the same dict, so an inbound INVITE during our
    # own outbound call is rejected too.
    handler = make_handler(config_factory)
    handler.active_calls["outbound-1"] = object()
    assert handler._busy() is True


# --- Capacity gate (MAX_CONCURRENT_CALLS) -----------------------------------

class _DeadCall:
    """A pj.Call whose isActive() blows up: treated as dead and pruned."""

    def isActive(self):
        raise RuntimeError("call object destroyed")


def test_capacity_two_admits_a_second_call(config_factory):
    handler = make_handler(config_factory, SIP_BUSY_REJECT="true",
                           MAX_CONCURRENT_CALLS="2")
    handler.active_calls["c1"] = object()
    assert handler._busy() is False   # one live call, capacity two
    handler.active_calls["c2"] = object()
    assert handler._busy() is True    # at capacity: third INVITE is busy


def test_capacity_default_is_one(config_factory):
    # Shipping default: exactly today's one-live-call behavior.
    handler = make_handler(config_factory, SIP_BUSY_REJECT="true")
    handler.active_calls["c1"] = object()
    assert handler._busy() is True


def test_capacity_ignores_and_prunes_dead_calls(config_factory):
    handler = make_handler(config_factory, SIP_BUSY_REJECT="true",
                           MAX_CONCURRENT_CALLS="2")
    handler.active_calls["alive"] = object()
    handler.active_calls["dead"] = _DeadCall()
    assert handler._busy() is False   # only one truly-live call
    assert "dead" not in handler.active_calls  # pruned


def test_capacity_disabled_gate_never_busy(config_factory):
    handler = make_handler(config_factory, SIP_BUSY_REJECT="false",
                           MAX_CONCURRENT_CALLS="2")
    handler.active_calls["c1"] = object()
    handler.active_calls["c2"] = object()
    handler.active_calls["c3"] = object()
    assert handler._busy() is False
