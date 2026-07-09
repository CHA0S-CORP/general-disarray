"""Unit tests for call_events: payload building and CALL_EVENTS parsing."""
import time
from types import SimpleNamespace

import pytest

import call_events

pytestmark = pytest.mark.unit


def _session(direction="inbound", transcript_id="in-1751970000-1"):
    return SimpleNamespace(
        call_info=SimpleNamespace(call_id="4", remote_uri="sip:1001@pbx"),
        direction=direction,
        transcript_id=transcript_id,
        start_time=time.time() - 12.0,
    )


def _record(**overrides):
    record = {
        "call_id": "in-1751970000-1",
        "direction": "inbound",
        "remote_uri": "sip:1001@pbx",
        "started_at": "2026-07-08T17:00:00+00:00",
        "ended_at": "2026-07-08T17:00:12+00:00",
        "turns": [{"role": "user", "content": "hello", "ts": "2026-07-08T17:00:05+00:00"}],
    }
    record.update(overrides)
    return record


def test_started_payload_shape():
    payload = call_events.build_call_event_payload(
        "call.started", _session(), _record(ended_at=None, turns=[]), True)
    assert payload["event"] == "call.started"
    assert payload["call_id"] == "in-1751970000-1"
    assert payload["sip_call_id"] == "4"
    assert payload["direction"] == "inbound"
    assert payload["remote_uri"] == "sip:1001@pbx"
    assert payload["started_at"] == "2026-07-08T17:00:00+00:00"
    assert "timestamp" in payload
    assert "duration_seconds" not in payload
    assert "transcript" not in payload


def test_ended_payload_includes_duration_and_transcript():
    record = _record()
    payload = call_events.build_call_event_payload(
        "call.ended", _session(), record, True)
    assert payload["event"] == "call.ended"
    assert payload["duration_seconds"] == pytest.approx(12.0, abs=1.0)
    assert payload["transcript"] is record
    assert payload["transcript"]["turns"][0]["content"] == "hello"


def test_ended_payload_can_omit_transcript():
    payload = call_events.build_call_event_payload(
        "call.ended", _session(), _record(), False)
    assert "transcript" not in payload
    assert "duration_seconds" in payload


def test_payload_survives_missing_transcript_record():
    payload = call_events.build_call_event_payload(
        "call.started", _session(), None, True)
    # Falls back to session fields.
    assert payload["remote_uri"] == "sip:1001@pbx"
    assert payload["started_at"]  # ISO string derived from session.start_time


def test_enabled_events_default():
    cfg = SimpleNamespace(call_events="call.started,call.ended")
    assert call_events.enabled_events(cfg) == {"call.started", "call.ended"}


def test_enabled_events_subset_and_whitespace():
    cfg = SimpleNamespace(call_events=" call.ended , ")
    assert call_events.enabled_events(cfg) == {"call.ended"}


def test_enabled_events_ignores_unknown_names():
    cfg = SimpleNamespace(call_events="call.started,call.exploded")
    assert call_events.enabled_events(cfg) == {"call.started"}
