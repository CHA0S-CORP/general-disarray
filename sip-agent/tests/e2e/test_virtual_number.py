"""E2E: virtual numbers — create an ephemeral extension, call it, assert the
per-number greeting/context, the completion webhook, and single-use cleanup.

Dials sip:<number>@sip-agent:5060 directly (PJSUA2 accepts INVITEs to
arbitrary user parts), so no PBX routing is needed on the compose network.

Requires VIRTUAL_NUMBERS_ENABLED=true in the sip-agent environment; the tests
skip otherwise. Completion is asserted via the structured log events
(virtual_number_matched / virtual_number_completed) and the single-use 404 —
webhook delivery itself is covered by the deliver_webhook component tests.
"""
import os
import subprocess
import time
import uuid

import httpx
import pytest

pytestmark = pytest.mark.e2e

AGENT_API = os.environ.get("E2E_AGENT_API", "http://localhost:8080")

GREETING = "Thanks for calling about your pizza order"
PURPOSE = ("The caller is confirming pizza order number four two one one "
           "for pickup. Confirm the order and tell them it will be ready "
           "in twenty minutes.")


_API_AUTH_TOKEN = os.environ.get("E2E_API_AUTH_TOKEN", "")


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    # Send the bearer token when the deployment has API_AUTH_TOKEN enabled.
    if _API_AUTH_TOKEN:
        headers = {"Authorization": f"Bearer {_API_AUTH_TOKEN}", **kwargs.pop("headers", {})}
        kwargs["headers"] = headers
    return httpx.request(method, f"{AGENT_API}{path}", timeout=15.0, **kwargs)


@pytest.fixture
def feature_enabled(stack):
    r = _api("GET", "/virtual-numbers")
    if r.status_code == 403:
        pytest.skip("VIRTUAL_NUMBERS_ENABLED is not set on this stack")
    assert r.status_code == 200
    return True


def test_virtual_number_call_lifecycle(feature_enabled, softphone_image,
                                       question_wav, assert_spoke, transcribe,
                                       agent_events, event_names):
    import gen_audio
    gen_audio.FIXTURES.setdefault(
        "vn_confirm_order.wav", "Yes, I'm calling to confirm my pizza order.")
    fn = question_wav("vn_confirm_order.wav")

    # Create an ephemeral extension with a distinctive greeting.
    r = _api("POST", "/virtual-numbers", json={
        "purpose": PURPOSE,
        "greeting": GREETING + ". Is that right?",
        "ttl_s": 300,
        "include_transcript": True,
    })
    assert r.status_code == 200, r.text
    vn = r.json()
    number = vn["number"]

    # Dial the virtual extension directly (not the agent's main identity).
    from conftest import _AUDIO_DIR, NETWORK  # same-harness internals
    from datetime import datetime, timezone
    captured = _AUDIO_DIR / "vn_captured.wav"
    if captured.exists():
        captured.unlink()
    cname = f"e2e-vn-{uuid.uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc)
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "-i", "--name", cname,
         "--network", NETWORK, "-v", f"{_AUDIO_DIR}:/audio",
         softphone_image,
         "--rtp-port", "4000",
         "--auto-play", "--play-file", "/audio/vn_confirm_order.wav",
         "--auto-rec", "--rec-file", "/audio/vn_captured.wav",
         "--duration", "30", "--stdout-no-buf",
         f"sip:{number}@sip-agent:5060"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(34)
    try:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)

    # Layer 0: the agent spoke.
    assert_spoke(captured)

    # Layer 1: the call matched the virtual number and completed it.
    events = agent_events(started_at)
    names = event_names(events)
    assert "virtual_number_matched" in names, f"no match event; saw {sorted(set(names))}"
    assert "virtual_number_completed" in names

    # Layer 2: the custom greeting reached the caller.
    transcript = transcribe(captured).lower()
    reply_text = " ".join(
        (e.get("data") or {}).get("text", "") for e in events
        if e.get("event") == "assistant_response").lower()
    assert ("pizza order" in transcript) or ("pizza order" in reply_text), (
        f"virtual-number context never surfaced.\n  transcript={transcript!r}"
        f"\n  reply_text={reply_text!r}")

    # Single-use: the number is gone after the call.
    assert _api("GET", f"/virtual-numbers/{vn['id']}").status_code == 404


def test_virtual_number_ttl_expiry(feature_enabled):
    """An unused number disappears after its TTL (sweeper runs every second)."""
    r = _api("POST", "/virtual-numbers", json={"purpose": "expiry probe", "ttl_s": 3})
    assert r.status_code == 200
    vn_id = r.json()["id"]
    assert _api("GET", f"/virtual-numbers/{vn_id}").status_code == 200
    time.sleep(6)
    assert _api("GET", f"/virtual-numbers/{vn_id}").status_code == 404


def test_trigger_number_survives_call_and_fires_speech(
        feature_enabled, softphone_image, question_wav, assert_spoke,
        agent_events, event_names):
    """A persistent trigger number answers a call, emits the answered +
    first_speech webhooks (asserted via the structured webhook log events),
    and is still registered afterwards — until DELETE."""
    import gen_audio
    gen_audio.FIXTURES.setdefault(
        "vn_confirm_order.wav", "Yes, I'm calling to confirm my pizza order.")
    question_wav("vn_confirm_order.wav")

    r = _api("POST", "/virtual-numbers", json={
        "purpose": "Calls to this number start a workflow.",
        "persistent": True,
        "events": ["answered", "first_speech"],
        # Unroutable but syntactically fine; delivery failure is logged, not fatal.
        "callback_url": "http://sip-agent:8080/health",
    })
    if r.status_code == 400 and "private" in r.text:
        pytest.skip("WEBHOOK_ALLOW_PRIVATE is not set on this stack")
    assert r.status_code == 200, r.text
    vn = r.json()
    assert vn["persistent"] is True and vn["expires_at"] == 0
    number = vn["number"]

    from conftest import _AUDIO_DIR, NETWORK
    from datetime import datetime, timezone
    cname = f"e2e-tn-{uuid.uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc)
    try:
        proc = subprocess.Popen(
            ["docker", "run", "--rm", "-i", "--name", cname,
             "--network", NETWORK, "-v", f"{_AUDIO_DIR}:/audio",
             softphone_image,
             "--rtp-port", "4000",
             "--auto-play", "--play-file", "/audio/vn_confirm_order.wav",
             "--duration", "25", "--stdout-no-buf",
             f"sip:{number}@sip-agent:5060"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(29)
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)

        events = agent_events(started_at)
        names = event_names(events)
        assert "virtual_number_matched" in names, f"saw {sorted(set(names))}"
        assert "virtual_number_completed" in names
        # fire_webhook logs a virtual_number_webhook event per delivery
        # (status = answered / first_speech / ...); both subscribed events
        # must have fired, and the unsubscribed "completed" must not.
        fired = {(e.get("data") or {}).get("status")
                 for e in events if e.get("event") == "virtual_number_webhook"}
        assert "answered" in fired, f"answered webhook never fired; fired={fired}"
        assert "first_speech" in fired, f"first_speech webhook never fired; fired={fired}"
        assert "completed" not in fired

        # Persistent: still registered after the call, un-claimed.
        r = _api("GET", f"/virtual-numbers/{vn['id']}")
        assert r.status_code == 200 and r.json()["status"] == "active"
    finally:
        _api("DELETE", f"/virtual-numbers/{vn['id']}")
    assert _api("GET", f"/virtual-numbers/{vn['id']}").status_code == 404
