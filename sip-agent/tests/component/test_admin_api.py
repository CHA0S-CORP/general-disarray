"""Component tests for the admin dashboard surface.

Covers the /calls summaries, the live-call endpoints, the /admin/events SSE
stream (via a real uvicorn server — TestClient/ASGITransport buffer whole
responses, which would hang on an endless stream), auth enforcement, and the
/admin static page.
"""
import json
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

pytestmark = pytest.mark.component


# --- GET /calls ---------------------------------------------------------------

def test_calls_empty(client):
    r = client.get("/calls")
    assert r.status_code == 200
    assert r.json() == []


def test_calls_lists_recent_and_live(client, assistant):
    ts = assistant.transcripts
    ts.start("done-1", "inbound", "sip:100@host")
    ts.add_turn("done-1", "user", "hi")
    ts.add_turn("done-1", "assistant", "hello")
    ts.end("done-1")
    ts.start("live-1", "outbound", "sip:200@host")
    ts.add_turn("live-1", "user", "yo")

    r = client.get("/calls")
    assert r.status_code == 200
    by_id = {c["call_id"]: c for c in r.json()}
    assert set(by_id) == {"done-1", "live-1"}

    done = by_id["done-1"]
    assert done["live"] is False
    assert done["direction"] == "inbound"
    assert done["remote_uri"] == "sip:100@host"
    assert done["turns"] == 2
    assert done["started_at"] and done["ended_at"]

    live = by_id["live-1"]
    assert live["live"] is True
    assert live["turns"] == 1
    assert live["ended_at"] is None
    # Summaries only — turn contents stay behind the transcript endpoint.
    assert "content" not in json.dumps(r.json())


# --- /calls/active + hangup ----------------------------------------------------

def _fake_session():
    return SimpleNamespace(
        transcript_id="in-1234-1",
        direction="inbound",
        start_time=time.time() - 12.0,
        call_info=SimpleNamespace(remote_uri="sip:42@host", is_active=True),
        conversation_history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "and another thing"},
        ],
    )


def test_active_call_idle(client):
    r = client.get("/calls/active")
    assert r.status_code == 200
    assert r.json() == {"active": False, "count": 0, "calls": []}


def test_active_call_summary(client, assistant):
    assistant.session = _fake_session()
    body = client.get("/calls/active").json()
    assert body["active"] is True
    assert body["count"] == 1
    call = body["calls"][0]
    assert call["call_id"] == "in-1234-1"
    assert call["caller"] == "sip:42@host"
    assert call["direction"] == "inbound"
    assert call["turns"] == 2  # user turns only
    assert call["duration_seconds"] >= 12.0


def test_hangup_idle_is_404(client):
    assert client.post("/calls/active/hangup").status_code == 404


def test_hangup_active_call(client, assistant):
    hung = []

    async def hangup_call(call_info):
        hung.append(call_info)

    assistant.sip_handler.hangup_call = hangup_call
    assistant.session = _fake_session()

    r = client.post("/calls/active/hangup")
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert r.json()["call_id"] == "in-1234-1"
    assert hung == [assistant.session.call_info]


def test_hangup_rejects_cross_site_browser_requests(client, assistant):
    """CSRF guard: a header-free POST is a CORS-simple request any web page
    can fire, so browser-identified cross-site requests must be rejected even
    in tokenless localhost mode."""
    hung = []

    async def hangup_call(call_info):
        hung.append(call_info)

    assistant.sip_handler.hangup_call = hangup_call
    assistant.session = _fake_session()

    # Modern browser, cross-site page (fetch metadata).
    r = client.post("/calls/active/hangup",
                    headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403

    # Older browser: Origin header only, host mismatch.
    r = client.post("/calls/active/hangup",
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403

    assert hung == []  # neither request reached the SIP handler

    # Same-origin browser request (the admin page itself) still works.
    r = client.post("/calls/active/hangup",
                    headers={"Sec-Fetch-Site": "same-origin",
                             "Origin": "http://testserver"})
    assert r.status_code == 200
    assert hung == [assistant.session.call_info]

    # Non-browser clients (no Origin, no fetch metadata) pass through:
    # covered by test_hangup_active_call above.


# --- auth -----------------------------------------------------------------------

def test_admin_endpoints_require_auth_when_token_set(make_client, config_factory):
    cfg = config_factory(api_auth_token="s3cret")
    c, _ = make_client(cfg)

    assert c.get("/calls").status_code == 401
    assert c.get("/calls/active").status_code == 401
    assert c.post("/calls/active/hangup").status_code == 401
    # The 401 fires in the dependency, before the stream starts.
    assert c.get("/admin/events").status_code == 401

    ok = c.get("/calls", headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    # The page itself is a static shell with no data: served without auth.
    assert c.get("/admin").status_code == 200


# --- SSE stream -------------------------------------------------------------------

class _LiveServer:
    """A real uvicorn server in a thread: the SSE stream never ends, and both
    TestClient and httpx's ASGITransport buffer entire responses."""

    def __init__(self, app):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        cfg = uvicorn.Config(app, host="127.0.0.1", port=self.port,
                             log_level="warning", lifespan="off")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("live server failed to start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture
def live_server(assistant):
    import asyncio

    from api import create_api

    inner = create_api(assistant, call_queue=None)
    loop_holder = {}

    async def app(scope, receive, send):
        # Capture the server's event loop so the test can publish onto it
        # thread-safely (the bus is not a cross-thread primitive by design).
        loop_holder["loop"] = asyncio.get_running_loop()
        await inner(scope, receive, send)

    srv = _LiveServer(app)
    srv.loop_holder = loop_holder
    srv.start()
    yield srv
    srv.stop()


def _wait_until(predicate, timeout=5.0, interval=0.02):
    """Poll `predicate` until truthy or timeout; returns its final value."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_sse_stream_delivers_bus_events(live_server, assistant):
    with httpx.Client(base_url=live_server.url, timeout=10) as c:
        with c.stream("GET", "/admin/events") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")

            # The stream generator subscribes lazily (leak-proof on aborted
            # connects), so wait for the subscription before publishing.
            assert _wait_until(lambda: assistant.events.subscriber_count == 1)

            # Publish two events from inside the server's event loop.
            loop = live_server.loop_holder["loop"]
            loop.call_soon_threadsafe(
                assistant.events.publish, "user_turn", "c-77", {"text": "hi"})
            loop.call_soon_threadsafe(
                assistant.events.publish, "tool_call", "-",
                {"tool": "CALC", "success": True})

            items = []
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    items.append(json.loads(line.split("data:", 1)[1]))
                    if len(items) == 2:
                        break

            assert items[0]["event"] == "user_turn"
            assert items[0]["call_id"] == "c-77"
            assert items[0]["data"] == {"text": "hi"}
            assert isinstance(items[0]["ts"], float) and items[0]["ts"] > 0
            assert items[1]["event"] == "tool_call"
            assert items[1]["call_id"] == "-"
            assert items[1]["data"] == {"tool": "CALC", "success": True}

    # Regression guard for the unsubscribe-on-disconnect path: closing the
    # stream mid-flight (the server never ends it) must run the generator's
    # finally and remove the queue from the bus — otherwise every dashboard
    # reconnect would leak a 256-slot queue that receives events forever.
    assert _wait_until(lambda: assistant.events.subscriber_count == 0), \
        f"SSE queue leaked: {assistant.events.subscriber_count} subscriber(s) remain"


def test_sse_immediate_abort_leaks_no_subscriber(live_server, assistant):
    """Aborting /admin/events without ever reading the body (the dashboard's
    auto-reconnect loop does this on every network blip) must not leak bus
    queues — including when the disconnect lands before the stream generator's
    first iteration."""
    for _ in range(5):
        with httpx.Client(base_url=live_server.url, timeout=10) as c:
            with c.stream("GET", "/admin/events"):
                pass  # drop the connection immediately

    assert _wait_until(lambda: assistant.events.subscriber_count == 0), \
        f"SSE queue leaked: {assistant.events.subscriber_count} subscriber(s) remain"


async def test_tool_execution_publishes_tool_call_event(client, assistant):
    """ToolManager.execute_tool mirrors name + success (never params) onto
    the bus."""
    q = assistant.events.subscribe()  # bus attached by create_api

    result = await assistant.tool_manager.execute_tool(
        SimpleNamespace(name="CALC", params={"expression": "2+2"}))
    assert result.status.value == "success"
    item = q.get_nowait()
    assert item["event"] == "tool_call"
    assert item["call_id"] == "-"  # no live call session
    assert item["data"] == {"tool": "CALC", "success": True}
    assert "expression" not in json.dumps(item)

    await assistant.tool_manager.execute_tool(
        SimpleNamespace(name="NO_SUCH_TOOL", params={}))
    assert q.get_nowait()["data"] == {"tool": "NO_SUCH_TOOL", "success": False}


def test_sse_slow_consumer_drops_oldest_without_blocking(client, assistant):
    """A subscriber that never drains loses its OLDEST events; publish stays
    non-blocking throughout (this runs synchronously — a blocking publish
    would deadlock the test)."""
    from admin_events import DEFAULT_QUEUE_SIZE
    bus = assistant.events  # attached by create_api (client fixture)
    q = bus.subscribe()
    try:
        overflow = 44
        for i in range(DEFAULT_QUEUE_SIZE + overflow):
            bus.publish("user_turn", "c-1", {"n": i})
        assert q.qsize() == DEFAULT_QUEUE_SIZE
        # The first `overflow` events were dropped, newest kept in order.
        assert q.get_nowait()["data"]["n"] == overflow
    finally:
        bus.unsubscribe(q)


# --- /admin page ---------------------------------------------------------------

def test_admin_page_served_when_enabled(client):
    r = client.get("/admin")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "General Disarray" in r.text
    # Self-contained page: no external scripts/styles.
    assert "https://cdn" not in r.text
    assert 'src="http' not in r.text


def test_admin_page_404_when_disabled(make_client, config_factory):
    cfg = config_factory(admin_ui_enabled="false")
    c, _ = make_client(cfg)
    assert c.get("/admin").status_code == 404
