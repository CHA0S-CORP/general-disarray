"""Component tests for ToolManager: execution, the CALLBACK caller-number
special-casing, and the scheduled-task bookkeeping.
"""
from types import SimpleNamespace

import pytest

from tool_manager import ToolStatus

pytestmark = pytest.mark.component


def _call(name, **params):
    # execute_tool only needs .name and .params.
    return SimpleNamespace(name=name, params=params, raw="")


async def test_execute_calc(assistant):
    result = await assistant.tool_manager.execute_tool(_call("CALC", expression="2+2"))
    assert result.status == ToolStatus.SUCCESS
    assert "4" in result.message


async def test_unknown_tool(assistant):
    result = await assistant.tool_manager.execute_tool(_call("NOPE"))
    assert result.status == ToolStatus.FAILED
    assert "unknown tool" in result.message.lower()


def _callbacks(tm):
    return [t for t in tm.scheduled_tasks.values() if t.task_type == "callback"]


async def test_callback_defaults_to_caller_number(assistant):
    assistant.current_call = SimpleNamespace(remote_uri="sip:+15551234567@host")
    result = await assistant.tool_manager.execute_tool(_call("CALLBACK", delay=120))
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "I'll call you back in 2 minutes"
    # CallbackTool itself defaults to the caller (no manager interception).
    [task] = _callbacks(assistant.tool_manager)
    assert task.target_uri == "sip:+15551234567@host"
    delay = (task.execute_at - assistant.tool_manager._local_now()).total_seconds()
    assert 115 <= delay <= 120


async def test_callback_without_number_fails(assistant):
    assistant.current_call = None
    result = await assistant.tool_manager.execute_tool(_call("CALLBACK"))
    assert result.status == ToolStatus.FAILED
    assert _callbacks(assistant.tool_manager) == []


async def test_schedule_and_cancel_tasks(assistant):
    tm = assistant.tool_manager
    task_id = await tm.schedule_task("timer", 3600, "ping")
    pending = tm.get_pending_tasks()
    assert any(t.id == task_id for t in pending)

    cancelled = await tm.cancel_tasks("all")
    assert cancelled >= 1
    assert tm.get_pending_tasks() == []


async def test_scheduled_calls_persist_across_restart(assistant, comp_config):
    from datetime import timedelta
    from tool_manager import ToolManager

    tm = assistant.tool_manager
    kept = await tm.schedule_task("scheduled_call", 3600, "future call",
                                  target_uri="1001", metadata={"extension": "1001"})
    # Timers are call-bound and must NOT survive a restart.
    await tm.schedule_task("timer", 3600, "call-bound timer")
    assert (comp_config.data_dir / "scheduled_tasks.json").exists()

    # Simulate a stale one-shot missed by more than the grace period
    # (execute_at lives on the scheduler's LOCAL_TIMEZONE clock).
    stale = await tm.schedule_task("scheduled_call", 3600, "stale call",
                                   target_uri="1002", metadata={"extension": "1002"})
    tm.scheduled_tasks[stale].execute_at = tm._local_now() - timedelta(hours=2)
    tm._persist_tasks()

    # "Restart": a fresh manager reloads from the same data dir.
    tm2 = ToolManager(assistant)
    tm2._load_persisted_tasks()
    assert kept in tm2.scheduled_tasks           # future one-shot restored
    assert stale not in tm2.scheduled_tasks      # too-late one-shot dropped
    assert all(t.task_type != "timer" for t in tm2.scheduled_tasks.values())


async def test_legacy_persisted_tasks_migrate_to_local_clock(assistant, comp_config):
    """Entries without the 'clock' marker (written pre-LOCAL_TIMEZONE by the
    container clock) are converted on load and re-persisted with the marker."""
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from tool_manager import ToolManager

    # Far enough out that no container-vs-local offset makes it look missed.
    legacy_at = datetime.now() + timedelta(hours=48)
    store = comp_config.data_dir / "scheduled_tasks.json"
    store.write_text(json.dumps([{
        "id": "legacy-1", "task_type": "scheduled_call",
        "execute_at": legacy_at.isoformat(), "message": "legacy",
        "target_uri": "1001", "metadata": {"extension": "1001"},
        "completed": False,
    }]))

    tm = ToolManager(assistant)
    tm._load_persisted_tasks()

    task = tm.scheduled_tasks["legacy-1"]
    expected = (legacy_at
                .replace(tzinfo=datetime.now().astimezone().tzinfo)
                .astimezone(ZoneInfo(assistant.config.local_timezone))
                .replace(tzinfo=None))
    assert abs((task.execute_at - expected).total_seconds()) < 1
    # Re-persisted with the marker so the migration only ever runs once.
    entries = json.loads(store.read_text())
    assert entries and all(e.get("clock") == "local" for e in entries)


async def test_plugin_autodiscovery_from_data_dir(assistant, comp_config):
    """A tool file dropped into data/plugins is discovered and registered."""
    from textwrap import dedent
    from tool_manager import ToolManager

    plugin_dir = comp_config.data_dir / "plugins"
    plugin_dir.mkdir(exist_ok=True)
    (plugin_dir / "ping_tool.py").write_text(dedent("""
        from tool_plugins import BaseTool, ToolResult, ToolStatus

        class PingTool(BaseTool):
            name = "PING_TEST"
            description = "test-only ping tool"
            parameters = {}

            async def execute(self, params):
                return ToolResult(status=ToolStatus.SUCCESS, message="pong")
    """))

    tm = ToolManager(assistant)
    assert tm.has_tool("PING_TEST")
    # Builtins are still explicitly registered, not shadowed by discovery.
    assert tm.has_tool("CALC")

    result = await tm.get_tool("PING_TEST").execute({})
    assert result.message == "pong"


async def test_cancel_task_removes_from_persistence(assistant, comp_config):
    import json as _json
    tm = assistant.tool_manager
    task_id = await tm.schedule_task("callback", 3600, "call me",
                                     target_uri="sip:1001@host")
    assert tm.cancel_task(task_id) is True
    assert tm.cancel_task(task_id) is False
    persisted = _json.loads((comp_config.data_dir / "scheduled_tasks.json").read_text())
    assert all(entry["id"] != task_id for entry in persisted)


async def test_verify_required_tool_is_gated(make_client, config_factory, tmp_path):
    """A tool in VERIFY_REQUIRED_TOOLS refuses until session.verified is True."""
    cfg = config_factory(data_dir=str(tmp_path), verify_required_tools="CALC")
    _, a = make_client(cfg)
    a.session = SimpleNamespace(verified=False)

    blocked = await a.tool_manager.execute_tool(_call("CALC", expression="2+2"))
    assert blocked.status == ToolStatus.FAILED
    assert "verify" in blocked.message.lower()

    a.session.verified = True
    ok = await a.tool_manager.execute_tool(_call("CALC", expression="2+2"))
    assert ok.status == ToolStatus.SUCCESS
    assert "4" in ok.message


# --- CALLBACK toll-fraud policy ----------------------------------------------

def _caller(uri="sip:+15551234567@host", call_id="call-A"):
    return SimpleNamespace(remote_uri=uri, call_id=call_id, is_active=True)


@pytest.mark.parametrize("destination", [
    "+44 20 7946 0958", "011442079460958", "1-900-555-0100",
    "sip:2001@evil.example",
])
async def test_callback_to_blocked_destination_is_refused(assistant, destination):
    """Regression (toll fraud): a caller could have the agent dial
    international / premium-rate numbers via CALLBACK destination=."""
    assistant.current_call = _caller()
    result = await assistant.tool_manager.execute_tool(
        _call("CALLBACK", delay=60, destination=destination))
    assert result.status == ToolStatus.FAILED
    assert result.message == "I can only call you back at your own number."
    assert _callbacks(assistant.tool_manager) == []


async def test_callback_rest_path_enforces_policy(assistant):
    """The REST /tools/CALLBACK/execute + /webhook/call paths call the wrapper's
    execute() directly — the policy must hold there too."""
    tool = assistant.tool_manager.get_tool("CALLBACK")
    result = await tool.execute({"delay": 60, "destination": "+442079460958"})
    assert result.status == ToolStatus.FAILED
    assert _callbacks(assistant.tool_manager) == []


async def test_callback_to_callers_own_number_always_allowed(assistant, config_factory):
    # Even with a restrictive allow pattern, the caller's own number is fine;
    # the caller's own URI is dialed, not the spoken digits.
    assistant.config.voice_dial_allow_pattern = r"2\d{3}"
    assistant.current_call = _caller()
    result = await assistant.tool_manager.execute_tool(
        _call("CALLBACK", delay=60, destination="(555) 123-4567"))
    assert result.status == ToolStatus.SUCCESS
    [task] = _callbacks(assistant.tool_manager)
    assert task.target_uri == "sip:+15551234567@host"


async def test_callback_to_allowed_other_number(assistant):
    assistant.current_call = _caller()
    result = await assistant.tool_manager.execute_tool(
        _call("CALLBACK", delay=60, destination="2001"))
    assert result.status == ToolStatus.SUCCESS
    assert result.message == "I'll call 2001 in 1 minute"
    [task] = _callbacks(assistant.tool_manager)
    assert task.target_uri == "2001"


@pytest.mark.parametrize("delay", [-300, "-1", 86401, "soon"])
async def test_callback_delay_out_of_range_refused(assistant, delay):
    """Regression: negative delays were accepted ('in -300 seconds')."""
    assistant.current_call = _caller()
    result = await assistant.tool_manager.execute_tool(_call("CALLBACK", delay=delay))
    assert result.status == ToolStatus.FAILED
    assert "-" not in result.message
    assert _callbacks(assistant.tool_manager) == []


async def test_callback_per_call_cap(assistant):
    assistant.current_call = _caller()
    limit = assistant.config.callback_max_per_call
    for _ in range(limit):
        ok = await assistant.tool_manager.execute_tool(_call("CALLBACK", delay=60))
        assert ok.status == ToolStatus.SUCCESS
    over = await assistant.tool_manager.execute_tool(_call("CALLBACK", delay=60))
    assert over.status == ToolStatus.FAILED
    assert len(_callbacks(assistant.tool_manager)) == limit
    # A different call is not affected by the first call's count.
    assistant.current_call = _caller(uri="sip:2002@host", call_id="call-B")
    ok = await assistant.tool_manager.execute_tool(_call("CALLBACK", delay=60))
    assert ok.status == ToolStatus.SUCCESS


async def test_callback_per_call_cap_uses_session_tool_state(assistant):
    call = _caller()
    assistant.current_call = call
    assistant.session = SimpleNamespace(call_info=call, transcript_id="t-1",
                                        caller_id="", tool_state={}, verified=False)
    limit = assistant.config.callback_max_per_call
    for _ in range(limit):
        await assistant.tool_manager.execute_tool(_call("CALLBACK", delay=60))
    assert assistant.session.tool_state["callbacks_scheduled"] == limit
    over = await assistant.tool_manager.execute_tool(_call("CALLBACK", delay=60))
    assert over.status == ToolStatus.FAILED


# --- CANCEL / STATUS scope ----------------------------------------------------

async def test_voice_cancel_and_status_only_see_own_tasks(assistant):
    """Regression: any caller could hear and cancel every pending task —
    including other callers' callbacks and REST-scheduled calls."""
    tm = assistant.tool_manager
    rest_id = await tm.schedule_task("scheduled_call", 3600, "briefing",
                                     target_uri="1001", metadata={"extension": "1001"})

    assistant.current_call = _caller(uri="sip:1001@host", call_id="call-A")
    await tm.execute_tool(_call("SET_TIMER", duration=600))
    await tm.execute_tool(_call("CALLBACK", delay=600))
    a_ids = {t.id for t in tm.scheduled_tasks.values() if t.owner_caller == "1001"}
    assert len(a_ids) == 2

    # Caller B sees and cancels nothing of A's (nor the REST schedule).
    assistant.current_call = _caller(uri="sip:2002@host", call_id="call-B")
    status = await tm.execute_tool(_call("STATUS"))
    assert status.message == "You have no pending timers or callbacks"
    cancel = await tm.execute_tool(_call("CANCEL", task_type="all"))
    assert cancel.message == "No tasks to cancel"
    assert a_ids <= set(tm.scheduled_tasks)

    # Caller A (even on a later call) sees/cancels only their own two tasks;
    # the REST scheduled_call to their extension is untouched.
    assistant.current_call = _caller(uri="sip:1001@host", call_id="call-A2")
    status = await tm.execute_tool(_call("STATUS"))
    assert status.data["pending_count"] == 2
    cancel = await tm.execute_tool(_call("CANCEL", task_type="all"))
    assert cancel.data["cancelled_count"] == 2
    assert rest_id in tm.scheduled_tasks
    assert not (a_ids & set(tm.scheduled_tasks))

    # REST cancel-by-id still works.
    assert tm.cancel_task(rest_id) is True


async def test_cancel_with_no_live_call_is_operator_scope(assistant):
    """No live call means a REST/operator invocation (e.g. cleaning up after a
    call, as the e2e suite does): every timer/callback is visible, REST
    /schedule calls never are. Regression: owner scoping made REST CANCEL a
    no-op once the call that set the timer had ended."""
    tm = assistant.tool_manager
    rest_id = await tm.schedule_task("scheduled_call", 3600, "briefing",
                                     target_uri="1001", metadata={"extension": "1001"})
    assistant.current_call = _caller(uri="sip:1001@host", call_id="call-A")
    await tm.execute_tool(_call("SET_TIMER", duration=600))
    await tm.schedule_task("callback", 3600, "x", target_uri="2001")  # unowned

    assistant.current_call = None
    status = await tm.execute_tool(_call("STATUS"))
    assert status.data["pending_count"] == 2
    result = await tm.execute_tool(_call("CANCEL", task_type="all"))
    assert result.data["cancelled_count"] == 2
    assert list(tm.scheduled_tasks) == [rest_id]


async def test_owner_fields_persist_and_legacy_records_load(assistant, comp_config):
    import json
    from tool_manager import ToolManager

    tm = assistant.tool_manager
    assistant.current_call = _caller(uri="sip:1001@host", call_id="call-A")
    await tm.execute_tool(_call("CALLBACK", delay=3600))
    store = comp_config.data_dir / "scheduled_tasks.json"
    entries = json.loads(store.read_text())
    assert entries[0]["owner_caller"] == "1001"
    assert entries[0]["owner_call_id"] == "call-A"
    # A legacy record without owner keys (plus an unknown key) still loads.
    legacy = dict(entries[0], id="legacy", future_field=1)
    legacy.pop("owner_caller")
    legacy.pop("owner_call_id")
    store.write_text(json.dumps(entries + [legacy]))
    tm2 = ToolManager(assistant)
    tm2._load_persisted_tasks()
    assert tm2.scheduled_tasks[entries[0]["id"]].owner_caller == "1001"
    assert tm2.scheduled_tasks["legacy"].owner_caller is None


# --- recurring scheduled calls -----------------------------------------------

def _local(dt_aware, config):
    from zoneinfo import ZoneInfo
    return dt_aware.astimezone(ZoneInfo(config.local_timezone)).replace(tzinfo=None)


async def _recurring_task(tm, config, at, pattern="daily",
                          tz="America/New_York", at_time="07:00"):
    from zoneinfo import ZoneInfo
    task_id = await tm.schedule_task(
        "scheduled_call", 3600, "wake up", target_uri="1001",
        metadata={"extension": "1001", "recurring": pattern,
                  "timezone": tz, "at_time": at_time})
    task = tm.scheduled_tasks[task_id]
    task.execute_at = _local(at.replace(tzinfo=ZoneInfo(tz)), config)
    return task


async def test_recurring_next_run_keeps_wall_clock_across_dst(assistant):
    """Regression: next run was now+1 day — drifting by the call's duration
    and shifting an hour across DST (naive arithmetic)."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    tm, cfg = assistant.tool_manager, assistant.config
    ny = ZoneInfo("America/New_York")
    # Sat 2026-10-31 07:00 EDT; DST ends Sun 2026-11-01.
    task = await _recurring_task(tm, cfg, datetime(2026, 10, 31, 7, 0))
    # Evaluated 25 minutes after the occurrence (call duration + retries).
    nxt = tm._next_occurrence(task, task.execute_at + timedelta(minutes=25))
    assert nxt == _local(datetime(2026, 11, 1, 7, 0, tzinfo=ny), cfg)


async def test_recurring_weekdays_skip_weekend(assistant):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    tm, cfg = assistant.tool_manager, assistant.config
    task = await _recurring_task(tm, cfg, datetime(2026, 10, 9, 7, 0),  # Friday
                                 pattern="weekdays")
    nxt = tm._next_occurrence(task, task.execute_at + timedelta(minutes=5))
    assert nxt == _local(datetime(2026, 10, 12, 7, 0,
                                  tzinfo=ZoneInfo("America/New_York")), cfg)


async def test_recurring_call_rescheduled_in_place_even_on_failure(assistant):
    """Regression: a failed occurrence ended the series, and success created
    a NEW id so DELETE /schedule/{id} stopped working."""
    from datetime import timedelta
    tm = assistant.tool_manager
    assistant.config.callback_retry_attempts = 1

    async def failing_dial(uri, message):
        raise RuntimeError("no answer")

    assistant.make_outbound_call = failing_dial
    task_id = await tm.schedule_task(
        "scheduled_call", 0, "wake up", target_uri="1001",
        metadata={"extension": "1001", "recurring": "daily",
                  "timezone": "America/New_York"})
    task = tm.scheduled_tasks[task_id]
    task.execute_at = tm._local_now() - timedelta(seconds=1)
    task.completed = True  # as the scheduler marks it at dispatch
    await tm._execute_scheduled_task(task)

    assert set(tm.scheduled_tasks) == {task_id}       # same id, no new task
    again = tm.scheduled_tasks[task_id]
    assert again.completed is False
    assert again.execute_at > tm._local_now()
    assert tm.cancel_task(task_id) is True             # DELETE still works


async def test_recurring_call_cancelled_mid_call_is_not_resurrected(assistant):
    tm = assistant.tool_manager
    assistant.config.callback_retry_attempts = 1
    task_id = await tm.schedule_task(
        "scheduled_call", 0, "x", target_uri="1001",
        metadata={"extension": "1001", "recurring": "daily"})
    task = tm.scheduled_tasks[task_id]
    task.completed = True

    async def dial_then_cancelled(uri, message):
        tm.cancel_task(task_id)          # DELETE /schedule/{id} during the call

    assistant.make_outbound_call = dial_then_cancelled
    await tm._execute_scheduled_task(task)
    assert task_id not in tm.scheduled_tasks


async def test_scheduled_call_webhook_timestamp_is_utc_aware(assistant, monkeypatch):
    from datetime import datetime
    import tool_manager as tm_module
    sent = []

    async def fake_deliver(url, payload, config, api_name=None):
        sent.append(payload)
        return True

    monkeypatch.setattr(tm_module, "deliver_webhook", fake_deliver)
    task = SimpleNamespace(id="abc")
    await assistant.tool_manager._send_scheduled_call_webhook(
        task, {"callback_url": "https://example.com/hook"}, "completed")
    stamp = datetime.fromisoformat(sent[0]["timestamp"])
    assert stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0


# --- wrapper / execution plumbing --------------------------------------------

class _SpokenTool:
    name = "SPOKEN_TEST"
    description = "test"
    parameters = {}
    speak_result = True

    def __init__(self, exc=None):
        self.exc = exc

    async def execute(self, params):
        from tool_plugins import ToolResult as PluginResult, ToolStatus as PluginStatus
        if self.exc:
            raise self.exc
        return PluginResult(status=PluginStatus.SUCCESS,
                            message="raw [1] http://x snippets",
                            spoken_message="Here is the short answer.")


async def test_wrapper_preserves_spoken_message(assistant):
    """Regression: PluginToolWrapper dropped spoken_message, so to_speech()
    never existed and WEB_SEARCH read raw scraped results aloud."""
    tm = assistant.tool_manager
    assert tm.register_tool_instance(_SpokenTool())
    result = await tm.execute_tool(_call("SPOKEN_TEST"))
    assert result.message == "raw [1] http://x snippets"
    assert result.to_speech() == "Here is the short answer."


async def test_tool_exception_is_not_spoken_verbatim(assistant):
    tool = _SpokenTool(exc=RuntimeError("connect to 10.0.0.5:5432 failed"))
    tool.name = "BOOM_TEST"
    assistant.tool_manager.register_tool_instance(tool)
    result = await assistant.tool_manager.execute_tool(_call("BOOM_TEST"))
    assert result.status == ToolStatus.FAILED
    assert "10.0.0.5" not in result.message


async def test_unknown_tool_metric_label_is_bounded(assistant, monkeypatch):
    import tool_manager as tm_module
    labels = []
    monkeypatch.setattr(tm_module.Metrics, "record_tool_call",
                        classmethod(lambda cls, name: labels.append(name)))
    monkeypatch.setattr(tm_module.Metrics, "record_tool_error",
                        classmethod(lambda cls, name, kind: labels.append(name)))
    await assistant.tool_manager.execute_tool(_call("HALLUCINATED_TOOL_123"))
    await assistant.tool_manager.execute_tool(_call("CALC", expression="1+1"))
    assert "HALLUCINATED_TOOL_123" not in labels
    assert "unknown" in labels and "CALC" in labels


async def test_reload_plugins_swaps_only_on_success(assistant, monkeypatch):
    """Regression: reload_plugins called a nonexistent _load_plugins after
    clearing every tool."""
    tm = assistant.tool_manager
    assert tm.register_tool_instance(_SpokenTool())   # external (MCP-like)
    before = set(tm.tools)
    count = tm.reload_plugins()
    assert count == len(tm.tools) and set(tm.tools) == before

    def boom():
        raise RuntimeError("bad plugin")

    monkeypatch.setattr(tm, "_load_tools", boom)
    assert tm.reload_plugins() == len(before)
    assert set(tm.tools) == before
