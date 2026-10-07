"""Regression tests for sip_handler fixes: recorder survives re-INVITE media
updates, DISCONNECTED status capture, registration truthfulness, command
deadlines, PJSIP init-failure reporting, and temp-WAV leaks."""
import os
import threading
import time
from types import SimpleNamespace

import pytest

import sip_handler
from logging_utils import WAV_HEADER_SIZE
from sip_handler import CallInfo, PlaylistPlayer, SIPAccount, SIPCall, SIPHandler

pytestmark = pytest.mark.unit


class _Recorder:
    created = []

    def __init__(self):
        _Recorder.created.append(self)
        self.path = None

    def createRecorder(self, path):
        self.path = path


class _AudMed:
    def __init__(self, fail_for=None):
        self.transmits = []
        self.fail_for = fail_for

    def startTransmit(self, sink):
        if sink is self.fail_for:
            raise RuntimeError("PJ_EINVAL")
        self.transmits.append(sink)


@pytest.fixture
def fake_pj(monkeypatch):
    _Recorder.created = []
    fake = SimpleNamespace(
        AudioMediaRecorder=_Recorder,
        PJMEDIA_TYPE_AUDIO=1, PJSUA_CALL_MEDIA_ACTIVE=1,
        PJSIP_INV_STATE_CONFIRMED=5, PJSIP_INV_STATE_DISCONNECTED=6,
    )
    monkeypatch.setattr(sip_handler, "pj", fake, raising=False)
    return fake


def _media_call(aud_med):
    call = SIPCall(None, 0, SimpleNamespace())
    call.call_info = CallInfo(call_id="c1", remote_uri="sip:a@b",
                              is_active=True, start_time=time.time())
    info = SimpleNamespace(media=[SimpleNamespace(type=1, status=1, index=0)])
    call.getInfo = lambda: info
    call.getAudioMedia = lambda idx: aud_med
    return call


def test_reinvite_reattaches_recorder_and_keeps_file(fake_pj):
    med = _AudMed()
    call = _media_call(med)
    call.onCallMediaState(None)
    path = call.call_info.record_file
    try:
        assert path and os.path.exists(path)
        assert call.call_info.record_file_pos == WAV_HEADER_SIZE
        call.call_info.record_file_pos = 50_000  # reader is deep into the call

        call.onCallMediaState(None)  # re-INVITE / session refresh
        assert len(_Recorder.created) == 1
        assert call.call_info.record_file == path
        assert call.call_info.record_file_pos == 50_000
        assert med.transmits == [call.recorder, call.recorder]
    finally:
        call._cleanup_media()
    assert not os.path.exists(path)


def test_failed_reattach_replaces_file_and_resets_offset(fake_pj):
    call = _media_call(_AudMed())
    call.onCallMediaState(None)
    old_path = call.call_info.record_file
    old_recorder = call.recorder
    call.call_info.record_file_pos = 50_000

    call.getAudioMedia = lambda idx: _AudMed(fail_for=old_recorder)
    call.onCallMediaState(None)
    try:
        assert call.recorder is not old_recorder
        assert call.call_info.record_file != old_path
        assert call.call_info.record_file_pos == WAV_HEADER_SIZE
        assert not os.path.exists(old_path)  # old WAV not leaked
    finally:
        call._cleanup_media()


def test_disconnected_records_ended_and_status(fake_pj):
    ended = []
    call = SIPCall(None, 0, SimpleNamespace(_on_call_ended=ended.append))
    call.call_info = CallInfo(call_id="c1", remote_uri="sip:a@b",
                              is_active=True, start_time=time.time())
    info = call.call_info
    call.getInfo = lambda: SimpleNamespace(
        state=6, stateText="DISCONNECTED", callIdString="x", lastStatusCode=486)
    call.onCallState(None)
    assert info.ended is True
    assert info.last_status_code == 486
    assert info.is_active is False
    assert ended == [call]


# --- registration -------------------------------------------------------------

def _account(handler, **ai):
    acc = SIPAccount(handler)
    info = SimpleNamespace(regStatusText="x", uri="sip:me@x", **ai)
    acc.getInfo = lambda: info
    return acc


@pytest.mark.parametrize("ai,expected", [
    ({"regStatus": 200, "regIsActive": True}, True),
    ({"regStatus": 408, "regIsActive": False}, False),
    ({"regStatus": 403, "regIsActive": False}, False),
    ({"regStatus": 200, "regIsActive": False}, False),  # expired/unregistered
])
def test_registration_flag_tracks_state(config, ai, expected):
    handler = SIPHandler(config, on_call_callback=None)
    handler._registered.set()  # previously registered
    _account(handler, **ai).onRegState(None)
    assert handler._registered.is_set() is expected


# --- command deadlines ---------------------------------------------------------

def _cmd_handler(config, timeout=0.05):
    h = SIPHandler(config, on_call_callback=None)
    h._running = True
    h._cmd_timeout_s = timeout
    executed = []
    h._execute_command = lambda cmd, args, kwargs: executed.append(cmd) or "ok"
    return h, executed


def test_timed_out_make_call_is_dropped_not_run_late(config):
    h, executed = _cmd_handler(config)
    assert h._queue_command("make_call", "sip:x@y") is None  # nobody processed it
    h._process_commands()  # PJSIP thread finally catches up
    assert executed == []
    assert h._result_queues == {}


def test_timed_out_hangup_still_runs(config):
    h, executed = _cmd_handler(config)
    assert h._queue_command("hangup", object()) is None
    h._process_commands()
    assert executed == ["hangup"]


def test_waiter_keeps_result_of_command_already_executing(config):
    h = SIPHandler(config, on_call_callback=None)
    h._running = True
    h._cmd_timeout_s = 0.2

    def slow_exec(cmd, args, kwargs):
        time.sleep(0.3)  # longer than the first wait window
        return "placed"

    h._execute_command = slow_exec
    stop = threading.Event()

    def pjsip_loop():
        while not stop.is_set():
            h._process_commands()
            time.sleep(0.005)

    t = threading.Thread(target=pjsip_loop, daemon=True)
    t.start()
    try:
        assert h._queue_command("make_call", "sip:x@y") == "placed"
    finally:
        stop.set()
        t.join(timeout=2)


async def test_pjsip_init_failure_is_reported(config, monkeypatch):
    def boom():
        raise RuntimeError("no transport")

    monkeypatch.setattr(sip_handler, "PJSUA_AVAILABLE", True)
    monkeypatch.setattr(sip_handler, "pj", SimpleNamespace(Endpoint=boom),
                        raising=False)
    h = SIPHandler(config, on_call_callback=None)
    assert await h.start() is False
    assert h._running is False
    assert "no transport" in (h.init_error or "")
    assert h._queue_command("make_call", "sip:x@y") is None


# --- temp WAV leaks ------------------------------------------------------------

def _tmpfile(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"RIFF")
    return str(p)


def test_stop_all_unlinks_current_file(tmp_path):
    player = PlaylistPlayer(SimpleNamespace(), "c1")
    current = _tmpfile(tmp_path, "playing.wav")
    queued = _tmpfile(tmp_path, "queued.wav")
    player._current_file = current
    player._is_playing = True
    player.file_queue.put((queued, 0.1, None))
    player.stop_all()
    assert not os.path.exists(current)
    assert not os.path.exists(queued)


def test_enqueue_on_stopped_player_unlinks_file(tmp_path):
    player = PlaylistPlayer(SimpleNamespace(), "c1")
    player.stop_all()
    late = _tmpfile(tmp_path, "late.wav")
    player.enqueue_file(late)
    assert not os.path.exists(late)
    assert player.file_queue.empty()


def test_call_end_releases_pj_player_on_pjsip_thread(config):
    """Regression (e2e crash): the PlaylistPlayer outlives the call through
    CallInfo.stream_player, so its pjsua2 AudioMediaPlayer was destroyed when
    the session was dropped on the asyncio thread -> pjlib abort(). It must be
    released in _on_call_ended, which runs on the PJSIP thread."""
    handler = SIPHandler(config, lambda *a: None)
    call = SIPCall(None, 0, handler)
    call.call_info = CallInfo(call_id="c1", remote_uri="sip:a@b",
                              is_active=False, start_time=time.time())
    player = PlaylistPlayer(handler, "c1")
    player._pj_player = object()
    call.call_info.stream_player = player
    handler.active_calls["c1"] = call
    handler._playlist_players["c1"] = player
    info = call.call_info

    handler._on_call_ended(call)

    assert player._pj_player is None
    assert player._stopped
    assert "c1" not in handler._playlist_players
    assert info.stream_player is player  # the session may still hold it
