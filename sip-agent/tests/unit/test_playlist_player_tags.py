"""Unit tests for PlaylistPlayer tag bookkeeping (the playback ledger).

Drives _poll_and_update directly (the PJSIP-thread entry point) with a stub
pj_call and a fake `pj` module, exactly as the real handler would, and checks
that snapshot() tells the truth about which tagged files finished playing.
"""
import time
import wave
from types import SimpleNamespace

import pytest

import sip_handler
from sip_handler import PlaylistPlayer

pytestmark = pytest.mark.unit


class _FakeAudioMediaPlayer:
    """Stands in for pj.AudioMediaPlayer (no real PJSIP in unit tests)."""

    def createPlayer(self, path, flags):
        pass

    def startTransmit(self, aud_med):
        pass

    def stopTransmit(self, aud_med):
        pass


@pytest.fixture
def fake_pj(monkeypatch):
    fake = SimpleNamespace(AudioMediaPlayer=_FakeAudioMediaPlayer,
                           PJMEDIA_FILE_NO_LOOP=0)
    # raising=False: in mock mode (no pjsua2) the module has no `pj` at all.
    monkeypatch.setattr(sip_handler, "pj", fake, raising=False)
    return fake


@pytest.fixture
def pj_call():
    """Stub pj_call: has audio media and an active call (playback can start)."""
    return SimpleNamespace(aud_med=object(),
                           call_info=SimpleNamespace(is_active=True))


@pytest.fixture
def player():
    return PlaylistPlayer(SimpleNamespace(), "call-1")


def _wav(tmp_path, name, seconds=0.2, rate=8000):
    path = tmp_path / name
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * int(seconds * rate))
    return str(path)


def _finish_current(player):
    """Rewind the current file's start time so the next poll sees it done."""
    with player._lock:
        player._current_start = time.time() - 1000


def test_idle_snapshot(player):
    assert player.snapshot() == ([], None, 0.0)


def test_tags_complete_in_play_order(fake_pj, pj_call, player, tmp_path):
    player.enqueue_file(_wav(tmp_path, "a.wav"), tag=1)
    player.enqueue_file(_wav(tmp_path, "b.wav"), tag=2)

    player._poll_and_update(pj_call)  # starts file 1
    completed, current, fraction = player.snapshot()
    assert completed == []
    assert current == 1
    assert 0.0 <= fraction <= 1.0

    _finish_current(player)
    player._poll_and_update(pj_call)  # file 1 done, file 2 starts
    completed, current, _ = player.snapshot()
    assert completed == [1]
    assert current == 2

    _finish_current(player)
    player._poll_and_update(pj_call)  # file 2 done, queue empty
    assert player.snapshot() == ([1, 2], None, 0.0)


def test_untagged_files_never_enter_ledger(fake_pj, pj_call, player, tmp_path):
    player.enqueue_file(_wav(tmp_path, "chime.wav"))  # no tag
    player._poll_and_update(pj_call)
    completed, current, fraction = player.snapshot()
    assert completed == []
    assert current is None  # playing, but untagged
    _finish_current(player)
    player._poll_and_update(pj_call)
    assert player.snapshot() == ([], None, 0.0)


def test_clear_does_not_complete_interrupted_file(fake_pj, pj_call, player, tmp_path):
    player.enqueue_file(_wav(tmp_path, "a.wav"), tag=1)
    player.enqueue_file(_wav(tmp_path, "b.wav"), tag=2)
    player._poll_and_update(pj_call)  # file 1 playing
    assert player.snapshot()[1] == 1

    player.clear()  # barge-in: drops queued file 2, flags a flush
    player._poll_and_update(pj_call)  # PJSIP thread picks up the flush

    completed, current, fraction = player.snapshot()
    assert completed == []           # interrupted file 1 did NOT finish
    assert current is None
    assert fraction == 0.0
    assert player.file_queue.empty()  # queued-and-dropped tag 2 is gone

    # Player still usable after clear(): a new tagged file plays and completes.
    player.enqueue_file(_wav(tmp_path, "c.wav"), tag=3)
    player._poll_and_update(pj_call)
    _finish_current(player)
    player._poll_and_update(pj_call)
    assert player.snapshot()[0] == [3]


def test_completed_survive_clear(fake_pj, pj_call, player, tmp_path):
    player.enqueue_file(_wav(tmp_path, "a.wav"), tag=1)
    player.enqueue_file(_wav(tmp_path, "b.wav"), tag=2)
    player._poll_and_update(pj_call)
    _finish_current(player)
    player._poll_and_update(pj_call)  # tag 1 completed, tag 2 playing

    player.clear()
    player._poll_and_update(pj_call)

    assert player.snapshot() == ([1], None, 0.0)


def test_stop_all_does_not_complete_interrupted_file(fake_pj, pj_call, player, tmp_path):
    player.enqueue_file(_wav(tmp_path, "a.wav"), tag=1)
    player._poll_and_update(pj_call)
    player.stop_all()
    completed, current, _ = player.snapshot()
    assert completed == []
    assert current is None


def test_snapshot_fraction_clamped(fake_pj, pj_call, player, tmp_path):
    player.enqueue_file(_wav(tmp_path, "a.wav"), tag=1)
    player._poll_and_update(pj_call)

    with player._lock:
        player._current_start = time.time() + 100  # "starts in the future"
    assert player.snapshot()[2] == 0.0

    with player._lock:
        player._current_start = time.time() - 100  # long past the duration
    assert player.snapshot()[2] == 1.0
