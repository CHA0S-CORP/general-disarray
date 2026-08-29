"""Unit tests for the shared DTMF code collector."""
import asyncio

import pytest

from dtmf_collect import collect_dtmf_code

pytestmark = pytest.mark.unit


class _Call:
    is_active = True


class _Sip:
    def __init__(self, script):
        # script: list of digits or floats (sleep seconds before next digit)
        self._script = list(script)
        self.cleared = 0
        self.sent = []
        self.player_clears = 0
        self._loop = asyncio.get_event_loop()
        self._ready_at = self._loop.time()

    def clear_dtmf(self, call_info):
        self.cleared += 1

    async def send_audio(self, call_info, audio, tag=None):
        self.sent.append(audio)

    def get_playlist_player(self, call_info):
        sip = self

        class _P:
            def clear(self_inner):
                sip.player_clears += 1
        return _P()

    def get_dtmf_digit(self, call_info):
        while self._script and isinstance(self._script[0], float):
            self._ready_at = max(self._ready_at, self._loop.time()) + self._script.pop(0)
        if self._loop.time() < self._ready_at or not self._script:
            return None
        return self._script.pop(0)


def _run(script, **kw):
    async def go():
        sip = _Sip(script)
        code = await collect_dtmf_code(sip, _Call(), prompt_audio=b"\x00\x00", **kw)
        return sip, code
    return asyncio.run(go())


def test_pound_submits_and_first_key_mutes_prompt():
    sip, code = _run(["1", "2", "3", "#"], timeout=1.0, interdigit=1.0)
    assert code == "123"
    assert sip.cleared == 1 and sip.sent == [b"\x00\x00"]
    assert sip.player_clears == 1


def test_interdigit_gap_auto_submits():
    sip, code = _run(["4", "5"], timeout=1.0, interdigit=0.1)
    assert code == "45"


def test_star_restarts_entry_and_the_first_digit_timer():
    # timeout 0.3s: caller keys '1' late, hits '*' after the original window
    # would have expired, then keys the real code. '*' must NOT abort.
    sip, code = _run([0.2, "1", 0.2, "*", 0.2, "9", "8", "#"], timeout=0.3, interdigit=1.0)
    assert code == "98"


def test_timeout_with_nothing_entered_is_none():
    sip, code = _run([], timeout=0.1, interdigit=1.0)
    assert code is None


def test_length_cap():
    sip, code = _run(list("1234567890123"), timeout=1.0, interdigit=1.0)
    assert code == "123456789012"
