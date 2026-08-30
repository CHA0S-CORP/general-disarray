"""
DTMF code collection
====================
One keypad-entry loop shared by the two identity-verification paths — the
in-call VERIFY tool (plugins/verify_tool.py) and the outbound
``POST /verify/call`` flow (api.OutboundCallHandler) — so their semantics
can't drift.

Behaviour:
- The prompt (``prompt_audio``) is queued non-blocking; the caller's FIRST
  keypress mutes it (barge-in via the playlist player's clear()) and is kept as
  the first digit, so nobody has to wait out the prompt.
- Digits accumulate; ``#`` submits early; ``*`` clears the entry so far AND
  restarts the first-digit timer (a restart is a fresh attempt at entry, not
  an abort); a length cap bounds the entry.
- Two timers: ``timeout`` bounds the wait for the FIRST digit; once digits are
  being keyed an ``interdigit`` gap auto-submits, so a time-based code isn't
  left to expire while we wait for ``#``.
- Returns the digit string, or None on timeout/hangup with nothing entered.

The code is never spoken, so it never reaches STT. NEVER log it.
"""

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Longest code accepted before auto-submitting (TOTP is 6, PINs are short);
# entry is normally ended by '#'. A safety cap, not a real limit.
MAX_CODE_LEN = 12


def mute_playback(sip, call_info) -> None:
    """Barge-in: flush the prompt currently playing into the call (best-effort).

    Uses the playlist player's clear() (transient flush, NOT stop_all which
    latches the player stopped). No-ops for handlers/test doubles without one.
    """
    try:
        get_player = getattr(sip, "get_playlist_player", None)
        player = get_player(call_info) if get_player else None
        if player is not None:
            player.clear()
    except Exception as e:
        logger.debug(f"dtmf barge-in mute failed: {e}")


async def collect_dtmf_code(sip, call_info, *, timeout: float, interdigit: float,
                            prompt_audio: Optional[bytes] = None,
                            max_len: int = MAX_CODE_LEN) -> Optional[str]:
    """Play ``prompt_audio`` (if any) and collect a keypad code. See module doc."""
    get_dtmf = getattr(sip, "get_dtmf_digit", None)
    clear_dtmf = getattr(sip, "clear_dtmf", None)
    if not get_dtmf:
        return None
    # Drop any keys buffered before the prompt so stale digits don't
    # pre-answer it; keys pressed once the prompt starts are kept below.
    if clear_dtmf:
        clear_dtmf(call_info)
    if prompt_audio:
        await sip.send_audio(call_info, prompt_audio)

    digits: List[str] = []
    muted = False
    loop = asyncio.get_event_loop()
    start = loop.time()
    last_key = start
    while True:
        if not getattr(call_info, "is_active", False):
            break
        now = loop.time()
        # Before the first digit: wait up to `timeout`. After: submit once
        # the caller pauses for `interdigit` seconds.
        if not digits and now - start >= timeout:
            break
        if digits and now - last_key >= interdigit:
            break
        digit = get_dtmf(call_info)
        if digit is None:
            await asyncio.sleep(0.05)
            continue
        # First keypress silences the still-playing prompt (barge-in).
        if not muted:
            mute_playback(sip, call_info)
            muted = True
        if digit == "#":
            break
        if digit == "*":
            # Restart entry: clear digits and give the caller a fresh
            # first-digit window rather than aborting on the old clock.
            digits = []
            start = now
            last_key = now
            continue
        if digit.isdigit():
            digits.append(digit)
            last_key = now
            if len(digits) >= max_len:
                break
    return "".join(digits) if digits else None
