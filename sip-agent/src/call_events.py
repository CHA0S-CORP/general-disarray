"""
Call Events
===========
Pure helpers for the call-lifecycle event webhook (``CALL_EVENT_WEBHOOK_URL``).

Builds the ``call.started`` / ``call.ended`` payloads and parses the
``CALL_EVENTS`` filter. Delivery itself goes through ``api.deliver_webhook``
(SSRF pinning, HMAC signing, retries) — see ``SIPAIAssistant._emit_call_event``.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

KNOWN_EVENTS = {"call.started", "call.ended"}

# Unknown names found in CALL_EVENTS that were already warned about (warn once).
_warned_unknown: Set[str] = set()


def enabled_events(config) -> Set[str]:
    """Parse config.call_events into the set of events to emit.

    Unknown event names are logged once and ignored.
    """
    names = {n.strip() for n in config.call_events.split(",") if n.strip()}
    unknown = names - KNOWN_EVENTS
    for name in unknown - _warned_unknown:
        _warned_unknown.add(name)
        logger.warning(
            "Unknown event %r in CALL_EVENTS (known: %s); ignoring",
            name, ", ".join(sorted(KNOWN_EVENTS)))
    return names & KNOWN_EVENTS


def build_call_event_payload(event: str, session,
                             transcript_record: Optional[Dict[str, Any]],
                             include_transcript: bool) -> Dict[str, Any]:
    """Build the JSON payload for a call-lifecycle event webhook.

    `transcript_record` is the TranscriptStore record for the session (may be
    None if the store has already evicted it); `call.ended` embeds it whole
    when `include_transcript` is set.
    """
    record = transcript_record or {}
    started_at = record.get("started_at") or datetime.fromtimestamp(
        session.start_time, tz=timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "event": event,
        "call_id": session.transcript_id,
        "sip_call_id": getattr(session.call_info, "call_id", None),
        "direction": session.direction,
        "remote_uri": record.get("remote_uri")
        or getattr(session.call_info, "remote_uri", ""),
        "started_at": started_at,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if event == "call.ended":
        payload["duration_seconds"] = round(time.time() - session.start_time, 1)
        if include_transcript:
            payload["transcript"] = transcript_record
    return payload
