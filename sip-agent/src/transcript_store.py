"""
Transcript Store
================
Per-call conversation transcripts: live calls accumulate turns in memory, and
finished calls are kept in a bounded LRU plus persisted as JSON files under
data/transcripts/ so they survive restarts.
"""

import json
import logging
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# call_ids become filenames, so constrain them hard (no path separators, no
# dot-dot). Anything else is stored in memory only.
_SAFE_CALL_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class TranscriptStore:
    """Records who said what on each call.

    Turns are appended while a call is live; end() persists the transcript to
    data/transcripts/<call_id>.json. get() serves the active call first, then
    a bounded in-memory LRU of recent calls, then falls back to disk.
    """

    MAX_RECENT = 50
    MAX_TURNS_PER_CALL = 500

    def __init__(self, config):
        self._dir: Path = config.data_dir / "transcripts"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._active: Dict[str, Dict[str, Any]] = {}
        self._recent: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _path(self, call_id: str) -> Optional[Path]:
        if not _SAFE_CALL_ID.fullmatch(call_id or ""):
            return None
        return self._dir / f"{call_id}.json"

    def start(self, call_id: str, direction: str, remote_uri: str = "") -> None:
        """Begin a transcript for a call. Idempotent per call_id."""
        if not call_id or call_id in self._active:
            return
        self._active[call_id] = {
            "call_id": call_id,
            "direction": direction,
            "remote_uri": remote_uri,
            "started_at": self._now(),
            "ended_at": None,
            "turns": [],
        }

    def add_turn(self, call_id: str, role: str, content: str) -> None:
        record = self._active.get(call_id)
        if record is None or not content:
            return
        if len(record["turns"]) >= self.MAX_TURNS_PER_CALL:
            return
        record["turns"].append({"role": role, "content": content, "ts": self._now()})

    def remove_last_turn(self, call_id: str, role: str, content: str) -> bool:
        """Retract the most recent turn of a live call iff it matches.

        Used by the speculative cancel-merge path: the cancelled turn's user
        fragment was already recorded but will be re-added merged with the
        follow-up speech, so it must not linger as a phantom duplicate. The
        exact-match guard makes a stale/raced call a no-op.
        """
        record = self._active.get(call_id)
        if record is None or not record["turns"]:
            return False
        last = record["turns"][-1]
        if last.get("role") == role and last.get("content") == content:
            record["turns"].pop()
            return True
        return False

    def end(self, call_id: str) -> None:
        """Finish a transcript: move to the recent LRU and persist to disk."""
        record = self._active.pop(call_id, None)
        if record is None:
            return
        record["ended_at"] = self._now()
        self._recent[call_id] = record
        while len(self._recent) > self.MAX_RECENT:
            self._recent.popitem(last=False)
        path = self._path(call_id)
        if path is None:
            logger.warning(f"Not persisting transcript for unsafe call_id: {call_id!r}")
            return
        try:
            path.write_text(json.dumps(record, indent=2))
        except Exception as e:
            logger.error(f"Failed to persist transcript for {call_id}: {e}")

    def list_recent(self) -> List[Dict[str, Any]]:
        """Summaries of live + recent calls (newest first) for the admin UI.

        Covers what the store already holds in memory (live calls plus the
        bounded LRU of finished calls); transcripts that only exist on disk
        are not enumerated. Returns metadata only — turn contents stay behind
        GET /call/{id}/transcript.
        """
        def _summary(record: Dict[str, Any], live: bool) -> Dict[str, Any]:
            return {
                "call_id": record["call_id"],
                "direction": record.get("direction", ""),
                "remote_uri": record.get("remote_uri", ""),
                "started_at": record.get("started_at"),
                "ended_at": record.get("ended_at"),
                "turns": len(record.get("turns", [])),
                "live": live,
            }

        items = [_summary(r, False) for r in self._recent.values()]
        items += [_summary(r, True) for r in self._active.values()]
        items.sort(key=lambda s: s["started_at"] or "", reverse=True)
        return items

    def get(self, call_id: str) -> Optional[Dict[str, Any]]:
        if call_id in self._active:
            return self._active[call_id]
        if call_id in self._recent:
            return self._recent[call_id]
        path = self._path(call_id)
        if path is not None and path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.error(f"Failed to read transcript for {call_id}: {e}")
        return None
