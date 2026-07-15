"""
Virtual Numbers
===============
Ephemeral inbound extensions: a REST-created temporary number the agent
listens for in the background. A call dialed to it is answered as the normal
assistant with per-number context injected; when the call completes the
outcome (+ transcript) is webhooked to the number's callback_url and the
number is cleared. Numbers are single-use, expire after a TTL when no call
arrives, and the registry persists across restarts
(data/virtual_numbers.json, same atomic-write pattern as the scheduler).

Threading model: all registry state is touched only from the asyncio event
loop (API handlers, the call path, and the sweep task), so no locks are
needed. The PJSIP thread never calls in here — it only captures the dialed
URI string (CallInfo.local_uri).
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import Config
from logging_utils import log_event

logger = logging.getLogger(__name__)

# Extension shape for explicit numbers (matches dialable user parts).
_NUMBER_RE = re.compile(r"^[0-9*#]{2,32}$")

# Dialed user parts we extract from a To-URI. Superset of _NUMBER_RE (plain
# SIP usernames too) so the call path can pass any dialed extension to
# claim(), which matches by exact number.
_EXTENSION_RE = re.compile(r"[A-Za-z0-9._+*#-]{1,64}")


def extension_from_uri(uri: str) -> Optional[str]:
    """'sip:*77@pbx' / '<sip:1001@pbx>' -> '*77' / '1001'; None when unusable.

    Unlike caller_memory.caller_id_from_uri (which constrains ids to safe
    filenames), this keeps '*' and '#' so explicit virtual numbers like
    '*77' are matchable."""
    m = re.search(r'sips?:([^@;>\s]+)', uri or "")
    ext = m.group(1) if m else (uri or "").strip()
    if ext and _EXTENSION_RE.fullmatch(ext):
        return ext
    return None

# Sweep cadence. Expiry precision of ~1s is plenty for minutes-scale TTLs.
_SWEEP_INTERVAL_S = 1.0


class VirtualNumberError(Exception):
    """Registry-level rejection; carries an HTTP-ish status for the API."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class VirtualNumber:
    """One ephemeral inbound extension."""
    id: str
    number: str
    purpose: str
    greeting: str = ""
    callback_url: str = ""
    include_transcript: bool = True
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    # Set when a call to this number is live; a claimed entry is exempt from
    # the TTL sweep so it can't vanish mid-call.
    claimed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "number": self.number,
            "purpose": self.purpose,
            "greeting": self.greeting,
            "callback_url": self.callback_url,
            "include_transcript": self.include_transcript,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "claimed": self.claimed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VirtualNumber':
        return cls(**data)


class VirtualNumberRegistry:
    """Create/match/consume store for virtual numbers, with TTL sweeping."""

    def __init__(self, config: Config):
        self.config = config
        self._entries: Dict[str, VirtualNumber] = {}  # id -> entry
        self._sweep_task: Optional[asyncio.Task] = None
        # Strong refs to in-flight webhook deliveries (event loop won't GC them).
        self._webhook_tasks: set = set()
        # Entries found already-expired at load time; their "expired"
        # webhooks fire from start() (async context), not during __init__.
        self._expired_on_load: List[VirtualNumber] = []
        self._load()

    # -- lifecycle -----------------------------------------------------------

    async def start(self):
        if not self.config.virtual_numbers_enabled:
            return
        for entry in self._expired_on_load:
            self.fire_webhook(entry, status="expired")
        self._expired_on_load = []
        self._sweep_task = asyncio.create_task(self._sweep_loop())
        log_event(logger, logging.INFO,
                  f"Virtual number registry started ({len(self._entries)} active)",
                  event="virtual_numbers_started", active=len(self._entries))

    async def stop(self):
        if self._sweep_task:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None

    # -- CRUD ----------------------------------------------------------------

    def create(self, number: Optional[str] = None, ttl_s: Optional[int] = None,
               purpose: str = "", greeting: str = "", callback_url: str = "",
               include_transcript: bool = True) -> VirtualNumber:
        """Register a new virtual number; raises VirtualNumberError on policy
        violations (400 bad number, 409 collision, 503 exhausted/at cap)."""
        if len(self._entries) >= self.config.virtual_number_max_active:
            raise VirtualNumberError(
                503, f"Too many active virtual numbers "
                     f"(VIRTUAL_NUMBER_MAX_ACTIVE={self.config.virtual_number_max_active})")

        ttl = int(ttl_s) if ttl_s else self.config.virtual_number_default_ttl_s
        ttl = max(1, min(ttl, self.config.virtual_number_max_ttl_s))

        if number:
            number = number.strip()
            if not _NUMBER_RE.fullmatch(number):
                raise VirtualNumberError(
                    400, "Invalid number: 2-32 characters of digits, * or #")
            if number == self.config.sip_user:
                raise VirtualNumberError(
                    409, "Number collides with the agent's own SIP identity")
            if self._by_number(number):
                raise VirtualNumberError(409, f"Number {number} is already active")
        else:
            number = self._allocate()

        entry = VirtualNumber(
            id=str(uuid.uuid4())[:8],
            number=number,
            purpose=purpose,
            greeting=greeting,
            callback_url=callback_url,
            include_transcript=include_transcript,
            expires_at=time.time() + ttl,
        )
        self._entries[entry.id] = entry
        self._persist()
        log_event(logger, logging.INFO,
                  f"Virtual number created: {number} (ttl {ttl}s)",
                  event="virtual_number_created", number=number,
                  virtual_number_id=entry.id, ttl_s=ttl)
        return entry

    def get(self, entry_id: str) -> Optional[VirtualNumber]:
        return self._entries.get(entry_id)

    def list_active(self) -> List[VirtualNumber]:
        return sorted(self._entries.values(), key=lambda e: e.created_at)

    def delete(self, entry_id: str) -> bool:
        """Explicit DELETE; no webhook (the creator asked for removal)."""
        entry = self._entries.pop(entry_id, None)
        if entry is None:
            return False
        self._persist()
        log_event(logger, logging.INFO,
                  f"Virtual number deleted: {entry.number}",
                  event="virtual_number_deleted", number=entry.number,
                  virtual_number_id=entry.id)
        return True

    # -- call-path hooks -----------------------------------------------------

    def claim(self, number: str) -> Optional[VirtualNumber]:
        """Match a dialed user part to an active entry and mark it claimed
        (exempt from the sweep while its call is live)."""
        if not self.config.virtual_numbers_enabled or not number:
            return None
        entry = self._by_number(number)
        if entry is None or entry.claimed:
            return None
        entry.claimed = True
        self._persist()
        return entry

    def consume(self, entry_id: str) -> Optional[VirtualNumber]:
        """Pop an entry after its call finished (single-use). Idempotent:
        returns None when already gone."""
        entry = self._entries.pop(entry_id, None)
        if entry is not None:
            self._persist()
        return entry

    def release(self, entry_id: str):
        """Un-claim without consuming (call never actually started)."""
        entry = self._entries.get(entry_id)
        if entry is not None and entry.claimed:
            entry.claimed = False
            self._persist()

    # -- internals -----------------------------------------------------------

    def _by_number(self, number: str) -> Optional[VirtualNumber]:
        for entry in self._entries.values():
            if entry.number == number:
                return entry
        return None

    def _allocate(self) -> str:
        """Lowest free number from VIRTUAL_NUMBER_RANGE."""
        try:
            start_s, end_s = self.config.virtual_number_range.split("-", 1)
            start, end = int(start_s), int(end_s)
        except ValueError:
            raise VirtualNumberError(
                500, f"Invalid VIRTUAL_NUMBER_RANGE "
                     f"'{self.config.virtual_number_range}' (expected 'start-end')")
        taken = {e.number for e in self._entries.values()}
        taken.add(self.config.sip_user)
        for candidate in range(start, end + 1):
            if str(candidate) not in taken:
                return str(candidate)
        raise VirtualNumberError(
            503, f"No free virtual numbers in range {start}-{end}")

    async def _sweep_loop(self):
        while True:
            await asyncio.sleep(_SWEEP_INTERVAL_S)
            try:
                self._sweep()
            except Exception as e:
                logger.error(f"Virtual number sweep error: {e}")

    def _sweep(self):
        now = time.time()
        expired = [e for e in self._entries.values()
                   if not e.claimed and e.expires_at <= now]
        if not expired:
            return
        for entry in expired:
            self._entries.pop(entry.id, None)
            log_event(logger, logging.INFO,
                      f"Virtual number expired unused: {entry.number}",
                      event="virtual_number_expired", number=entry.number,
                      virtual_number_id=entry.id)
            self.fire_webhook(entry, status="expired")
        self._persist()

    def fire_webhook(self, entry: VirtualNumber, status: str,
                      extra: Optional[Dict[str, Any]] = None):
        """Fire-and-forget the number's callback_url (if any)."""
        if not entry.callback_url:
            return
        payload = {
            "event": f"virtual_number.{status}",
            "id": entry.id,
            "number": entry.number,
            "status": status,
            "purpose": entry.purpose,
            "created_at": entry.created_at,
            "timestamp": time.time(),
        }
        if extra:
            payload.update(extra)
        # Imported lazily to avoid an import cycle with api.py.
        from api import deliver_webhook
        task = asyncio.create_task(deliver_webhook(
            entry.callback_url, payload, self.config,
            api_name="virtual_number_webhook"))
        # Keep a strong reference until delivery completes.
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)

    # -- persistence ---------------------------------------------------------

    @property
    def _store_file(self):
        return self.config.data_dir / "virtual_numbers.json"

    def _persist(self):
        try:
            tmp = self._store_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(
                [e.to_dict() for e in self._entries.values()], indent=2))
            os.replace(tmp, self._store_file)
        except Exception as e:
            logger.error(f"Failed to persist virtual numbers: {e}")

    def _load(self):
        if not self._store_file.exists():
            return
        try:
            raw = json.loads(self._store_file.read_text())
        except Exception as e:
            logger.error(f"Failed to load virtual numbers: {e}")
            return
        now = time.time()
        loaded = 0
        for item in raw if isinstance(raw, list) else []:
            try:
                entry = VirtualNumber.from_dict(item)
            except Exception as e:
                logger.warning(f"Skipping bad virtual number record: {e}")
                continue
            # A claimed entry from before a crash reloads as active (its call
            # outcome is lost); expiry then applies normally.
            entry.claimed = False
            if entry.expires_at <= now:
                self._expired_on_load.append(entry)
                continue
            self._entries[entry.id] = entry
            loaded += 1
        if loaded or self._expired_on_load:
            logger.info(f"Loaded {loaded} virtual numbers "
                        f"({len(self._expired_on_load)} expired while down)")
            self._persist()
