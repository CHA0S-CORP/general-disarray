"""
Caller Memory
=============
Cross-call memory about each caller, keyed by the user part of their SIP URI.
One JSON file per caller under data/caller_memory/:

    {"caller": "1001", "updated_at": "...", "call_count": 3,
     "facts": ["Name is Bob", "Prefers morning callbacks"],
     "last_call_summary": "Asked about the weather and set a timer."}

Facts are extracted by the LLM after each call (off the call path) and merged
with the existing list. Everything is fail-open: a corrupt file, a failed
extraction, or an unsafe caller id simply means no memory — never a broken
call.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from logging_utils import log_event

logger = logging.getLogger(__name__)

# Caller ids become filenames, so constrain them like transcript call_ids
# (no path separators, no dot-dot).
_SAFE_CALLER_ID = re.compile(r"[A-Za-z0-9._+-]{1,64}")

_FACT_EXTRACTION_PROMPT = """You maintain long-term memory about a phone caller for a voice assistant.
Given the existing remembered facts and the transcript of the caller's latest call, produce the updated memory.

Record ONLY durable facts about the CALLER as a person — things that will still matter on a FUTURE, unrelated call:
- their name or how they want to be addressed, their location, timezone, or language
- stable preferences and standing instructions ("always text me the address", "I'm hard of hearing, speak slowly")
- commitments or open items you agreed to ("calling back Friday about the invoice")

Do NOT record (these make every future call worse):
- the assistant's persona, voice, tone, or speaking style, or ANY request to talk, act, or reply a certain way (e.g. "talk like a pirate", "use Pig Latin", "be formal", "use the newscaster persona"). These apply ONLY to the call they were made in and must NEVER carry over.
- the content of one-off questions the caller asked this call (recipes, definitions, trivia, weather, showtimes, facts they looked up). Remember facts ABOUT the caller, not what they happened to ask this time.
- small talk, jokes, games, or transient chit-chat.
- anything you are not confident is durable — when in doubt, leave it out.

Merge new durable facts into the existing ones; correct facts the new call contradicts; keep still-valid old facts. ALSO remove any existing fact that violates the rules above (e.g. a previously-saved persona/style preference or one-off question) — clean up past mistakes.
Each fact is one short sentence about the caller.
Also write a one-sentence summary of this latest call: describe what the caller wanted or discussed, but do NOT mention the persona or speaking style used (this summary is shown on the next call, so a persona mention there would leak too).
Respond with ONLY a JSON object, no other text:
{"facts": ["fact one", "fact two"], "last_call_summary": "one sentence"}"""


def caller_id_from_uri(remote_uri: str) -> Optional[str]:
    """'sip:1001@pbx' / '"Bob" <sip:1001@pbx>' -> '1001'; None when unusable."""
    m = re.search(r'sips?:([^@;>\s]+)', remote_uri or "")
    caller = m.group(1) if m else (remote_uri or "").strip()
    if caller and _SAFE_CALLER_ID.fullmatch(caller):
        return caller
    return None


class CallerMemoryStore:
    """File-per-caller JSON memory (same pattern as TranscriptStore)."""

    def __init__(self, config):
        self.config = config
        self._dir: Path = config.data_dir / "caller_memory"
        self._dir.mkdir(parents=True, exist_ok=True)
        # One lock per caller: update_from_call holds it across the LLM
        # extraction so two overlapping post-call updates for the same caller
        # can't each write from their own stale snapshot.
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, caller_id: str) -> asyncio.Lock:
        lock = self._locks.get(caller_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[caller_id] = lock
        return lock

    def _path(self, caller_id: str) -> Optional[Path]:
        if not caller_id or not _SAFE_CALLER_ID.fullmatch(caller_id):
            return None
        return self._dir / f"{caller_id}.json"

    def get(self, caller_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(caller_id)
        if path is None or not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
            if isinstance(record, dict):
                return record
        except Exception as e:
            logger.warning(f"Could not read caller memory for {caller_id}: {e}")
        return None

    def format_for_prompt(self, caller_id: str) -> str:
        """Render a caller's memory for the system prompt, bounded by the
        configured fact/char limits. Empty string when nothing is known."""
        record = self.get(caller_id)
        if not record:
            return ""
        facts = [f for f in record.get("facts", []) if isinstance(f, str) and f.strip()]
        facts = facts[: self.config.caller_memory_max_facts]
        lines = [f"- {fact}" for fact in facts]
        last = record.get("last_call_summary")
        if isinstance(last, str) and last.strip():
            lines.append(f"- Last call: {last.strip()}")
        text = "\n".join(lines)
        max_chars = self.config.caller_memory_max_chars
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] or text[:max_chars]
        return text

    async def update_from_call(self, remote_uri: str,
                               transcript: Optional[Dict[str, Any]],
                               llm_engine) -> None:
        """Extract/merge facts from a finished call's transcript. Fail-open:
        on any failure the existing memory file is left untouched."""
        caller_id = caller_id_from_uri(remote_uri)
        if caller_id is None:
            return
        turns = (transcript or {}).get("turns") or []
        # A call with no caller speech teaches us nothing.
        if not any(t.get("role") == "user" and t.get("content") for t in turns):
            return

        async with self._lock_for(caller_id):
            await self._extract_and_merge(caller_id, turns, llm_engine)

    async def _extract_and_merge(self, caller_id: str, turns: List[Dict[str, Any]],
                                 llm_engine) -> None:
        existing = self.get(caller_id) or {}
        existing_facts = [f for f in existing.get("facts", []) if isinstance(f, str)]

        lines = []
        if existing_facts:
            lines.append("Existing facts:")
            lines.extend(f"- {fact}" for fact in existing_facts)
        else:
            lines.append("Existing facts: (none)")
        lines.append("\nTranscript of the latest call:")
        for t in turns:
            role = "Caller" if t.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {t.get('content', '')}")

        try:
            raw = await llm_engine.summarize_text(
                _FACT_EXTRACTION_PROMPT, "\n".join(lines),
                self.config.caller_memory_timeout_s)
            if not raw:
                return
            parsed = self._parse_extraction(raw)
            if parsed is None:
                log_event(logger, logging.WARNING,
                          f"Caller memory extraction unparseable for {caller_id}",
                          event="caller_memory", outcome="rejected",
                          caller=caller_id)
                return

            facts, last_summary = parsed

            # Re-read under no-await conditions: the caller may have redialled
            # and used REMEMBER/FORGET while the extraction above was running,
            # and those synchronous writes must not be clobbered by our stale
            # snapshot. (Everything from here to _write_atomic is atomic on
            # the event loop.)
            current = self.get(caller_id) or {}
            current_facts = [f for f in current.get("facts", []) if isinstance(f, str)]
            added_since = [f for f in current_facts if f not in existing_facts]
            removed_since = [f for f in existing_facts if f not in current_facts]

            merged = [f for f in facts if f not in removed_since]
            merged.extend(f for f in added_since if f not in merged)

            record = {
                "caller": caller_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "call_count": int(current.get("call_count", 0)) + 1,
                "facts": merged[-self.config.caller_memory_max_facts:],
                "last_call_summary": last_summary,
            }
            self._write_atomic(caller_id, record)
            log_event(logger, logging.INFO,
                      f"Caller memory updated for {caller_id} "
                      f"({len(record['facts'])} facts)",
                      event="caller_memory", outcome="ok",
                      caller=caller_id, facts=len(record["facts"]))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Caller memory update failed for {caller_id}: {e}")

    def add_fact(self, caller_id: str, fact: str) -> bool:
        """Explicitly remember one fact for a caller (REMEMBER tool).

        Appends to the existing record (creating one if needed), bounded by
        the configured max facts (oldest dropped first). Returns False on any
        failure — callers speak an apology instead of raising.
        """
        fact = (fact or "").strip()
        if not fact or self._path(caller_id) is None:
            return False
        try:
            record = self.get(caller_id) or {
                "caller": caller_id, "call_count": 0,
                "facts": [], "last_call_summary": "",
            }
            facts = [f for f in record.get("facts", []) if isinstance(f, str)]
            if fact not in facts:
                facts.append(fact)
            record["facts"] = facts[-self.config.caller_memory_max_facts:]
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_atomic(caller_id, record)
            return True
        except Exception as e:
            logger.warning(f"add_fact failed for {caller_id}: {e}")
            return False

    def remove_facts(self, caller_id: str, needle: str) -> int:
        """Forget facts matching `needle` (case-insensitive substring).

        Returns how many facts were removed (0 on no match or failure).
        """
        needle = (needle or "").strip().lower()
        if not needle:
            return 0
        try:
            record = self.get(caller_id)
            if not record:
                return 0
            facts = [f for f in record.get("facts", []) if isinstance(f, str)]
            kept = [f for f in facts if needle not in f.lower()]
            removed = len(facts) - len(kept)
            if removed:
                record["facts"] = kept
                record["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._write_atomic(caller_id, record)
            return removed
        except Exception as e:
            logger.warning(f"remove_facts failed for {caller_id}: {e}")
            return 0

    @staticmethod
    def _parse_extraction(raw: str):
        """Parse the extraction LLM's JSON (tolerating fencing/prose around
        it). Returns (facts, last_call_summary) or None."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        facts = [f.strip() for f in data.get("facts", [])
                 if isinstance(f, str) and f.strip()]
        last = data.get("last_call_summary")
        last = last.strip() if isinstance(last, str) else ""
        if not facts and not last:
            return None
        return facts, last

    def _write_atomic(self, caller_id: str, record: Dict[str, Any]) -> None:
        path = self._path(caller_id)
        if path is None:
            return
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.rename(path)
