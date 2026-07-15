"""
Persona Store
=============
Named demeanor/behavior profiles for the agent, persisted across restarts.

A "persona" is a short instruction block that shapes how the agent speaks for
the duration of ONE call ("be terse and formal", "talk like a friendly pirate")
— it is layered on top of the base system prompt, never replacing it. The
active persona lives on the CallSession (per-call); this store is only the
save/load side: callers can name a demeanor, save it, and ask for it back on a
later call.

Storage is a single JSON object {name: text} under data/personas.json. Small,
human-editable, and shared across all callers by design — a persona is a
property of the agent, not of who dialed in. Fail-open throughout: a corrupt or
unwritable file must never take a call down.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# A persona is a system-prompt fragment, not an essay. Cap it so a runaway
# instruction can't blow out the context window or the JSON file.
MAX_PERSONA_CHARS = 600
MAX_NAME_CHARS = 60
# A safety valve on the number of saved profiles — this is a toy feature, not a
# database. Well above any real use.
MAX_PROFILES = 200


def normalize_name(name: str) -> str:
    """Canonical lookup key for a profile name: trimmed, lowercased, internal
    whitespace collapsed. "Formal Butler", "formal  butler" -> one profile."""
    return " ".join((name or "").split()).lower()


class PersonaStore:
    """Load/save named persona profiles in data/personas.json.

    Thread-safe (the SIP command thread and the async loop can both reach it);
    all disk access is guarded by a single lock and every path fails open.
    """

    def __init__(self, config):
        self.config = config
        self.path: Path = getattr(config, "persona_file", None) or (
            Path(getattr(config, "data_dir", Path("./data"))) / "personas.json")
        self._lock = threading.Lock()

    # -- disk ------------------------------------------------------------
    def _load_raw(self) -> Dict[str, Dict[str, str]]:
        """Read the whole file. Returns {} on any problem (missing, corrupt)."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, dict)}
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not read persona store {self.path}: {e}")
        return {}

    def _write_raw(self, data: Dict[str, Dict[str, str]]) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.replace(self.path)
            return True
        except Exception as e:
            logger.warning(f"Could not write persona store {self.path}: {e}")
            return False

    # -- API -------------------------------------------------------------
    def save(self, name: str, text: str) -> bool:
        """Save (or overwrite) a named profile. Returns False on bad input or a
        write failure."""
        key = normalize_name(name)
        text = (text or "").strip()
        if not key or not text:
            return False
        with self._lock:
            data = self._load_raw()
            if key not in data and len(data) >= MAX_PROFILES:
                logger.warning("Persona store full; refusing new profile")
                return False
            # Preserve the display name as the caller said it; key on the
            # normalized form.
            data[key] = {"name": name.strip()[:MAX_NAME_CHARS],
                         "text": text[:MAX_PERSONA_CHARS]}
            return self._write_raw(data)

    def load(self, name: str) -> Optional[str]:
        """Return a saved profile's persona text, or None if there's no match."""
        key = normalize_name(name)
        if not key:
            return None
        with self._lock:
            entry = self._load_raw().get(key)
        return entry.get("text") if entry else None

    def delete(self, name: str) -> bool:
        """Remove a saved profile. Returns True only if one was removed."""
        key = normalize_name(name)
        if not key:
            return False
        with self._lock:
            data = self._load_raw()
            if key not in data:
                return False
            del data[key]
            return self._write_raw(data)

    def names(self) -> List[str]:
        """Display names of all saved profiles, alphabetical."""
        with self._lock:
            data = self._load_raw()
        return sorted(e.get("name", k) for k, e in data.items())
