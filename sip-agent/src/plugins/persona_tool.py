"""
Persona Tool Plugin
===================
Let the caller shape the agent's demeanor for the current call, and save/recall
those demeanors by name across calls.

The active demeanor lives on the CallSession (session.persona) and is layered
onto the system prompt every turn (see llm_engine._build_system_prompt); saved
profiles live in the PersonaStore (data/personas.json). One tool with an
`action`:

    set    - adopt a demeanor for this call ("be a terse, formal butler")
    save   - store the current demeanor under a name for later calls
    load   - adopt a previously saved demeanor by name
    list   - what profiles are saved
    clear  - drop back to the default demeanor for the rest of this call

Usage in conversation:
User: "For this call, act like an excitable game show host."
LLM: [TOOL:PERSONA:action=set,text=An excitable game show host: high energy, lots of enthusiasm]
User: "Save that as 'game show'."
LLM: [TOOL:PERSONA:action=save,name=game show]
User: "Answer like my formal butler profile."
LLM: [TOOL:PERSONA:action=load,name=formal butler]
User: "Go back to normal."
LLM: [TOOL:PERSONA:action=clear]
"""

import logging
import re
from typing import Any, Dict, Optional

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event
from persona_store import MAX_PERSONA_CHARS, normalize_name

logger = logging.getLogger(__name__)

_VALID_ACTIONS = ("set", "save", "load", "list", "clear")


class PersonaTool(BaseTool):
    """Set / save / load the agent's demeanor for a call."""

    name = "PERSONA"
    description = (
        "Change the agent's demeanor or speaking style for THIS call, or save "
        "and recall named demeanor profiles. "
        "When the caller names a saved style ('use the pirate persona', "
        "'switch to Pig Latin', 'talk like my formal butler profile'), use "
        "action=load with name set to that style — do NOT reinvent it with set. "
        "Use action=set only when the caller describes a NEW style in their own "
        "words. action=save stores the current demeanor under `name`; "
        "action=load recalls a saved demeanor by `name`; action=list names the "
        "saved profiles; action=clear returns to the default demeanor.")
    enabled = True
    speak_result = True  # informational: confirmation is spoken

    parameters = {
        "action": {
            "type": "string",
            "description": "One of: set, save, load, list, clear",
            "enum": list(_VALID_ACTIONS),
            "required": True,
        },
        "text": {
            "type": "string",
            "description": ("For action=set: the demeanor/speaking style to "
                            "adopt, as a short instruction "
                            "(e.g. 'a calm, concise expert')"),
            "required": False,
        },
        "name": {
            "type": "string",
            "description": ("For action=save/load: the profile name to store "
                            "under or recall"),
            "required": False,
        },
    }

    def _session(self):
        return getattr(self.assistant, "session", None) if self.assistant else None

    def _store(self):
        return getattr(self.assistant, "persona_store", None) if self.assistant else None

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        action = normalize_name(str(params.get("action") or "")).replace(" ", "")
        session = self._session()
        if session is None:
            return ToolResult(status=ToolStatus.FAILED,
                              message="I can only change how I'm speaking during a call.")

        if action == "set":
            return self._do_set(session, params)
        if action == "clear":
            return self._do_clear(session)
        if action == "save":
            return self._do_save(session, params)
        if action == "load":
            return self._do_load(session, params)
        if action == "list":
            return self._do_list()

        return ToolResult(
            status=ToolStatus.FAILED,
            message="I can set, save, load, list, or clear a speaking style.")

    # -- actions ---------------------------------------------------------
    def _do_set(self, session, params: Dict[str, Any]) -> ToolResult:
        text = (params.get("text") or "").strip()
        if not text:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Tell me how you'd like me to act and I'll do it.")

        # Safety net for the model reaching for `set` with its own paraphrase of
        # a SAVED profile ("use pig latin" -> set text="speaking in pig latin,
        # bouncy cartoon..."), which loses the saved profile's exact rules. If
        # the description names exactly one saved profile, load that instead so
        # the real saved text wins.
        matched = self._saved_profile_named_in(text)
        if matched is not None:
            name, saved_text = matched
            session.persona = saved_text[:MAX_PERSONA_CHARS]
            log_event(logger, logging.INFO,
                      f"Persona set redirected to saved profile: {name}",
                      event="persona_load", name=name, via="set_redirect")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=f"Alright, switching to {name}.",
                data={"name": name, "persona": session.persona})

        session.persona = text[:MAX_PERSONA_CHARS]
        log_event(logger, logging.INFO, "Persona set for call",
                  event="persona_set", chars=len(session.persona))
        return ToolResult(status=ToolStatus.SUCCESS,
                          message="Okay, I'll speak that way for the rest of this call.",
                          data={"persona": session.persona})

    def _saved_profile_named_in(self, text: str):
        """If exactly one saved profile's name appears in `text`, return
        (display_name, text); else None. Guards the set->load redirect so an
        ambiguous description never silently loads the wrong profile."""
        store = self._store()
        if store is None:
            return None
        norm = normalize_name(text)
        hits = []
        for display in store.names():
            key = normalize_name(display)
            # Match whole-word so "calm" doesn't fire inside "calminded".
            if key and re.search(rf"\b{re.escape(key)}\b", norm):
                saved = store.load(display)
                if saved:
                    hits.append((display, saved))
        return hits[0] if len(hits) == 1 else None

    def _do_clear(self, session) -> ToolResult:
        had = bool(session.persona)
        session.persona = ""
        log_event(logger, logging.INFO, "Persona cleared for call",
                  event="persona_clear")
        msg = ("Back to my normal self." if had
               else "I'm already using my normal demeanor.")
        return ToolResult(status=ToolStatus.SUCCESS, message=msg)

    def _do_save(self, session, params: Dict[str, Any]) -> ToolResult:
        store = self._store()
        if store is None:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Saving demeanors isn't available right now.")
        name = (params.get("name") or "").strip()
        # Save the current demeanor by default; allow saving a supplied text too.
        text = (params.get("text") or session.persona or "").strip()
        if not name:
            return ToolResult(status=ToolStatus.FAILED,
                              message="What would you like to call this style?")
        if not text:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="There's no demeanor set to save yet. Tell me how to act first.")
        if store.save(name, text):
            log_event(logger, logging.INFO, f"Persona saved: {name}",
                      event="persona_save", name=name)
            return ToolResult(status=ToolStatus.SUCCESS,
                              message=f"Saved that style as {name}.",
                              data={"name": name})
        return ToolResult(status=ToolStatus.FAILED,
                          message=f"I couldn't save {name}.")

    def _do_load(self, session, params: Dict[str, Any]) -> ToolResult:
        store = self._store()
        if store is None:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Saved demeanors aren't available right now.")
        name = (params.get("name") or "").strip()
        if not name:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Which saved style would you like?")
        text = store.load(name)
        if not text:
            available = store.names()
            if available:
                return ToolResult(
                    status=ToolStatus.FAILED,
                    message=(f"I don't have one called {name}. "
                             f"I have: {_join(available)}."))
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"I don't have a saved style called {name} yet.")
        session.persona = text[:MAX_PERSONA_CHARS]
        log_event(logger, logging.INFO, f"Persona loaded: {name}",
                  event="persona_load", name=name)
        return ToolResult(status=ToolStatus.SUCCESS,
                          message=f"Alright, switching to {name}.",
                          data={"name": name, "persona": session.persona})

    def _do_list(self) -> ToolResult:
        store = self._store()
        names = store.names() if store else []
        if not names:
            return ToolResult(status=ToolStatus.SUCCESS,
                              message="I don't have any saved demeanors yet.")
        return ToolResult(status=ToolStatus.SUCCESS,
                          message=f"I have these saved: {_join(names)}.",
                          data={"names": names})


def _join(items) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
