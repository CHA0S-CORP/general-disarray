"""Unit tests for the PERSONA tool and its PersonaStore, over a temporary data
directory. Covers the per-call demeanor (set/clear on the session) and the
save/load/list persistence, plus the prompt-injection wiring."""
import pytest

from types import SimpleNamespace

from call_session import CallSession
from persona_store import PersonaStore, normalize_name
from plugins.persona_tool import PersonaTool
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def make_assistant(tmp_path, config_factory, with_session=True, **overrides):
    cfg = config_factory(data_dir=str(tmp_path), **overrides)
    store = PersonaStore(cfg)
    session = (CallSession(call_info=SimpleNamespace(remote_uri="sip:1001@pbx"),
                           direction="inbound", transcript_id="t1")
               if with_session else None)
    return SimpleNamespace(config=cfg, session=session, persona_store=store)


# --- set / clear (per call) --------------------------------------------------

async def test_set_applies_persona_to_the_session(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    tool = PersonaTool(a)

    result = await tool.execute({"action": "set", "text": "a terse, formal butler"})

    assert result.status == ToolStatus.SUCCESS
    assert a.session.persona == "a terse, formal butler"


async def test_set_requires_text(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    result = await PersonaTool(a).execute({"action": "set", "text": "  "})
    assert result.status == ToolStatus.FAILED
    assert a.session.persona == ""


async def test_clear_reverts_to_default(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    a.session.persona = "a pirate"
    result = await PersonaTool(a).execute({"action": "clear"})
    assert result.status == ToolStatus.SUCCESS
    assert a.session.persona == ""


async def test_set_naming_a_saved_profile_redirects_to_load(tmp_path, config_factory):
    """The model often reaches for set with its own paraphrase of a saved
    profile ("use pig latin" -> set text="speaking in pig latin, bouncy..."),
    losing the saved rules. When the description names a saved profile, the
    real saved text must win."""
    a = make_assistant(tmp_path, config_factory)
    a.persona_store.save("Pig Latin", "Reply ONLY in Pig Latin. Every word.")

    result = await PersonaTool(a).execute({
        "action": "set",
        "text": "speaking in pig latin, like a playful bouncy cartoon character",
    })

    assert result.status == ToolStatus.SUCCESS
    # The saved profile's exact text is applied, not the model's paraphrase.
    assert a.session.persona == "Reply ONLY in Pig Latin. Every word."
    assert "Pig Latin" in result.message


async def test_set_with_novel_description_is_not_redirected(tmp_path, config_factory):
    """A genuinely new style that matches no saved profile is set as-is."""
    a = make_assistant(tmp_path, config_factory)
    a.persona_store.save("Pirate", "Talk like a pirate.")

    result = await PersonaTool(a).execute({
        "action": "set", "text": "a sleepy, mumbling night-shift clerk",
    })
    assert result.status == ToolStatus.SUCCESS
    assert a.session.persona == "a sleepy, mumbling night-shift clerk"


async def test_multiword_profile_wins_over_incidental_adjective(tmp_path, config_factory):
    """The model's paraphrase brushes an incidental adjective that's also a
    profile name ('...Pig Latin...cheerful tone'). The specific multi-word
    profile must win, not bail on the ambiguity — this was the live bug where
    Pig Latin never loaded and the vague description was used instead."""
    a = make_assistant(tmp_path, config_factory)
    a.persona_store.save("Pig Latin", "Reply ONLY in Pig Latin.")
    a.persona_store.save("Cheerful", "Be upbeat.")

    result = await PersonaTool(a).execute({
        "action": "set",
        "text": "a bouncy voice that speaks in Pig Latin with a cheerful tone",
    })
    assert a.session.persona == "Reply ONLY in Pig Latin."
    assert "Pig Latin" in result.message


async def test_two_incidental_adjectives_stay_a_novel_set(tmp_path, config_factory):
    """Two single-word profile-name adjectives with no specific winner is a
    genuine novel description — set as-is, don't hijack it."""
    a = make_assistant(tmp_path, config_factory)
    a.persona_store.save("Calm", "Be calm.")
    a.persona_store.save("Witty", "Be witty.")

    await PersonaTool(a).execute({"action": "set", "text": "a calm, witty expert"})
    assert a.session.persona == "a calm, witty expert"


async def test_set_redirect_is_word_boundary_safe(tmp_path, config_factory):
    """A saved name must match as a whole word, not inside another word."""
    a = make_assistant(tmp_path, config_factory)
    a.persona_store.save("Calm", "Be calm and steady.")
    # "calminded" contains "calm" but is not naming the Calm profile.
    result = await PersonaTool(a).execute({
        "action": "set", "text": "a calminded improviser with wild energy",
    })
    assert a.session.persona == "a calminded improviser with wild energy"


async def test_persona_is_capped(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    await PersonaTool(a).execute({"action": "set", "text": "x" * 5000})
    assert len(a.session.persona) <= 600


async def test_no_session_fails_gracefully(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory, with_session=False)
    result = await PersonaTool(a).execute({"action": "set", "text": "a pirate"})
    assert result.status == ToolStatus.FAILED


# --- save / load / list (across calls) ---------------------------------------

async def test_save_then_load_roundtrip(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    tool = PersonaTool(a)
    await tool.execute({"action": "set", "text": "an excitable game show host"})

    saved = await tool.execute({"action": "save", "name": "Game Show"})
    assert saved.status == ToolStatus.SUCCESS

    # A fresh call: new session, same store (persistence).
    a.session = CallSession(call_info=SimpleNamespace(remote_uri="sip:2002@pbx"),
                            direction="inbound", transcript_id="t2")
    loaded = await PersonaTool(a).execute({"action": "load", "name": "game show"})
    assert loaded.status == ToolStatus.SUCCESS
    assert a.session.persona == "an excitable game show host"


async def test_set_with_name_saves_and_applies(tmp_path, config_factory):
    """The model conflates set and save: "save a style named X, the style is Y"
    comes in as set(text=Y, name=X). That must save Y under X AND apply it —
    not get hijacked by the set->load redirect."""
    a = make_assistant(tmp_path, config_factory)
    # Pirate is a saved profile; the text names it, which would otherwise
    # redirect. The name param must win (this is a save, not a reference).
    a.persona_store.save("Pirate", "seed pirate")

    result = await PersonaTool(a).execute({
        "action": "set",
        "text": "talk like a pirate, say ahoy and call me matey",
        "name": "Space Captain",
    })
    assert result.status == ToolStatus.SUCCESS
    # Saved under the new name...
    assert a.persona_store.load("space captain") == \
        "talk like a pirate, say ahoy and call me matey"
    # ...and applied to the call now.
    assert a.session.persona == "talk like a pirate, say ahoy and call me matey"


async def test_save_uses_current_persona_when_no_text(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    a.session.persona = "calm and concise"
    result = await PersonaTool(a).execute({"action": "save", "name": "zen"})
    assert result.status == ToolStatus.SUCCESS
    assert a.persona_store.load("zen") == "calm and concise"


async def test_save_without_name_fails(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    a.session.persona = "x"
    result = await PersonaTool(a).execute({"action": "save"})
    assert result.status == ToolStatus.FAILED


async def test_save_with_nothing_to_save_fails(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    result = await PersonaTool(a).execute({"action": "save", "name": "empty"})
    assert result.status == ToolStatus.FAILED


async def test_load_unknown_name_lists_alternatives(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    a.persona_store.save("Butler", "formal")
    result = await PersonaTool(a).execute({"action": "load", "name": "pirate"})
    assert result.status == ToolStatus.FAILED
    assert "Butler" in result.message


async def test_list_reports_saved_names(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    a.persona_store.save("Butler", "formal")
    a.persona_store.save("Pirate", "arr")
    result = await PersonaTool(a).execute({"action": "list"})
    assert result.status == ToolStatus.SUCCESS
    assert "Butler" in result.message and "Pirate" in result.message


async def test_list_empty(tmp_path, config_factory):
    a = make_assistant(tmp_path, config_factory)
    result = await PersonaTool(a).execute({"action": "list"})
    assert result.status == ToolStatus.SUCCESS
    assert "don't have any" in result.message.lower()


# --- store internals ---------------------------------------------------------

def test_normalize_name_collapses_case_and_space():
    assert normalize_name("  Formal   Butler ") == "formal butler"


def test_store_delete(tmp_path, config_factory):
    cfg = config_factory(data_dir=str(tmp_path))
    store = PersonaStore(cfg)
    store.save("temp", "text")
    assert store.delete("TEMP") is True
    assert store.load("temp") is None
    assert store.delete("temp") is False


def test_store_survives_corrupt_file(tmp_path, config_factory):
    cfg = config_factory(data_dir=str(tmp_path))
    store = PersonaStore(cfg)
    store.path.write_text("{not valid json", encoding="utf-8")
    # Fail-open: reads as empty, and a save still repairs the file.
    assert store.names() == []
    assert store.save("ok", "text") is True
    assert store.load("ok") == "text"
