"""Unit tests for the cross-call caller memory store."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from caller_memory import CallerMemoryStore, caller_id_from_uri

pytestmark = pytest.mark.unit


# --- caller id extraction ------------------------------------------------------

def test_caller_id_from_plain_uri():
    assert caller_id_from_uri("sip:1001@pbx.lan") == "1001"


def test_caller_id_from_display_name_uri():
    assert caller_id_from_uri('"Bob" <sip:1001@pbx.lan>') == "1001"


def test_caller_id_rejects_path_traversal():
    assert caller_id_from_uri("../../etc/passwd") is None
    assert caller_id_from_uri("sip:../evil@pbx") is None


def test_caller_id_empty():
    assert caller_id_from_uri("") is None
    assert caller_id_from_uri(None) is None


# --- store ----------------------------------------------------------------------

class _StubEngine:
    """summarize_text stub returning a scripted extraction."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def summarize_text(self, system_prompt, text, timeout_s):
        self.calls.append(text)
        return self.reply


def _transcript(*user_lines):
    turns = []
    for line in user_lines:
        turns.append({"role": "user", "content": line, "ts": "t"})
        turns.append({"role": "assistant", "content": "ok", "ts": "t"})
    return {"call_id": "c1", "turns": turns}


@pytest.fixture
def store(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path))
    return CallerMemoryStore(cfg)


async def test_update_and_recall_round_trip(store):
    engine = _StubEngine(json.dumps({
        "facts": ["Name is Bob", "Prefers morning callbacks"],
        "last_call_summary": "Asked about the weather.",
    }))
    await store.update_from_call("sip:1001@pbx", _transcript("my name is Bob"),
                                 engine)

    record = store.get("1001")
    assert record["facts"] == ["Name is Bob", "Prefers morning callbacks"]
    assert record["call_count"] == 1

    prompt = store.format_for_prompt("1001")
    assert "- Name is Bob" in prompt
    assert "Last call: Asked about the weather." in prompt


async def test_update_increments_call_count_and_feeds_existing_facts(store):
    first = _StubEngine(json.dumps({"facts": ["Name is Bob"],
                                    "last_call_summary": "s1"}))
    await store.update_from_call("sip:1001@pbx", _transcript("hi"), first)

    second = _StubEngine(json.dumps({"facts": ["Name is Bob", "Has a dog"],
                                     "last_call_summary": "s2"}))
    await store.update_from_call("sip:1001@pbx", _transcript("my dog"), second)

    assert "Name is Bob" in second.calls[0]  # existing facts fed to the LLM
    record = store.get("1001")
    assert record["call_count"] == 2
    assert "Has a dog" in record["facts"]


async def test_unparseable_extraction_keeps_old_memory(store):
    good = _StubEngine(json.dumps({"facts": ["Name is Bob"],
                                   "last_call_summary": "s"}))
    await store.update_from_call("sip:1001@pbx", _transcript("hi"), good)

    bad = _StubEngine("I could not comply, here is prose instead.")
    await store.update_from_call("sip:1001@pbx", _transcript("hi again"), bad)

    assert store.get("1001")["facts"] == ["Name is Bob"]


async def test_extraction_json_inside_prose_is_tolerated(store):
    engine = _StubEngine(
        'Sure! Here you go:\n{"facts": ["Likes jazz"], "last_call_summary": "s"}')
    await store.update_from_call("sip:1001@pbx", _transcript("jazz"), engine)
    assert store.get("1001")["facts"] == ["Likes jazz"]


async def test_no_user_speech_skips_update(store):
    engine = _StubEngine("should never be called")
    await store.update_from_call(
        "sip:1001@pbx",
        {"turns": [{"role": "assistant", "content": "hello?"}]},
        engine)
    assert engine.calls == []
    assert store.get("1001") is None


async def test_engine_failure_is_fail_open(store):
    await store.update_from_call("sip:1001@pbx", _transcript("hi"),
                                 _StubEngine(None))
    assert store.get("1001") is None


def test_corrupt_file_reads_as_no_memory(store, config_factory, tmp_path):
    (tmp_path / "caller_memory" / "1001.json").write_text("{not json")
    assert store.get("1001") is None
    assert store.format_for_prompt("1001") == ""


def test_format_for_prompt_bounds_facts(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), caller_memory_max_facts="2")
    store = CallerMemoryStore(cfg)
    store._write_atomic("1001", {
        "caller": "1001", "call_count": 1,
        "facts": ["one", "two", "three", "four"],
        "last_call_summary": "",
    })
    prompt = store.format_for_prompt("1001")
    assert "- one" in prompt and "- two" in prompt
    assert "three" not in prompt


def test_format_for_prompt_bounds_chars(config_factory, tmp_path):
    cfg = config_factory(data_dir=str(tmp_path), caller_memory_max_chars="40")
    store = CallerMemoryStore(cfg)
    store._write_atomic("1001", {
        "caller": "1001", "call_count": 1,
        "facts": ["a" * 30, "b" * 30, "c" * 30],
        "last_call_summary": "",
    })
    assert len(store.format_for_prompt("1001")) <= 40

# --- concurrency: lost-update protection --------------------------------------

class _SlowEngine:
    """Extraction that yields control, letting another writer interleave."""

    def __init__(self, reply, on_await=None):
        self.reply = reply
        self.on_await = on_await

    async def summarize_text(self, system_prompt, text, timeout_s):
        await asyncio.sleep(0)  # hand the event loop to the racing writer
        if self.on_await:
            self.on_await()
        await asyncio.sleep(0)
        return self.reply


async def test_remember_during_extraction_is_not_clobbered(store):
    """Regression: a REMEMBER made on a redial must survive the previous
    call's post-call fact extraction completing afterwards."""
    store._write_atomic("1001", {
        "caller": "1001", "call_count": 1,
        "facts": ["Name is Bob"], "last_call_summary": "s0",
    })

    def redial_remember():
        store.add_fact("1001", "Address is 5 Elm Street")

    engine = _SlowEngine(
        json.dumps({"facts": ["Name is Bob", "Likes coffee"],
                    "last_call_summary": "s1"}),
        on_await=redial_remember)

    await store.update_from_call("sip:1001@pbx", _transcript("hi"), engine)

    facts = store.get("1001")["facts"]
    assert "Address is 5 Elm Street" in facts, "REMEMBER was clobbered"
    assert "Likes coffee" in facts, "extracted fact lost"


async def test_forget_during_extraction_is_respected(store):
    """A FORGET during the extraction window must not be undone by the
    extraction writing the old fact back."""
    store._write_atomic("1001", {
        "caller": "1001", "call_count": 1,
        "facts": ["Name is Bob", "Hates jazz"], "last_call_summary": "s0",
    })

    engine = _SlowEngine(
        json.dumps({"facts": ["Name is Bob", "Hates jazz"],
                    "last_call_summary": "s1"}),
        on_await=lambda: store.remove_facts("1001", "jazz"))

    await store.update_from_call("sip:1001@pbx", _transcript("hi"), engine)

    facts = store.get("1001")["facts"]
    assert not any("jazz" in f.lower() for f in facts), "FORGET was undone"


async def test_concurrent_updates_for_same_caller_serialize(store):
    """Two overlapping post-call extractions must not both write from the
    same stale snapshot (call_count would undercount)."""
    engine_a = _SlowEngine(json.dumps({"facts": ["A"], "last_call_summary": "a"}))
    engine_b = _SlowEngine(json.dumps({"facts": ["B"], "last_call_summary": "b"}))

    await asyncio.gather(
        store.update_from_call("sip:1001@pbx", _transcript("one"), engine_a),
        store.update_from_call("sip:1001@pbx", _transcript("two"), engine_b),
    )

    assert store.get("1001")["call_count"] == 2


def test_extraction_prompt_excludes_persona_and_oneoffs():
    """Regression: the extraction prompt must tell the model NOT to remember
    persona/speaking-style choices or one-off questions, or those bleed into
    every future call and mix (observed live: 'Pigleton'/Pig-Latin facts
    injected into unrelated calls)."""
    from caller_memory import _FACT_EXTRACTION_PROMPT
    p = _FACT_EXTRACTION_PROMPT.lower()
    assert "persona" in p and "pig latin" in p, "must name persona/style exclusion"
    assert "carry over" in p or "only to the call" in p
    assert "one-off" in p or "not what they" in p
    # The one-line summary is ALSO injected into the next call, so it must be
    # told not to mention the persona either (else the style leaks via the
    # summary even when it's kept out of the facts).
    assert "summary" in p
    summary_guidance = p.split("summary of this latest call", 1)[-1]
    assert "do not mention the persona" in summary_guidance
