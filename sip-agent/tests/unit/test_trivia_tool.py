"""Unit tests for the TRIVIA tool plugin: game flow, fuzzy answer matching,
per-call state on the session, and the trivia.json bank override."""
import json
from types import SimpleNamespace

import pytest

from call_session import CallSession
from plugins.trivia_tool import (
    QUESTION_BANK,
    TriviaTool,
    _is_correct,
    _normalize_answer,
)
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def make_assistant(config=None):
    return SimpleNamespace(
        config=config,
        session=CallSession(call_info=None, direction="inbound", transcript_id="t1"),
    )


# --- pure helpers ------------------------------------------------------------

def test_normalize_answer():
    assert _normalize_answer("  The PARIS!  ") == "the paris"
    assert _normalize_answer("forty-two, obviously") == "forty two obviously"
    assert _normalize_answer("") == ""


def test_fuzzy_matching():
    # substring in either direction, punctuation/case insensitive
    assert _is_correct("The Paris", ["paris"])
    assert _is_correct("paris", ["the city of paris"])
    assert _is_correct("It's 8!", ["eight", "8"])
    assert not _is_correct("london", ["paris"])
    assert not _is_correct("", ["paris"])


# --- game flow ----------------------------------------------------------------

async def test_ask_answer_score_flow():
    tool = TriviaTool(make_assistant())

    result = await tool.execute({"action": "ask"})
    assert result.status == ToolStatus.SUCCESS
    state = tool.assistant.session.tool_state["trivia"]
    assert state["current"] is not None
    assert result.message == QUESTION_BANK[state["current"]]["question"]

    correct_answer = QUESTION_BANK[state["current"]]["answers"][0]
    result = await tool.execute({"action": "answer", "answer": correct_answer})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["correct"] is True
    assert result.data["score"] == 1
    assert result.data["rounds"] == 1
    assert "That is right" in result.message
    assert state["current"] is None

    result = await tool.execute({"action": "score"})
    assert result.status == ToolStatus.SUCCESS
    assert result.data == {"score": 1, "rounds": 1}


async def test_wrong_answer():
    tool = TriviaTool(make_assistant())
    await tool.execute({"action": "ask"})

    result = await tool.execute({"action": "answer", "answer": "xyzzy flurble"})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["correct"] is False
    assert result.data["score"] == 0
    assert result.data["rounds"] == 1
    assert "Not quite" in result.message


async def test_answer_before_ask_fails():
    tool = TriviaTool(make_assistant())
    result = await tool.execute({"action": "answer", "answer": "paris"})
    assert result.status == ToolStatus.FAILED
    assert "Ask me for a question first" in result.message


async def test_no_session_fails():
    result = await TriviaTool(assistant=None).execute({"action": "ask"})
    assert result.status == ToolStatus.FAILED
    assert "during a call" in result.message

    idle = SimpleNamespace(config=None, session=None)  # between calls
    result = await TriviaTool(idle).execute({"action": "score"})
    assert result.status == ToolStatus.FAILED


async def test_bank_override_from_data_dir(tmp_path, config_factory):
    override = [{"question": "What is the answer to everything?",
                 "answers": ["42", "forty two"]}]
    (tmp_path / "trivia.json").write_text(json.dumps(override))

    cfg = config_factory(data_dir=str(tmp_path))
    tool = TriviaTool(make_assistant(cfg))
    assert len(tool._bank) == 1

    result = await tool.execute({"action": "ask"})
    assert result.message == "What is the answer to everything?"

    result = await tool.execute({"action": "answer", "answer": "I'd say forty two"})
    assert result.data["correct"] is True


async def test_malformed_override_falls_back(tmp_path, config_factory):
    (tmp_path / "trivia.json").write_text("{not json")
    cfg = config_factory(data_dir=str(tmp_path))
    tool = TriviaTool(make_assistant(cfg))
    assert tool._bank == QUESTION_BANK


async def test_exhausting_bank_resets_asked(tmp_path, config_factory):
    override = [{"question": "Only question?", "answers": ["yes"]},
                {"question": "Second question?", "answers": ["no"]}]
    (tmp_path / "trivia.json").write_text(json.dumps(override))
    cfg = config_factory(data_dir=str(tmp_path))
    tool = TriviaTool(make_assistant(cfg))
    state_key = "trivia"

    await tool.execute({"action": "ask"})
    await tool.execute({"action": "ask"})
    state = tool.assistant.session.tool_state[state_key]
    assert sorted(state["asked"]) == [0, 1]

    # Third ask: bank exhausted -> asked resets and a question repeats.
    result = await tool.execute({"action": "ask"})
    assert result.status == ToolStatus.SUCCESS
    assert state["asked"] == [state["current"]]
