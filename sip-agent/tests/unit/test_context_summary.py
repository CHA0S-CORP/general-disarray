"""Unit tests for the rolling conversation summary (context_manager)."""
import asyncio
from types import SimpleNamespace

import pytest

import context_manager
from call_session import CallSession

pytestmark = pytest.mark.unit


class _StubEngine:
    def __init__(self, reply="SUMMARY OF THE CALL"):
        self.reply = reply
        self.calls = []

    async def summarize_text(self, system_prompt, text, timeout_s):
        self.calls.append(text)
        return self.reply


def _assistant(config, engine):
    return SimpleNamespace(config=config, llm_engine=engine, _event_tasks=set())


def _session(n_messages):
    session = CallSession(call_info=None, direction="inbound",
                          transcript_id="t-1")
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        session.conversation_history.append(
            {"role": role, "content": f"message {i}"})
    return session


async def _drain(assistant):
    while assistant._event_tasks:
        await asyncio.gather(*list(assistant._event_tasks))


async def test_below_threshold_schedules_nothing(config_factory):
    cfg = config_factory(max_conversation_turns="10")
    engine = _StubEngine()
    assistant = _assistant(cfg, engine)
    session = _session(20)  # exactly the window, no overflow

    context_manager.maybe_schedule_summary(assistant, session)
    await _drain(assistant)
    assert engine.calls == []
    assert session.rolling_summary == ""


async def test_overflow_is_folded_into_summary(config_factory):
    cfg = config_factory(max_conversation_turns="2")
    engine = _StubEngine()
    assistant = _assistant(cfg, engine)
    session = _session(10)  # window is 4 -> 6 messages overflow

    context_manager.maybe_schedule_summary(assistant, session)
    await _drain(assistant)

    assert session.rolling_summary == "SUMMARY OF THE CALL"
    assert session.summarized_upto == 6
    assert not session.summary_task_running
    # The overflow messages (0..5) went to the summarizer; recent ones didn't.
    assert "message 5" in engine.calls[0]
    assert "message 6" not in engine.calls[0]


async def test_second_pass_folds_previous_summary(config_factory):
    cfg = config_factory(max_conversation_turns="2")
    engine = _StubEngine()
    assistant = _assistant(cfg, engine)
    session = _session(10)

    context_manager.maybe_schedule_summary(assistant, session)
    await _drain(assistant)

    session.conversation_history.extend(
        {"role": "user", "content": f"late {i}"} for i in range(4))
    context_manager.maybe_schedule_summary(assistant, session)
    await _drain(assistant)

    assert session.summarized_upto == 10
    assert "Previous summary:" in engine.calls[1]


async def test_single_flight(config_factory):
    cfg = config_factory(max_conversation_turns="2")
    engine = _StubEngine()
    assistant = _assistant(cfg, engine)
    session = _session(10)

    context_manager.maybe_schedule_summary(assistant, session)
    context_manager.maybe_schedule_summary(assistant, session)  # while running
    await _drain(assistant)
    assert len(engine.calls) == 1


async def test_failed_summary_is_fail_open(config_factory):
    cfg = config_factory(max_conversation_turns="2")
    engine = _StubEngine(reply=None)
    assistant = _assistant(cfg, engine)
    session = _session(10)

    context_manager.maybe_schedule_summary(assistant, session)
    await _drain(assistant)

    assert session.rolling_summary == ""
    assert session.summarized_upto == 0
    assert not session.summary_task_running  # retryable next turn


async def test_disabled_by_config(config_factory):
    cfg = config_factory(max_conversation_turns="2",
                         conversation_summary_enabled="false")
    engine = _StubEngine()
    assistant = _assistant(cfg, engine)
    session = _session(10)

    context_manager.maybe_schedule_summary(assistant, session)
    await _drain(assistant)
    assert engine.calls == []