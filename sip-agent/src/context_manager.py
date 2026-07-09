"""
Context Manager
===============
Rolling conversation summary: when a call outgrows the LLM history window
(max_conversation_turns), fold the overflow turns into a running summary
instead of silently dropping them. The summary is injected into the system
prompt ("Conversation so far") while recent turns stay verbatim.

Runs as a background task after the reply is spoken — never on the speaking
path — and is fail-open: on any error the session keeps its previous summary
and the engine falls back to the old tail-slice behavior.
"""

import asyncio
import logging

from call_session import CallSession
from logging_utils import log_event

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You maintain a running summary of an ongoing phone call for the assistant on the call.
Merge the previous summary (if any) with the new conversation turns into ONE updated summary.
Keep every fact that could matter later: names, numbers, requests, decisions, preferences, open questions, things the assistant promised to do.
Write compact plain prose, at most 150 words. Output ONLY the summary."""


def _overflow(session: CallSession, max_turns: int) -> int:
    """Index up to which history should be folded into the summary.

    Keeps the most recent max_turns*2 messages verbatim; everything before
    that (and after the last summarized point) is overflow.
    """
    return max(len(session.conversation_history) - max_turns * 2, 0)


def maybe_schedule_summary(assistant, session: CallSession) -> None:
    """Kick off a background summarization when the history has outgrown the
    window. Single-flight per session; safe to call every turn."""
    config = assistant.config
    if not config.summary_enabled or session.summary_task_running:
        return
    if _overflow(session, config.max_conversation_turns) <= session.summarized_upto:
        return
    session.summary_task_running = True
    task = asyncio.create_task(_summarize(assistant, session))
    # Keep a strong reference (mirrors the call-event webhook tasks).
    assistant._event_tasks.add(task)
    task.add_done_callback(assistant._event_tasks.discard)


async def _summarize(assistant, session: CallSession) -> None:
    try:
        config = assistant.config
        # Snapshot the range now; new turns appended while the LLM runs are
        # untouched and picked up by a later pass.
        upto = _overflow(session, config.max_conversation_turns)
        chunk = session.conversation_history[session.summarized_upto:upto]
        if not chunk:
            return

        lines = []
        if session.rolling_summary:
            lines.append(f"Previous summary:\n{session.rolling_summary}\n")
        lines.append("New turns:")
        for msg in chunk:
            role = "Caller" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")

        summary = await assistant.llm_engine.summarize_text(
            SUMMARY_SYSTEM_PROMPT, "\n".join(lines), config.summary_timeout_s)
        if not summary:
            return  # fail-open: keep the old summary, retry next turn

        session.rolling_summary = summary
        session.summarized_upto = upto
        log_event(logger, logging.INFO,
                  f"Conversation summary updated ({upto} msgs folded, "
                  f"{len(summary)} chars)",
                  event="conversation_summary", outcome="ok",
                  messages_folded=upto, chars=len(summary))
    except Exception as e:
        logger.warning(f"Conversation summarization failed: {e}")
    finally:
        session.summary_task_running = False
