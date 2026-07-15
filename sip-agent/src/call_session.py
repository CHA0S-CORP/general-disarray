"""
Call Session
============
Per-call state container for the assistant's live call.

All state owned by one call — conversation history, the in-flight response
turn, held transcripts, the audio loop — lives here rather than on the
assistant. A stale task from a replaced call therefore writes into its own
dead session instead of corrupting the next call's conversation.
"""

import asyncio
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# Below this played fraction, the chunk that was playing when the caller
# barged in is treated as not heard: a sentence cut off in its first third
# carried essentially no information, so it stays out of the reconstructed
# "what the caller actually heard" text. Deliberately a module constant, not
# an env var — it's a perceptual threshold, not deployment configuration.
MIN_HEARD_FRACTION = 0.3


@dataclass
class TurnLedger:
    """Playback ledger for one response turn: tag -> sentence-chunk text.

    Every audio chunk enqueued while speaking a response allocates a unique
    tag (CallSession.next_playback_tag) and records its text here, in enqueue
    order (dict insertion order). After a barge-in, the ledger plus the
    playlist player's snapshot() reconstruct the prefix the caller actually
    heard.
    """

    entries: Dict[int, str] = field(default_factory=dict)


def spoken_text(ledger: Optional[TurnLedger],
                completed_tags: Iterable[int],
                current_tag: Optional[int],
                fraction: float,
                min_fraction: float = MIN_HEARD_FRACTION) -> str:
    """Reconstruct the text the caller actually heard from a TurnLedger and a
    PlaylistPlayer.snapshot() reading.

    Includes every ledger entry whose tag completed playback (in enqueue
    order), plus the currently-playing entry only if at least ``min_fraction``
    of it played. Pure function — unit-testable without any player.
    """
    if ledger is None:
        return ""
    completed = set(completed_tags)
    parts: List[str] = []
    for tag, chunk in ledger.entries.items():
        if tag in completed:
            parts.append(chunk)
        elif current_tag is not None and tag == current_tag and fraction >= min_fraction:
            parts.append(chunk)
    return " ".join(parts)


@dataclass
class CallSession:
    """State owned by a single live call."""

    call_info: Any
    direction: str          # "inbound" | "outbound"
    transcript_id: str
    conversation_history: List[Dict] = field(default_factory=list)
    # True while a response turn is generating/speaking (drives barge-in).
    processing: bool = False
    # The in-flight response turn (ack + LLM + TTS), cancellable on barge-in.
    turn_task: Optional[asyncio.Task] = None
    # Transcript that arrived while a turn was in flight; dispatched next.
    pending_transcription: Optional[str] = None
    # Speculative endpointing (ENDPOINT_MODE=speculative): a transcript
    # fragment that didn't look complete yet, held so follow-up speech can
    # merge into one utterance, plus the monotonic time it was held at
    # (drives the ENDPOINT_MAX_SILENCE_MS dispatch deadline).
    held_fragment: Optional[str] = None
    held_fragment_at: float = 0.0
    # The text the in-flight turn was dispatched with (speculative mode
    # only): lets the cancel-merge branch rebuild the utterance when the
    # caller resumes speaking before the assistant is audibly speaking.
    speculative_turn_text: Optional[str] = None
    # The RTP read/VAD loop driving this session.
    audio_loop_task: Optional[asyncio.Task] = None
    start_time: float = field(default_factory=time.time)
    # True once a call.ended event webhook has been dispatched for this session
    # (the audio-loop tail and _teardown_session can both reach the emit site).
    ended_event_emitted: bool = False
    # Same once-only guard for the admin event bus's call.ended (independent
    # of the webhook flag, which is only set when a webhook URL is configured).
    admin_ended_published: bool = False
    # Rolling summary of turns that no longer fit the LLM history window, and
    # the index into conversation_history up to which they've been folded in.
    rolling_summary: str = ""
    summarized_upto: int = 0
    # Single-flight guard for the background summarization task.
    summary_task_running: bool = False
    # Memory key for this caller (user part of the SIP URI); empty when the
    # caller couldn't be identified or memory is disabled.
    caller_id: str = ""
    # Cross-call memory about this caller, formatted for the system prompt.
    # Loaded at call start and refreshed each turn, so mid-call REMEMBERs and
    # a just-finished extraction from the previous call become visible.
    caller_memory_prompt: str = ""
    # True once the post-call memory update has been dispatched (teardown and
    # the audio-loop tail can both reach the update site).
    memory_update_started: bool = False
    # Generic per-call scratch for stateful tools (e.g. a trivia game), keyed
    # by tool. Tool instances are singletons across calls — never stash call
    # state on the tool itself.
    tool_state: Dict[str, Any] = field(default_factory=dict)
    # The VirtualNumber entry this inbound call was matched to (None for
    # normal calls), and the once-only guard for its finalization webhook +
    # single-use consumption (teardown and the audio-loop tail both reach it).
    virtual_number: Optional[Any] = None
    virtual_number_finalized: bool = False
    # This call's audio-pipeline state (VAD + utterance buffer + latency
    # metrics; a SessionAudioState from audio_pipeline.new_session_state()).
    # Typed loosely so this module stays dependency-light.
    audio_state: Optional[Any] = None
    # Monotonically increasing playback-tag allocator (unique per call).
    # Only touched from the asyncio side (the speaking path).
    next_playback_tag: int = 1
    # Ledger of the in-flight response's playback (None outside a response).
    active_ledger: Optional[TurnLedger] = None


# ============================================================================
# Which call is "current": a ContextVar
# ============================================================================
# With several concurrent calls, "the" session is a property of the task tree
# acting on a call's behalf, not of the assistant. Every task that works for
# exactly one call (a response turn, the audio loop, an outbound interactive
# session, a shutdown-drain goodbye) binds its session here, and everything
# downstream — tools reaching for assistant.session/current_call, playback,
# the CALLBACK caller default — resolves the right call with no signature
# changes.
#
# IMPORTANT: asyncio tasks snapshot their context at create_task() time, so
# the variable must be set INSIDE the task body (its first statements), never
# after the task has been spawned — a later set() in the parent does not
# propagate into the child.
current_session: "contextvars.ContextVar[Optional[CallSession]]" = (
    contextvars.ContextVar("current_call_session", default=None))


def set_current_session(session: Optional[CallSession]) -> "contextvars.Token":
    """Bind ``session`` as the call the current task (tree) acts on behalf of.

    Call this at the TOP of a task body; tasks spawned afterwards inherit it.
    Returns the token for an optional contextvars reset."""
    return current_session.set(session)


def get_current_session() -> Optional[CallSession]:
    """The session the current task acts on behalf of, or None if unbound."""
    return current_session.get()
