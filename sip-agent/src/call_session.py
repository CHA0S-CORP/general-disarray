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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    # The RTP read/VAD loop driving this session.
    audio_loop_task: Optional[asyncio.Task] = None
    start_time: float = field(default_factory=time.time)
    # True once a call.ended event webhook has been dispatched for this session
    # (the audio-loop tail and _teardown_session can both reach the emit site).
    ended_event_emitted: bool = False
    # Rolling summary of turns that no longer fit the LLM history window, and
    # the index into conversation_history up to which they've been folded in.
    rolling_summary: str = ""
    summarized_upto: int = 0
    # Single-flight guard for the background summarization task.
    summary_task_running: bool = False
    # Cross-call memory about this caller, formatted for the system prompt
    # (loaded once at call start; empty when unknown caller or disabled).
    caller_memory_prompt: str = ""
    # True once the post-call memory update has been dispatched (teardown and
    # the audio-loop tail can both reach the update site).
    memory_update_started: bool = False
    # Generic per-call scratch for stateful tools (e.g. a trivia game), keyed
    # by tool. Tool instances are singletons across calls — never stash call
    # state on the tool itself.
    tool_state: Dict[str, Any] = field(default_factory=dict)
