"""
Grounding
=========
Pure detectors behind the "never guess" enforcement: decide whether a user
utterance asks for live data (and which category), and whether an assistant
reply merely *promises* to check something instead of doing it.

Both engines use these after a turn ends with zero tool calls: a hit triggers
one forced-tool retry (see LangChainEngine._grounding_retry and
LLMEngine._generate_native). Conservative by design — casual chat must never
be forced into tool calls, so a category fires only when an intent cue AND a
topic keyword both match.

Pure module: no config, no I/O — unit-testable like speech_text.
"""

import re
from typing import Dict, Optional, Tuple

# A live-data category fires only when the utterance also looks like a
# question/request. Guards against topical small talk ("nice weather we're
# having") triggering tools.
_INTENT_CUE = re.compile(
    r"(?:^(?:what|whats|what's|when|whens|when's|where|how|is|are|was|were|will|"
    r"any|tell me|give me|can you|could you|would you|do you|did|check|look|"
    r"search|find|show me|read me)\b"
    r"|\?"
    r"|\b(?:check|look up|pull up|search|find out|look into)\b)",
    re.IGNORECASE)

# Category -> topic keywords. Order matters: first match wins, so specific
# categories (ALERTS, QUAKES) come before broad ones (WEATHER, WEB_SEARCH).
_CATEGORY_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("ALERTS", re.compile(
        r"\b(?:alerts?|warnings?|advisor(?:y|ies)|tornado|flood(?:ing)?|"
        r"storm watch|weather watch)\b", re.IGNORECASE)),
    ("QUAKES", re.compile(
        r"\b(?:earthquakes?|quakes?|seismic|tremors?|aftershocks?)\b",
        re.IGNORECASE)),
    ("KP_INDEX", re.compile(
        r"\b(?:aurora|northern lights|kp[ -]?index|geomagnetic|solar storm|"
        r"space weather)\b", re.IGNORECASE)),
    ("WEATHER", re.compile(
        r"\b(?:weather|forecast|temperature|rain(?:ing)?|snow(?:ing)?|windy?|"
        r"humid(?:ity)?|sunny|cloudy|degrees (?:outside|out there)|"
        r"hot outside|cold outside)\b", re.IGNORECASE)),
    ("DATETIME", re.compile(
        r"(?:\btime is it\b|\bthe time\b|\btoday'?s date\b|\bwhat day\b|"
        r"\bdate today\b|\bday of the week\b|\bcurrent time\b)", re.IGNORECASE)),
    ("GPU_STATUS", re.compile(
        r"\b(?:gpu|graphics card|vram|video memory|cuda)\b", re.IGNORECASE)),
    ("WEB_SEARCH", re.compile(
        r"\b(?:search|look up|google|latest news|news about|current price|"
        r"price of|who won|score of|game score)\b", re.IGNORECASE)),
)

# Category -> the tool(s) that can answer it. The engine skips the retry when
# none of these are enabled (e.g. WEB_SEARCH without SearxNG configured).
CATEGORY_TOOLS: Dict[str, Tuple[str, ...]] = {
    "ALERTS": ("ALERTS",),
    "QUAKES": ("QUAKES",),
    "KP_INDEX": ("KP_INDEX",),
    "WEATHER": ("WEATHER", "FORECAST"),
    "DATETIME": ("DATETIME",),
    "GPU_STATUS": ("GPU_STATUS",),
    "WEB_SEARCH": ("WEB_SEARCH",),
    # PROMISED_ACTION: the model already declared it needs a tool; any bound
    # tool qualifies, so the engine treats it as always-eligible.
    "PROMISED_ACTION": (),
}

# The forced-retry instruction, shared by every engine path (native, agent,
# text-marker) so the wording can't drift between them.
NUDGE = ("You must answer the caller's last question using your tools. "
         "Call the right tool now; do not answer from memory and do not "
         "promise to check later.")

# Reply phrases that promise an action instead of performing it. The turn is
# over when the caller hears this — there is no "later".
#
# "let me get that" is spelled out rather than folded into the verb list: a
# bare "get" would also swallow "let me get this straight", which is
# conversational, not a promise.
_PROMISE = re.compile(
    r"(?:\blet me (?:go )?(?:check|look|pull|find|see|verify|search|grab)\b"
    r"|\blet me get (?:that|it|those|them|you)\b"
    r"|\bone (?:moment|second|sec)\b"
    r"|\bjust a (?:moment|second|sec|minute)\b"
    r"|\bhang on\b|\bhold on\b"
    r"|\bi(?:'ll| will) (?:go )?(?:check|look|find|get|pull|grab|search)\b"
    r"|\bchecking (?:that|on that|now)\b"
    r"|\b(?:looking|searching) (?:that|it) up\b"
    r"|\bgive me a (?:moment|second|sec|minute)\b)",
    re.IGNORECASE)

# Sentence splitter for the trailing-promise check below.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _last_sentence(reply: str) -> str:
    parts = [p for p in _SENTENCE_END.split((reply or "").strip()) if p.strip()]
    return parts[-1] if parts else ""


def live_data_category(text: str) -> Optional[str]:
    """The live-data category a user utterance asks about, or None.

    Requires both an intent cue (question/imperative shape) and a topic
    keyword, so topical small talk never forces a tool call.
    """
    if not text or not _INTENT_CUE.search(text):
        return None
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return None


def promised_action(reply: str) -> bool:
    """True when the assistant's reply promises to check/look something up
    (which, at end of turn, means dead air — the promise can never be kept)."""
    return bool(reply and _PROMISE.search(reply))


def trailing_promise(reply: str) -> bool:
    """True when the reply SIGNS OFF with a promise — the last thing the caller
    hears is "let me get that for you", and then the turn ends.

    This is the check for turns that DID call a tool. promised_action() is too
    broad there: a reply can legitimately mention a check it already performed
    ("Let me check... it's 71 degrees"), and re-running that turn would just
    burn a second round-trip. A promise in the FINAL sentence is different —
    nothing follows it, so nothing was delivered, and the caller is left
    listening to silence. Seen in the wild: a recipe question that called the
    wrong tool, got nothing back, and closed with "Let me get that for you
    right away" — followed by seventeen seconds of dead air.
    """
    return promised_action(_last_sentence(reply))


def grounding_category(user_text: str, reply: str) -> Optional[str]:
    """Category for a zero-tool turn that needs a forced retry, or None."""
    return live_data_category(user_text) or (
        "PROMISED_ACTION" if promised_action(reply) else None)
