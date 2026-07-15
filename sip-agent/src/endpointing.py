"""
Endpointing Heuristics
======================
Pure text/timing heuristics behind ENDPOINT_MODE (config.endpoint_mode):
given what the caller has said so far (when a transcript is available) and
how long they spoke, suggest how much trailing silence should end the
utterance, and judge whether a transcript fragment reads as a finished
thought.

No I/O, no config, no state — fully unit-testable. The VAD consumes
suggest_timeout_ms() per chunk (adaptive mode); main.py's speculative
dispatch/hold/merge logic consumes looks_complete().
"""

import re
from typing import List, Optional

# Speech-length scaling anchors for the no-transcript case: utterances
# shorter than SHORT_SPEECH_MS get the full max hangover (the caller may
# just be winding up); utterances longer than LONG_SPEECH_MS lean back to
# the configured base timeout.
SHORT_SPEECH_MS = 1000.0
LONG_SPEECH_MS = 4000.0

# Utterances that are complete answers on their own, even without terminal
# punctuation ("yes", "nope", "that's all", ...).
_SHORT_ANSWERS = frozenset({
    "yes", "yeah", "yep", "yup", "no", "nope", "nah",
    "correct", "okay", "ok", "right", "sure", "fine", "done", "exactly",
    "that's all", "that's it", "that is all", "that is it",
    "nothing else", "no thanks", "no thank you",
    "goodbye", "bye", "bye bye", "thanks", "thank you",
})

# A trailing one of these means the caller is mid-thought.
_CONTINUATION_WORDS = frozenset({
    # conjunctions
    "and", "but", "or", "so", "because",
    # fillers
    "um", "uh", "er", "hmm", "erm", "umm", "uhh",
    # dangling prepositions / articles / possessives
    "to", "of", "in", "the", "a", "an", "my",
    "for", "at", "with", "from", "about",
    "your", "his", "her", "their", "our",
    # dangling copulas/auxiliaries ("and the rest is")
    "is", "are", "was", "were", "am",
})

# An unpunctuated fragment with at least this many words (and no continuation
# cue) reads as a finished thought: small Whisper models on telephone audio
# routinely omit terminal punctuation, and holding every plain sentence to the
# max-silence deadline would make speculative mode slower than fixed mode.
_MIN_COMPLETE_WORDS = 3

_DIGIT_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "oh",
})

_TERMINAL_PUNCT = ".!?"


def _normalize(text: str) -> str:
    """Strip surrounding whitespace and any trailing ellipsis (an STT artifact
    of a pause, not terminal punctuation)."""
    t = (text or "").strip()
    while True:
        if t.endswith("…"):
            t = t[:-1].rstrip()
        elif t.endswith("..."):
            t = t[:-3].rstrip()
        else:
            return t


def _core(t: str) -> str:
    """Lowercased text with trailing punctuation removed, for word checks."""
    return t.rstrip(_TERMINAL_PUNCT + ",;:").rstrip().lower()


def _digit_groups(core: str) -> List[str]:
    return re.findall(r"\d+", core)


def _mid_recitation(core: str) -> bool:
    """True when the text ends with a digit group shorter than the preceding
    ones — the caller is mid-way through reciting a number ("555 12")."""
    if not core or not core[-1].isdigit():
        return False
    groups = _digit_groups(core)
    if len(groups) < 2:
        return False
    return len(groups[-1]) < max(len(g) for g in groups[:-1])


def _bare_number(core: str) -> bool:
    """True for a bare number / digit-string answer ("42", "555 1234",
    "five five five")."""
    if not core:
        return False
    if any(ch.isdigit() for ch in core) and re.fullmatch(r"[\d\s\-,.()+]+", core):
        return True
    words = core.replace(",", " ").split()
    return bool(words) and all(w in _DIGIT_WORDS for w in words)


def looks_complete(text: str) -> bool:
    """Does this transcript fragment read as a finished thought?

    True: ends with terminal punctuation, is a known complete short answer
    ("yes", "that's all", ...), is a bare number/digit-string answer, or is
    an ordinary unpunctuated sentence of some substance (STT often omits
    terminal punctuation on telephone audio — see _MIN_COMPLETE_WORDS).
    False: ends with a continuation cue — a conjunction/filler/dangling
    preposition or article, a trailing comma, or a digit group (numeric or
    spelled out) that looks mid-recitation. Case-insensitive; trailing
    whitespace/ellipsis stripped first. Short ambiguous fragments are
    treated as incomplete (the safe side: wait a little longer rather than
    cut the caller off).
    """
    t = _normalize(text)
    if not t:
        return False
    if t.endswith(","):
        return False
    core = _core(t)
    words = core.split()
    last_word = words[-1] if words else ""
    # Continuation cues win even over terminal punctuation: Whisper happily
    # emits "Um." or "And..." for a thinking pause.
    if last_word in _CONTINUATION_WORDS:
        return False
    if _mid_recitation(core):
        return False
    if t[-1] in _TERMINAL_PUNCT:
        return True
    if core in _SHORT_ANSWERS:
        return True
    if _bare_number(core):
        return True
    # A trailing spelled-out digit inside a longer sentence reads as
    # mid-recitation ("my number is five five five"): hold for the rest.
    if last_word in _DIGIT_WORDS:
        return False
    # Plain unpunctuated sentence, no continuation cue: complete once it has
    # some substance ("turn off the kitchen lights"). Very short non-listed
    # fragments ("the weather") stay held.
    return len(words) >= _MIN_COMPLETE_WORDS


def suggest_timeout_ms(partial_text: Optional[str], speech_ms: float,
                       base_ms: int, min_ms: int, max_ms: int) -> int:
    """Suggest the end-of-utterance silence timeout for the current chunk.

    With a transcript: a complete-looking fragment clamps to ``min_ms``
    (answer fast), an incomplete-looking one to ``max_ms`` (let the caller
    finish). Without one (no interim transcripts pre-commit), scale by how
    long the caller has been speaking: under ~1s of speech leans to the max
    hangover, over ~4s leans back to ``base_ms``. Always within
    [min_ms, max_ms].
    """
    if max_ms < min_ms:
        max_ms = min_ms

    def clamp(value: float) -> int:
        return int(min(max(value, min_ms), max_ms))

    text = (partial_text or "").strip()
    if text:
        return clamp(min_ms if looks_complete(text) else max_ms)

    if speech_ms <= SHORT_SPEECH_MS:
        return clamp(max_ms)
    if speech_ms >= LONG_SPEECH_MS:
        return clamp(base_ms)
    fraction = (speech_ms - SHORT_SPEECH_MS) / (LONG_SPEECH_MS - SHORT_SPEECH_MS)
    return clamp(max_ms + (base_ms - max_ms) * fraction)
