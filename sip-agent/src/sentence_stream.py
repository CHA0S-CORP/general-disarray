"""
Sentence Stream
===============
Incremental sentence assembly for LLM→TTS token streaming.

`split_into_sentences` is the canonical sentence splitter for streaming TTS
(main.py re-exports it). `SentenceAssembler` applies the SAME split/merge
rules to a stream of text deltas so sentences can be spoken the moment they
form, with one extra guarantee: it never emits text that could be part of a
`[TOOL:...]` marker (partial prefixes are held until proven harmless; a full
marker stops emission for the rest of the stream so the engine can run the
marker/tool postprocess on the un-emitted remainder).

Pure module: no config, no I/O — unit-testable like speech_text.
"""

import re
from typing import List, Tuple

# Split after sentence punctuation followed by whitespace ("3.5" is safe: no
# whitespace after the dot).
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

_DEFAULT_MIN_CHARS = 25

# Text-marker tool-call prefix (case-sensitive, matching the parser regex in
# llm_engine._process_tool_calls).
TOOL_MARKER_PREFIX = "[TOOL:"


def split_into_sentences(text: str, min_chars: int = _DEFAULT_MIN_CHARS) -> List[str]:
    """Split text into speakable sentence chunks for streaming TTS.

    Fragments shorter than ``min_chars`` merge into the following one so
    abbreviations and one-word sentences don't produce choppy TTS calls.
    """
    parts = [p for p in _SENTENCE_SPLIT.split(text.strip()) if p]
    merged: List[str] = []
    for part in parts:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


class SentenceAssembler:
    """Feed streamed text deltas, get back sentence chunks as they complete.

    Chunk boundaries match ``split_into_sentences`` exactly: feeding any text
    delta-by-delta yields the same chunks as splitting the whole text, modulo
    the final partial chunk which stays in the buffer until ``flush()``.

    Marker hold: when the buffer tail could be the start of a ``[TOOL:``
    marker (a partial prefix like ``[``, ``[T``, ``[TO`` at the end, or an
    actual ``[TOOL:`` occurrence), emission stops at that point. A false
    prefix (e.g. ``[Total``) releases the hold; a full ``[TOOL:`` sets
    ``marker_seen`` permanently — from then on every delta accumulates
    silently and ``flush()`` returns the raw remainder (marker included) for
    the engine's tool postprocess.
    """

    def __init__(self, min_chars: int = _DEFAULT_MIN_CHARS):
        self.min_chars = min_chars
        self.marker_seen = False
        self._buf = ""

    def feed(self, delta: str) -> List[str]:
        """Add a text delta; return any sentence chunks now safe to emit."""
        if not delta:
            return []
        if self.marker_seen:
            # Marker mode: accumulate the raw remainder, emit nothing.
            self._buf += delta
            return []
        if not self._buf:
            # Leading / inter-sentence separator whitespace: the splitter
            # consumes it, so it must not start the next chunk.
            delta = delta.lstrip()
            if not delta:
                return []
        self._buf += delta
        return self._drain()

    def flush(self) -> str:
        """End of stream: the un-emitted remainder, verbatim (markers intact).

        May be a partial sentence, a held short fragment, or (after
        ``marker_seen``) the raw tail including the marker text.
        """
        remainder = self._buf.strip()
        self._buf = ""
        return remainder

    def _drain(self) -> List[str]:
        hold_idx, is_marker = self._scan_marker()
        if is_marker:
            self.marker_seen = True
        region = self._buf[:hold_idx]   # emission candidates live here
        rest = self._buf[hold_idx:]     # potential/actual marker: never emitted

        parts = _SENTENCE_SPLIT.split(region)
        complete, tail = parts[:-1], parts[-1]
        merged: List[str] = []
        for part in complete:
            if merged and len(merged[-1]) < self.min_chars:
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)

        keep = tail
        if merged and len(merged[-1]) < self.min_chars:
            # Too short to emit on its own: it must merge with whatever comes
            # next, joined by a single space exactly as split_into_sentences
            # would (the original separator was already consumed).
            short = merged.pop()
            keep = f"{short} {tail}" if tail else f"{short} "
        self._buf = keep + rest
        return merged

    def _scan_marker(self) -> Tuple[int, bool]:
        """(first index emission may not cross, full-marker-start present).

        Case-sensitive: only the exact ``[TOOL:`` prefix holds; anything that
        diverges from it (``[Total``, ``[tool:``) is ordinary text.
        """
        n = len(TOOL_MARKER_PREFIX)
        pos = 0
        while True:
            i = self._buf.find("[", pos)
            if i == -1:
                return len(self._buf), False
            seg = self._buf[i:i + n]
            if seg == TOOL_MARKER_PREFIX:
                return i, True
            if len(seg) < n and TOOL_MARKER_PREFIX.startswith(seg):
                # Partial prefix at the end of the buffer: hold until more
                # text proves whether it is a marker.
                return i, False
            pos = i + 1
