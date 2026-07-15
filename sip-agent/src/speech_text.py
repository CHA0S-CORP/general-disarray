"""
Speech Text
===========
Text normalization for TTS: make LLM output speakable.

Conservative by design: only strips constructs that are unambiguously
formatting (paired markers, line-anchored bullets/headers, URLs, emoji).
Math like "2*3" or "a * b" and snake_case identifiers pass through.
"""

import re

_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]*\)')             # [text](url) -> text
_CODE = re.compile(r'`{1,3}([^`]*)`{1,3}')                  # `code` -> code
_HEADER = re.compile(r'^\s{0,3}#{1,6}\s+', re.MULTILINE)    # "# Title" -> "Title"
_BULLET = re.compile(r'^\s*(?:[-*+]|\d+[.)])\s+', re.MULTILINE)
_BOLD_IT = re.compile(r'(\*{1,3})(?=\S)(.+?)(?<=\S)\1')     # paired * only
_UNDERLINE = re.compile(r'(_{2,3})(?=\S)(.+?)(?<=\S)\1')    # __bold__ only
_URL = re.compile(r'(?:https?://|www\.)\S+')
_EMOJI = re.compile(
    '[\U0001F000-\U0001FAFF\U00002600-\U000027BF'
    '\U0001F1E6-\U0001F1FF⬀-⯿️‍❤]+')
_WS = re.compile(r'\s+')
_SPACE_PUNCT = re.compile(r'\s+([.,!?;:])')


def sanitize_for_speech(text: str) -> str:
    """Strip formatting artifacts so text reads naturally through TTS.

    Must be a no-op on plain conversational prose (all pre-cached phrases
    pass through unchanged, keeping their TTS cache keys valid).
    """
    if not text:
        return ""
    t = _MD_LINK.sub(r'\1', text)
    t = _CODE.sub(r'\1', t)
    t = _HEADER.sub('', t)
    t = _BULLET.sub('', t)
    t = _BOLD_IT.sub(r'\2', t)
    t = _UNDERLINE.sub(r'\2', t)
    t = _URL.sub('a link', t)
    t = _EMOJI.sub('', t)
    t = _WS.sub(' ', t)          # newlines from ex-bullets flow into prose
    t = _SPACE_PUNCT.sub(r'\1', t)
    return t.strip()


# --- farewell detection ------------------------------------------------------
# Words that may pad a farewell without changing its meaning ("okay thanks,
# bye now"). Deliberately NOT farewells by themselves.
_FILLER_WORDS = frozenset(
    "ok okay alright all right well no nope yeah yep so anyway now then "
    "thanks thank you very much a lot".split())

# Whole phrases (post-normalization) that mean "this call is over". Matched
# greedily longest-first against the full utterance, so anything beyond a
# farewell + pleasantries disqualifies it ("bye the way..." never matches).
_FAREWELLS = frozenset((
    "bye", "bye bye", "goodbye", "good bye", "bye now", "goodbye now",
    "bye for now", "see ya", "see you", "see you later", "see you soon",
    "talk to you later", "talk to you soon", "catch you later", "take care",
    "gotta go", "i gotta go", "got to go", "i got to go", "i have to go",
    "i need to go", "i've got to go", "have a good one", "have a good day",
    "have a good night", "good night", "hang up", "hang up now",
    "end the call", "that's all", "that is all", "that'll be all",
    "that will be all", "i'm done", "we're done", "i'm all set",
))
# Phrases that merely CONTAIN a farewell prefix but mean "that's okay", not
# "goodbye" ("that's all" + "right"). Consumed like fillers, checked before
# _FAREWELLS so longest-first matching can't split them.
_NON_FAREWELLS = frozenset((
    "that's all right", "that is all right",
))
_MAX_FAREWELL_WORDS = 10
_NORMALIZE = re.compile(r"[^a-z']+")


def is_farewell(text: str) -> bool:
    """True when an utterance is nothing but a goodbye (plus pleasantries).

    Deliberately conservative: every word must belong to a farewell phrase or
    be a filler, so mixed utterances ("bye the way, one more thing") always
    go to the LLM instead of hanging up on the caller.
    """
    if not text:
        return False
    words = _NORMALIZE.sub(' ', text.lower()).split()
    if not words or len(words) > _MAX_FAREWELL_WORDS:
        return False

    matched_farewell = False
    i = 0
    while i < len(words):
        for n in (5, 4, 3, 2, 1):
            phrase = " ".join(words[i:i + n])
            if phrase in _NON_FAREWELLS:
                i += n
                break
            if phrase in _FAREWELLS:
                matched_farewell = True
                i += n
                break
        else:
            if words[i] in _FILLER_WORDS:
                i += 1
            else:
                return False
    return matched_farewell
