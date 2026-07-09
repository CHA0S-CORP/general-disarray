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
