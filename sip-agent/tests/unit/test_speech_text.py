"""Unit tests for the TTS text sanitizer."""
import pytest

from speech_text import sanitize_for_speech

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("raw,expected", [
    # Markdown emphasis
    ("**bold** and *italic*", "bold and italic"),
    ("__x__ and ___y___", "x and y"),
    ("***both***", "both"),
    # Headers and bullets flow into prose
    ("# Header\nbody", "Header body"),
    ("Here you go:\n- apples\n- pears\n1. figs", "Here you go: apples pears figs"),
    ("* starred\n+ plussed", "starred plussed"),
    # Code
    ("`code` and ```block```", "code and block"),
    # Links and URLs
    ("[docs](https://example.com)", "docs"),
    ("see https://example.com/x?y=1 now", "see a link now"),
    ("visit www.example.com today", "visit a link today"),
    # Emoji
    ("Great! 🎉👍", "Great!"),
    ("🎉👍", ""),
    ("a ❤️ b", "a b"),
    # Whitespace
    ("a\n\n b", "a b"),
    ("word .", "word."),
])
def test_sanitize(raw, expected):
    assert sanitize_for_speech(raw) == expected


@pytest.mark.parametrize("text", [
    "2*3 equals 6",
    "3 * 4",
    "snake_case_name stays",
    "3.5 seconds",
    "That's a plain sentence, with punctuation!",
])
def test_preserves_plain_text(text):
    assert sanitize_for_speech(text) == text


def test_empty_input():
    assert sanitize_for_speech("") == ""


def test_default_phrases_are_unchanged():
    """Cache safety: sanitizing must be a no-op on every pre-cached phrase."""
    from config import PhrasesConfig
    for phrase in PhrasesConfig().get_all_phrases_for_cache():
        assert sanitize_for_speech(phrase) == phrase


# --- farewell detection ------------------------------------------------------

from speech_text import is_farewell  # noqa: E402


@pytest.mark.parametrize("text", [
    "Bye.",
    "bye bye",
    "Goodbye!",
    "Good bye",
    "Okay, thanks. Bye!",
    "No thanks, goodbye.",
    "Thank you very much, bye now.",
    "Alright, take care!",
    "That's all, thanks!",
    "I'm done, thank you.",
    "gotta go, bye",
    "Hang up.",
    "End the call.",
    "Okay bye, take care.",
    "That's all right, bye.",           # non-farewell phrase + real farewell
])
def test_is_farewell_true(text):
    assert is_farewell(text) is True


@pytest.mark.parametrize("text", [
    "",
    "By the way, what's the weather?",
    "Bye the way",                      # STT slip, but extra non-filler word
    "Goodbye is a word I like.",
    "Tell me about the movie The Goodbye Girl.",
    "Can you say goodbye in French?",
    "No.",                              # filler only, no farewell
    "Thanks!",                          # pleasantry only, not a goodbye
    "I need to go to the store, set a timer first.",
    "What time is it?",
    "That's all right.",                # means "that's okay", not goodbye
    "that is all right",
])
def test_is_farewell_false(text):
    assert is_farewell(text) is False
