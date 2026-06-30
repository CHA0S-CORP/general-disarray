"""Unit tests for LLMEngine._parse_param_value (text-based tool-call params).

Phone numbers and zip/area codes must survive as strings; only canonical
integers/floats are coerced. Booleans accept true/yes/false/no.
"""
import pytest

from llm_engine import LLMEngine

pytestmark = pytest.mark.unit


@pytest.fixture
def engine():
    # _parse_param_value touches neither config nor tool_manager.
    return LLMEngine(config=None, tool_manager=None)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("300", 300),
        ("0", 0),
        ("-5", -5),
        ("3.14", 3.14),
        ("-2.5", -2.5),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("false", False),
        ("no", False),
        ("hello world", "hello world"),
        ("  trimmed  ", "trimmed"),
    ],
)
def test_parse_param_value(engine, raw, expected):
    assert engine._parse_param_value(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "+15551234567",  # leading '+' phone number -> keep as string
        "007",           # leading zero -> keep as string
        "0123",          # leading zero -> keep as string
        "1.2.3",         # not a canonical number
    ],
)
def test_identifier_like_values_stay_strings(engine, raw):
    result = engine._parse_param_value(raw)
    assert isinstance(result, str)
    assert result == raw.strip()
