"""Unit tests for the WEB_SEARCH tool (SearxNG-backed web search)."""
import pytest
from types import SimpleNamespace

import plugins.web_search_tool as web_search_tool
from plugins.web_search_tool import (
    WebSearchTool,
    _strip_html,
    _truncate_snippet,
)
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


def make_assistant(config_factory, **overrides):
    cfg = config_factory(searxng_url="http://searxng:8080", **overrides)
    return SimpleNamespace(config=cfg, session=None)


# --- _strip_html -------------------------------------------------------------

def test_strip_html_removes_tags():
    assert _strip_html("<b>Hello</b> <i>world</i>") == "Hello world"


def test_strip_html_unescapes_entities():
    assert _strip_html("Fish &amp; Chips &lt;fresh&gt;") == "Fish & Chips <fresh>"


def test_strip_html_collapses_whitespace():
    assert _strip_html("  too\n\tmany    spaces ") == "too many spaces"


def test_strip_html_empty():
    assert _strip_html("") == ""


# --- _truncate_snippet -------------------------------------------------------

def test_truncate_snippet_short_text_unchanged():
    assert _truncate_snippet("short and sweet") == "short and sweet"


def test_truncate_snippet_breaks_at_word_boundary():
    text = "word " * 100
    out = _truncate_snippet(text, max_chars=50)
    assert len(out) <= 54  # 50 chars + "..."
    assert out.endswith("...")
    assert " wor..." not in out  # no mid-word cut


# --- execute: happy path -----------------------------------------------------

async def test_search_happy_path(config_factory, monkeypatch):
    payload = {
        "results": [
            {"title": "<b>Burj</b> Khalifa", "content": "The tallest &amp; grandest building.",
             "url": "https://example.com/burj"},
            {"title": "Second", "content": "Another result.", "url": "https://example.com/2"},
            {"title": "Third", "content": "Yet another.", "url": "https://example.com/3"},
            {"title": "Fourth", "content": "Should be dropped.", "url": "https://example.com/4"},
        ]
    }
    seen = {}

    async def fake_fetch(url, params=None, headers=None):
        seen["url"] = url
        seen["params"] = params
        return payload

    monkeypatch.setattr(web_search_tool, "_fetch_json", fake_fetch)
    assistant = make_assistant(config_factory, web_search_max_results="2")
    tool = WebSearchTool(assistant)
    assert tool.enabled

    result = await tool.execute({"query": "tallest building"})
    assert result.status == ToolStatus.SUCCESS
    assert seen["url"] == "http://searxng:8080/search"
    assert seen["params"]["q"] == "tallest building"
    assert seen["params"]["format"] == "json"
    # max_results honored
    assert len(result.data["results"]) == 2
    # HTML stripped from the spoken message; URLs never spoken
    assert "<" not in result.message
    assert "https://" not in result.message
    assert "Burj Khalifa" in result.message
    assert result.message.startswith("Here is what I found.")
    # URLs preserved in data
    assert result.data["results"][0]["url"] == "https://example.com/burj"
    assert result.data["query"] == "tallest building"


async def test_search_empty_results(config_factory, monkeypatch):
    async def fake_fetch(url, params=None, headers=None):
        return {"results": []}

    monkeypatch.setattr(web_search_tool, "_fetch_json", fake_fetch)
    tool = WebSearchTool(make_assistant(config_factory))

    result = await tool.execute({"query": "gleeble frobnitz"})
    assert result.status == ToolStatus.SUCCESS
    assert "could not find anything about gleeble frobnitz" in result.message
    assert result.data["results"] == []


async def test_search_network_error_returns_failed(config_factory, monkeypatch):
    async def fake_fetch(url, params=None, headers=None):
        raise ConnectionError("searxng down")

    monkeypatch.setattr(web_search_tool, "_fetch_json", fake_fetch)
    tool = WebSearchTool(make_assistant(config_factory))

    result = await tool.execute({"query": "anything"})
    assert result.status == ToolStatus.FAILED
    assert result.message == "Search is not available right now."


async def test_search_missing_query_fails(config_factory, monkeypatch):
    async def fake_fetch(url, params=None, headers=None):  # must not be called
        raise AssertionError("should not fetch without a query")

    monkeypatch.setattr(web_search_tool, "_fetch_json", fake_fetch)
    tool = WebSearchTool(make_assistant(config_factory))

    result = await tool.execute({"query": "   "})
    assert result.status == ToolStatus.FAILED


def test_disabled_when_searxng_url_empty(config_factory):
    cfg = config_factory(searxng_url="")
    assistant = SimpleNamespace(config=cfg, session=None)
    tool = WebSearchTool(assistant)
    assert tool.enabled is False
