"""
Web Search Tool Plugin
======================
Searches the web through a self-hosted SearxNG instance and speaks a short
summary of the top results.

Requires configuration:
- SEARXNG_URL: Base URL of the SearxNG instance (empty disables the tool)
- WEB_SEARCH_MAX_RESULTS: How many results to include (default 3)

Usage in conversation:
User: "Search the web for the tallest building in the world"
LLM: [TOOL:WEB_SEARCH:query=tallest building in the world]
"""

import html
import logging
import re
from typing import Any, Dict, List, Optional

from plugins.helpers import fetch_json

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

SNIPPET_MAX_CHARS = 200
# Spoken titles are a list read down a phone line — keep each one short enough
# that three of them still land as one sentence.
TITLE_MAX_CHARS = 70


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    return await fetch_json(url, params=params, headers=headers)


def _strip_html(text: str) -> str:
    """Remove HTML tags/entities and collapse whitespace to single spaces."""
    if not text:
        return ""
    text = re.sub("<[^>]+>", "", text)
    text = html.unescape(text)
    return " ".join(text.split())


def _truncate_snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Truncate to roughly max_chars, breaking at a word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    return cut.rstrip(" ,;:.") + "..."


def _spoken_title(title: str) -> str:
    """Reduce a scraped page title to something a TTS voice can say.

    Search-result titles are written for search engines, not ears: they carry
    bullets and pipes as separators ("Events | Things To Do · San Diego"), a
    trailing site name after a dash, and decorative punctuation that TTS either
    reads aloud or stumbles over. Keep the leading segment — the part that
    names the thing — and drop the rest.
    """
    if not title:
        return ""
    # Split on the usual title separators and keep the first meaningful piece.
    # Titles often OPEN with a separator ("· Film Screening: Love Birds · ..."),
    # so take the first NON-EMPTY segment, not merely the first one.
    parts = [p.strip()
             for p in re.split(r"\s*[|·•‧–—]\s*|\s+-\s+", title.strip())]
    head = next((p for p in parts if p), "")
    # Strip anything that isn't speech: stray brackets, quotes, trailing punct.
    head = re.sub(r"[\[\](){}\"“”<>*#]+", " ", head)
    head = " ".join(head.split()).rstrip(" ,;:.-")
    # An over-long title is an article headline, not a name — cut it at a word
    # boundary rather than making the caller sit through it.
    if len(head) > TITLE_MAX_CHARS:
        cut = head[:TITLE_MAX_CHARS]
        space = cut.rfind(" ")
        head = (cut[:space] if space > 0 else cut).rstrip(" ,;:.-")
    return head


def _join_naturally(items: List[str]) -> str:
    """Join for the ear: "a", "a and b", "a, b, and c"."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


class WebSearchTool(BaseTool):
    """Search the web via a self-hosted SearxNG instance."""

    name = "WEB_SEARCH"
    description = ("Search the web for current information, facts, or news "
                   "you don't already know")
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "query": {
            "type": "string",
            "description": "What to search for, phrased as a short search query",
            "required": True,
        }
    }

    def __init__(self, assistant):
        super().__init__(assistant)
        if self.config and not self.config.searxng_url:
            self.enabled = False
            logger.info("WEB_SEARCH tool disabled - SEARXNG_URL not configured")

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = str(params.get("query") or "").strip()
        if not query:
            return ToolResult(status=ToolStatus.FAILED,
                              message="I need something to search for.")

        base_url = (self.config.searxng_url if self.config else "").rstrip("/")
        if not base_url:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Web search is not configured.")

        max_results = self.config.web_search_max_results if self.config else 3

        try:
            payload = await _fetch_json(
                base_url + "/search",
                params={"q": query, "format": "json", "safesearch": 1},
            )
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return ToolResult(status=ToolStatus.FAILED,
                              message="Search is not available right now.")

        raw_results = (payload or {}).get("results") or []
        results: List[Dict[str, str]] = []
        for item in raw_results[:max_results]:
            title = _strip_html(str(item.get("title") or ""))
            snippet = _truncate_snippet(_strip_html(str(item.get("content") or "")))
            results.append({
                "title": title,
                "snippet": snippet,
                "url": str(item.get("url") or ""),
            })

        log_event(logger, logging.INFO,
                  f"Web search '{query}': {len(results)} results",
                  event="web_search", query=query, result_count=len(results))

        if not results:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=f"I could not find anything about {query}.",
                data={"query": query, "results": []},
            )

        # What the MODEL reads: titles + snippets, so it has something to
        # actually answer the question from. Never URLs — nothing should tempt
        # it into reading one aloud.
        model_parts = []
        for r in results:
            if r["snippet"]:
                model_parts.append(f"{r['title']}: {r['snippet']}")
            else:
                model_parts.append(r["title"])
        message = "Search results for '{}':\n{}".format(
            query, "\n".join(f"- {p}" for p in model_parts))

        # What the CALLER hears, if this ever gets spoken instead of summarized.
        # Snippets are scraped web text — bullets, SEO boilerplate, sentence
        # fragments — and reading them verbatim down a phone line produced a
        # minute of unlistenable junk. Titles only, and only a few.
        spoken_titles = [_spoken_title(r["title"]) for r in results[:3]]
        spoken_titles = [t for t in spoken_titles if t]
        spoken = ("Here's what I found: " + _join_naturally(spoken_titles) + "."
                  if spoken_titles else
                  f"I found some results for {query}, but nothing I can read out.")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            spoken_message=spoken,
            data={"query": query, "results": results},
        )
