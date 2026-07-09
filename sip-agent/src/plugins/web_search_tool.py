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

import httpx

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

SNIPPET_MAX_CHARS = 200


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


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

        # Spoken summary never includes URLs; those live in data only.
        spoken_parts = []
        for r in results:
            if r["snippet"]:
                spoken_parts.append(f"{r['title']}: {r['snippet']}")
            else:
                spoken_parts.append(r["title"])
        message = "Here is what I found. " + ". ".join(spoken_parts) + "."

        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"query": query, "results": results},
        )
