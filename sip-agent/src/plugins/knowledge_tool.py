"""
Knowledge Tool Plugin
=====================
Looks up information in the local knowledge base (documents dropped into
data/knowledge/, indexed at startup by knowledge_base.KnowledgeBase).

Usage in conversation:
User: "What's our refund policy?"
LLM: [TOOL:KNOWLEDGE:query=refund policy]
"""

from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus


class KnowledgeTool(BaseTool):
    """Search the local knowledge base."""

    name = "KNOWLEDGE"
    description = ("Search the local knowledge base of reference documents "
                   "for facts you don't know (policies, procedures, "
                   "site-specific information)")
    enabled = True

    parameters = {
        "query": {
            "type": "string",
            "description": "What to look up, phrased as a short search query",
            "required": True,
        }
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = str(params.get("query") or "").strip()
        if not query:
            return ToolResult(status=ToolStatus.FAILED,
                              message="I need something to search for.")

        kb = getattr(self.assistant, "knowledge_base", None)
        if kb is None or not kb.available:
            return ToolResult(status=ToolStatus.FAILED,
                              message="The knowledge base isn't available right now.")
        if not kb.ready:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="The knowledge base is still indexing. Try again in a moment.")

        results = await kb.search(query)
        if not results:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=f"I couldn't find anything about {query} in my documents.",
                data={"query": query, "results": []})

        # Feed the excerpts back to the model (native/agent mode reads them
        # and answers naturally). In text-marker mode this message is spoken
        # as-is, so cap each chunk to keep the fallback listenable.
        excerpts = "\n\n".join(
            f"From {source}:\n{chunk[:400]}" for source, chunk in results)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=excerpts,
            data={"query": query,
                  "results": [{"source": s, "text": c} for s, c in results]})
