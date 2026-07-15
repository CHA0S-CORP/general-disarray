"""
Simon Says Tool Plugin
======================
Mirrors back exactly what the user says.

Usage in conversation:
User: "Simon says hello world"
LLM: [TOOL:SIMON_SAYS:text=hello world]
Assistant: "hello world"

User: "Repeat after me: the quick brown fox"
LLM: [TOOL:SIMON_SAYS:text=the quick brown fox]
Assistant: "the quick brown fox"
"""

import logging
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class SimonSaysTool(BaseTool):
    """Mirror back exactly what the user says."""
    
    name = "SIMON_SAYS"
    description = "Repeat back exactly what the user says. Use when user says 'simon says', 'repeat after me', 'say this', 'echo', or asks you to repeat something verbatim."
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode
    
    parameters = {
        "text": {
            "type": "string",
            "description": "The exact text to repeat back verbatim",
            "required": True
        }
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        text = params.get("text", "")
        
        if not text:
            logger.warning("Simon Says called with empty text")
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I didn't catch what you wanted me to say."
            )
        
        # Clean up the text slightly but preserve the content
        text = str(text).strip()
        
        logger.info(f"Simon Says: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=text,
            data={"echoed": text, "length": len(text)}
        )
