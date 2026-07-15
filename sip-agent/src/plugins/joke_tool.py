"""
Joke Tool Plugin
================
Tells random jokes. A simple example plugin.

Usage in conversation:
User: "Tell me a joke"
LLM: [TOOL:JOKE]
"""

import logging
import random
from typing import Any, Dict

# Import from the plugin system
from tool_plugins import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class JokeTool(BaseTool):
    """Tell a random joke."""
    
    name = "JOKE"
    description = "Tell a random joke to lighten the mood"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode
    
    # Optional parameters
    parameters = {
        "category": {
            "type": "string",
            "description": "Category of joke: general, tech, or dad",
            "required": False,
            "default": "general"
        }
    }
    
    # Joke database - keep punctuation simple for TTS
    JOKES = {
        "general": [
            "Why don't scientists trust atoms? Because they make up everything.",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Why did the scarecrow win an award? He was outstanding in his field.",
            "I'm reading a book about anti-gravity. It's impossible to put down.",
            "Why don't eggs tell jokes? They'd crack each other up.",
            "What do you call a fish without eyes? A fsh.",
            "I used to hate facial hair, but then it grew on me.",
        ],
        "tech": [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "There are only 10 types of people in the world. Those who understand binary and those who don't.",
            "A SQL query walks into a bar, walks up to two tables and asks, Can I join you?",
            "Why did the developer go broke? Because he used up all his cache.",
            "What's a programmer's favorite hangout place? Foo Bar.",
            "Why do Java developers wear glasses? Because they can't C sharp.",
        ],
        "dad": [
            "I'm afraid for the calendar. Its days are numbered.",
            "What do you call a fake noodle? An impasta.",
            "What did the ocean say to the beach? Nothing, it just waved.",
            "Why did the coffee file a police report? It got mugged.",
            "What do you call a bear with no teeth? A gummy bear.",
            "I'm on a seafood diet. I see food and I eat it.",
        ]
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        logger.debug(f"Joke tool executing with params: {params}")
        
        category = params.get("category", "general")
        if isinstance(category, str):
            category = category.lower().strip()
        else:
            category = "general"
        
        # Validate category
        if category not in self.JOKES:
            logger.debug(f"Unknown category '{category}', falling back to general")
            category = "general"
            
        # Pick a random joke
        joke = random.choice(self.JOKES[category])
        logger.info(f"Joke tool selected: {joke[:50]}...")
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=joke,
            data={"category": category}
        )
