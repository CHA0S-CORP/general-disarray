"""
Joke Tool Plugin
================
Tells random jokes. A simple example plugin.

Usage in conversation:
User: "Tell me a joke"
LLM: [TOOL:JOKE]
"""

import random
from typing import Any, Dict

# Import from the plugin system
from tool_plugins import BaseTool, ToolResult, ToolStatus


class JokeTool(BaseTool):
    """Tell a random joke."""
    
    name = "JOKE"
    description = "Tell a random joke to lighten the mood"
    enabled = True
    
    # Optional parameters
    parameters = {
        "category": {
            "type": "string",
            "description": "Category of joke (general, tech, dad)",
            "required": False,
            "default": "general"
        }
    }
    
    # Joke database
    JOKES = {
        "general": [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "I'm reading a book about anti-gravity. It's impossible to put down!",
            "Why don't eggs tell jokes? They'd crack each other up!",
        ],
        "tech": [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "There are only 10 types of people in the world: those who understand binary and those who don't.",
            "A SQL query walks into a bar, walks up to two tables and asks, 'Can I join you?'",
            "Why did the developer go broke? Because he used up all his cache!",
            "What's a programmer's favorite hangout place? Foo Bar!",
        ],
        "dad": [
            "I'm afraid for the calendar. Its days are numbered.",
            "What do you call a fake noodle? An impasta!",
            "I used to hate facial hair, but then it grew on me.",
            "What did the ocean say to the beach? Nothing, it just waved.",
            "Why did the coffee file a police report? It got mugged!",
        ]
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        category = params.get("category", "general").lower()
        
        # Validate category
        if category not in self.JOKES:
            category = "general"
            
        # Pick a random joke
        joke = random.choice(self.JOKES[category])
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=joke,
            data={"category": category}
        )
