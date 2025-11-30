"""
DateTime Tool Plugin
====================
Provides current date/time information.

Usage in conversation:
User: "What time is it?"
LLM: [TOOL:DATETIME]

User: "What's today's date?"
LLM: [TOOL:DATETIME:format=date]
"""

from datetime import datetime
from typing import Any, Dict
import pytz

from tool_plugins import BaseTool, ToolResult, ToolStatus


class DateTimeTool(BaseTool):
    """Get current date and time."""
    
    name = "DATETIME"
    description = "Get the current date and/or time"
    enabled = True
    
    parameters = {
        "format": {
            "type": "string",
            "description": "What to return: 'time', 'date', 'datetime', or 'full'",
            "required": False,
            "default": "datetime"
        },
        "timezone": {
            "type": "string",
            "description": "Timezone (e.g., 'US/Pacific', 'US/Eastern', 'UTC')",
            "required": False,
            "default": "US/Pacific"
        }
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        format_type = params.get("format", "datetime").lower()
        timezone_str = params.get("timezone", "US/Pacific")
        
        # Get timezone
        try:
            tz = pytz.timezone(timezone_str)
        except pytz.exceptions.UnknownTimeZoneError:
            tz = pytz.timezone("US/Pacific")
            timezone_str = "US/Pacific"
            
        now = datetime.now(tz)
        
        # Format based on request
        if format_type == "time":
            # "3:45 PM"
            time_str = now.strftime("%I:%M %p").lstrip("0")
            message = f"The current time is {time_str}"
            
        elif format_type == "date":
            # "Saturday, November 30th, 2025"
            day = now.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            date_str = now.strftime(f"%A, %B {day}{suffix}, %Y")
            message = f"Today is {date_str}"
            
        elif format_type == "full":
            # Full with timezone
            day = now.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            time_str = now.strftime("%I:%M %p").lstrip("0")
            date_str = now.strftime(f"%A, %B {day}{suffix}, %Y")
            tz_abbrev = now.strftime("%Z")
            message = f"It's {time_str} {tz_abbrev} on {date_str}"
            
        else:  # datetime
            time_str = now.strftime("%I:%M %p").lstrip("0")
            date_str = now.strftime("%A, %B %d")
            message = f"It's {time_str} on {date_str}"
            
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={
                "iso": now.isoformat(),
                "timezone": timezone_str,
                "unix": int(now.timestamp())
            }
        )
