# Tool Plugins

This directory contains tool plugins that extend the SIP AI Assistant's capabilities.

## Creating a New Plugin

1. Create a new Python file in this directory (e.g., `my_tool.py`)
2. Import the base classes from `tool_plugins`
3. Create a class that inherits from `BaseTool`
4. Set the required class attributes and implement `execute()`

### Minimal Example

```python
from tool_plugins import BaseTool, ToolResult, ToolStatus

class MyTool(BaseTool):
    name = "MY_TOOL"  # How LLM calls it: [TOOL:MY_TOOL]
    description = "What this tool does"
    
    async def execute(self, params: dict) -> ToolResult:
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message="Result spoken to user"
        )
```

### Full Example with Parameters

```python
from tool_plugins import BaseTool, ToolResult, ToolStatus
from typing import Any, Dict

class GreetingTool(BaseTool):
    name = "GREET"
    description = "Greet someone by name"
    enabled = True  # Set to False to disable
    
    # Define parameters the tool accepts
    parameters = {
        "name": {
            "type": "string",
            "description": "Name of person to greet",
            "required": True
        },
        "style": {
            "type": "string",
            "description": "Greeting style: formal or casual",
            "required": False,
            "default": "casual"
        }
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        name = params.get("name", "friend")
        style = params.get("style", "casual")
        
        if style == "formal":
            message = f"Good day, {name}. How may I assist you?"
        else:
            message = f"Hey {name}! What's up?"
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"name": name, "style": style}
        )
```

## Tool Invocation Format

The LLM uses this format to call tools:

```
[TOOL:TOOL_NAME]                           # No parameters
[TOOL:TOOL_NAME:param=value]              # Single parameter
[TOOL:TOOL_NAME:param1=value1,param2=value2]  # Multiple parameters
```

Examples:
- `[TOOL:JOKE]`
- `[TOOL:DATETIME:format=time]`
- `[TOOL:CALC:expression=15*0.85]`
- `[TOOL:GREET:name=John,style=formal]`

## Available Base Classes

### ToolStatus (Enum)
- `SUCCESS` - Tool executed successfully
- `FAILED` - Tool execution failed
- `PENDING` - Tool is waiting/async
- `CANCELLED` - Tool was cancelled

### ToolResult (Dataclass)
```python
ToolResult(
    status=ToolStatus.SUCCESS,
    message="Text spoken to user",
    data={"key": "value"}  # Optional extra data
)
```

### BaseTool (Abstract Class)

Class attributes:
- `name: str` - Tool identifier (UPPERCASE)
- `description: str` - Human-readable description
- `parameters: dict` - Parameter definitions
- `enabled: bool` - Whether tool is active

Instance attributes:
- `self.assistant` - Reference to SIPAIAssistant
- `self.config` - Reference to Config

Methods to override:
- `execute(params: dict) -> ToolResult` - **Required**
- `validate_params(params: dict) -> Optional[str]` - Optional custom validation

## Parameter Types

Supported types in `parameters` definition:
- `"string"` - Text value
- `"integer"` - Whole number (auto-converted)
- `"number"` - Float/decimal (auto-converted)
- `"boolean"` - True/false (auto-converted from "true"/"false"/"yes"/"no")

## Accessing Assistant Resources

Your tool has access to the assistant instance:

```python
async def execute(self, params):
    # Access configuration
    api_key = self.config.some_api_key
    
    # Access other components
    await self.assistant.speak("Hello!")
    
    # Get current call info
    if self.assistant.current_call:
        caller = self.assistant.current_call.remote_uri
```

## Error Handling

Return a failed result for errors:

```python
async def execute(self, params):
    try:
        result = do_something()
        return ToolResult(status=ToolStatus.SUCCESS, message=result)
    except SomeError as e:
        return ToolResult(
            status=ToolStatus.FAILED,
            message=f"Sorry, something went wrong: {e}"
        )
```

## Included Plugins

### Core Tools (previously built-in)
- `timer_tool.py` - Set timers and reminders: `[TOOL:SET_TIMER:duration=300]`
- `callback_tool.py` - Schedule callback calls: `[TOOL:CALLBACK:delay=60]`
- `hangup_tool.py` - End the current call: `[TOOL:HANGUP]`
- `weather_tool.py` - Get weather conditions: `[TOOL:WEATHER]`
- `status_tool.py` - Check pending tasks: `[TOOL:STATUS]`
- `cancel_tool.py` - Cancel timers/callbacks: `[TOOL:CANCEL]`

### Example Tools
- `joke_tool.py` - Tell random jokes: `[TOOL:JOKE]`
- `datetime_tool.py` - Current date/time: `[TOOL:DATETIME]`
- `calc_tool.py` - Math calculations: `[TOOL:CALC:expression=15*0.85]`

## Tips

1. Keep tool names short and memorable
2. Make `message` conversational - it's spoken aloud
3. Use `data` for structured return values
4. Handle errors gracefully with user-friendly messages
5. Test your plugin by placing it in this directory and restarting
