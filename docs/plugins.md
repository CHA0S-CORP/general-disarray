---
title: "Creating Plugins"
excerpt: "Build custom tools for the voice assistant"
category: "Development"
slug: "plugins"
---

# Creating Plugins

Extend the SIP AI Assistant with custom tools by creating Python plugins. This guide covers everything you need to build your own tools.

## Plugin Basics

Plugins are Python files in the `src/plugins/` directory that define tool classes. Each tool:

1. Inherits from `BaseTool`
2. Defines a name, description, and parameters
3. Implements an `execute()` method

## Minimal Example

Create `src/plugins/hello_tool.py`:

```python
from typing import Any, Dict
from tool_plugins import BaseTool, ToolResult, ToolStatus

class HelloTool(BaseTool):
    name = "HELLO"
    description = "Say hello to someone"
    enabled = True
    
    parameters = {
        "name": {
            "type": "string",
            "description": "Name to greet",
            "required": True
        }
    }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        name = params.get("name", "friend")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"Hello, {name}! Nice to meet you."
        )
```

After adding the file, register it in `tool_manager.py`:

```python
from plugins.hello_tool import HelloTool

# In _load_tools():
tool_classes = [
    # ... existing tools ...
    HelloTool,
]
```

Restart the service and your tool is available!

## Tool Class Structure

```python
from typing import Any, Dict
from tool_plugins import BaseTool, ToolResult, ToolStatus

class MyTool(BaseTool):
    # Required: Unique tool name (uppercase)
    name = "MY_TOOL"
    
    # Required: Description shown to LLM
    description = "What this tool does"
    
    # Optional: Enable/disable (default: True)
    enabled = True
    
    # Optional: Parameter definitions
    parameters = {
        "param_name": {
            "type": "string",      # string, integer, number, boolean
            "description": "...",   # Help text
            "required": True,       # Required or optional
            "default": "value"      # Default if not provided
        }
    }
    
    # Optional: Custom initialization
    def __init__(self, assistant):
        super().__init__(assistant)
        # Access config: self.config
        # Access assistant: self.assistant
    
    # Required: Execute the tool
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        # Your tool logic here
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message="Result to speak to user",
            data={"optional": "structured data"}
        )
```

## Parameter Types

| Type | Python Type | Example Value |
|------|-------------|---------------|
| `string` | str | "hello" |
| `integer` | int | 42 |
| `number` | float | 3.14 |
| `boolean` | bool | true/false |

Parameters are automatically validated and converted to the correct type.

## Return Values

Always return a `ToolResult`:

```python
from tool_plugins import ToolResult, ToolStatus

# Success
return ToolResult(
    status=ToolStatus.SUCCESS,
    message="Message to speak to user",
    data={"key": "value"}  # Optional structured data
)

# Failure
return ToolResult(
    status=ToolStatus.FAILED,
    message="Error message to speak"
)

# Pending (for async operations)
return ToolResult(
    status=ToolStatus.PENDING,
    message="Working on it...",
    data={"task_id": "abc123"}
)
```

## Accessing Resources

Your tool has access to:

```python
class MyTool(BaseTool):
    async def execute(self, params):
        # Configuration
        api_key = self.config.my_api_key
        
        # Current call info
        if self.assistant.current_call:
            caller_id = self.assistant.current_call.caller
        
        # Schedule tasks
        task_id = await self.assistant.tool_manager.schedule_task(
            task_type="my_task",
            delay_seconds=60,
            message="Task complete"
        )
        
        # Make HTTP requests
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com")
```

## Full Example: Stock Price Tool

```python
"""
Stock Price Tool
================
Get current stock prices from an API.
"""

import logging
import httpx
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class StockPriceTool(BaseTool):
    name = "STOCK"
    description = "Get the current price of a stock"
    enabled = True
    
    parameters = {
        "symbol": {
            "type": "string",
            "description": "Stock ticker symbol (e.g., AAPL, GOOGL)",
            "required": True
        }
    }
    
    def __init__(self, assistant):
        super().__init__(assistant)
        # Check for required config
        self.api_key = getattr(self.config, 'stock_api_key', None)
        if not self.api_key:
            self.enabled = False
            logger.info("Stock tool disabled - no API key configured")
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        symbol = params.get("symbol", "").upper()
        
        if not symbol:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Please specify a stock symbol"
            )
        
        try:
            # Fetch stock data
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://api.stockprovider.com/quote/{symbol}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                
                if response.status_code == 404:
                    return ToolResult(
                        status=ToolStatus.FAILED,
                        message=f"Stock symbol {symbol} not found"
                    )
                
                response.raise_for_status()
                data = response.json()
            
            # Format result
            price = data.get("price", 0)
            change = data.get("change", 0)
            change_pct = data.get("change_percent", 0)
            
            direction = "up" if change >= 0 else "down"
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=f"{symbol} is trading at ${price:.2f}, {direction} {abs(change_pct):.1f}% today",
                data={
                    "symbol": symbol,
                    "price": price,
                    "change": change,
                    "change_percent": change_pct
                }
            )
            
        except httpx.TimeoutException:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Stock service is not responding"
            )
        except Exception as e:
            logger.error(f"Stock fetch error: {e}")
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Error fetching stock data"
            )
```

## Full Example: Home Assistant Integration

```python
"""
Home Assistant Tool
===================
Control smart home devices via Home Assistant.
"""

import logging
import httpx
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class HomeAssistantTool(BaseTool):
    name = "HOME"
    description = "Control smart home devices (lights, switches, etc.)"
    enabled = True
    
    parameters = {
        "action": {
            "type": "string",
            "description": "Action: on, off, toggle, status",
            "required": True
        },
        "device": {
            "type": "string",
            "description": "Device name or entity ID",
            "required": True
        }
    }
    
    def __init__(self, assistant):
        super().__init__(assistant)
        self.ha_url = getattr(self.config, 'home_assistant_url', None)
        self.ha_token = getattr(self.config, 'home_assistant_token', None)
        
        if not self.ha_url or not self.ha_token:
            self.enabled = False
            logger.info("Home Assistant tool disabled - not configured")
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        action = params.get("action", "").lower()
        device = params.get("device", "")
        
        # Map friendly names to entity IDs
        entity_id = self._resolve_device(device)
        if not entity_id:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Device '{device}' not found"
            )
        
        try:
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                if action == "status":
                    # Get current state
                    response = await client.get(
                        f"{self.ha_url}/api/states/{entity_id}",
                        headers=headers
                    )
                    data = response.json()
                    state = data.get("state", "unknown")
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        message=f"The {device} is currently {state}"
                    )
                else:
                    # Perform action
                    service = self._get_service(action, entity_id)
                    response = await client.post(
                        f"{self.ha_url}/api/services/{service}",
                        headers=headers,
                        json={"entity_id": entity_id}
                    )
                    
                    if response.status_code == 200:
                        return ToolResult(
                            status=ToolStatus.SUCCESS,
                            message=f"Done! The {device} is now {action}"
                        )
                    else:
                        return ToolResult(
                            status=ToolStatus.FAILED,
                            message=f"Couldn't {action} the {device}"
                        )
                        
        except Exception as e:
            logger.error(f"Home Assistant error: {e}")
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Error communicating with Home Assistant"
            )
    
    def _resolve_device(self, device: str) -> str:
        """Map friendly names to entity IDs."""
        device_map = {
            "living room light": "light.living_room",
            "bedroom light": "light.bedroom",
            "kitchen light": "light.kitchen",
            "front door": "lock.front_door",
            "garage": "cover.garage_door",
        }
        return device_map.get(device.lower(), device)
    
    def _get_service(self, action: str, entity_id: str) -> str:
        """Get the appropriate service for the action."""
        domain = entity_id.split(".")[0]
        
        if action in ("on", "off", "toggle"):
            return f"{domain}/turn_{action}" if action != "toggle" else f"{domain}/toggle"
        
        return f"{domain}/{action}"
```

## Best Practices

### 1. Handle Errors Gracefully

Always catch exceptions and return user-friendly messages:

```python
try:
    result = await some_api_call()
except Exception as e:
    logger.error(f"Error: {e}")
    return ToolResult(
        status=ToolStatus.FAILED,
        message="I couldn't complete that request"
    )
```

### 2. Validate Input

Check parameters before using them:

```python
async def execute(self, params):
    value = params.get("value")
    if not value or len(value) > 100:
        return ToolResult(
            status=ToolStatus.FAILED,
            message="Please provide a valid value"
        )
```

### 3. Use Logging

Log important events for debugging:

```python
from logging_utils import log_event

log_event(logger, logging.INFO, "Tool executed",
         event="my_tool_success", data={"key": "value"})
```

### 4. Keep Messages Conversational

Messages are spoken aloud - keep them natural:

```python
# Good
message="The temperature is 72 degrees"

# Bad  
message="Temperature: 72°F | Humidity: 45%"
```

### 5. Disable When Unconfigured

Check for required config and disable if missing:

```python
def __init__(self, assistant):
    super().__init__(assistant)
    if not self.config.required_setting:
        self.enabled = False
```

## Testing Your Plugin

Test via the API:

```bash
# List tools (should include yours)
curl http://localhost:8080/tools

# Execute your tool
curl -X POST http://localhost:8080/tools/MY_TOOL/execute \
  -H "Content-Type: application/json" \
  -d '{"params": {"param1": "value1"}}'
```

## Plugin Directory Structure

```
src/plugins/
├── __init__.py
├── README.md
├── timer_tool.py
├── callback_tool.py
├── weather_tool.py
├── hangup_tool.py
├── status_tool.py
├── cancel_tool.py
├── datetime_tool.py
├── calc_tool.py
├── joke_tool.py
└── my_custom_tool.py   # Your plugins here!
```
