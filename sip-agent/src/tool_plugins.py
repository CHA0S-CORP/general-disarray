"""
Tool Plugin System
==================
Makes adding new tools as easy as dropping a Python file into the plugins/ directory.

Each plugin file should:
1. Define a class that inherits from BaseTool
2. Set the `name` and `description` class attributes
3. Implement the `execute(self, params: Dict[str, Any]) -> ToolResult` method

Example plugin (plugins/my_tool.py):
```python
from tool_plugins import BaseTool, ToolResult, ToolStatus

class MyCustomTool(BaseTool):
    name = "MY_TOOL"
    description = "Does something cool"
    
    # Optional: define parameters the tool accepts
    parameters = {
        "param1": {"type": "string", "description": "First parameter", "required": True},
        "param2": {"type": "integer", "description": "Optional number", "required": False, "default": 10}
    }
    
    async def execute(self, params: dict) -> ToolResult:
        param1 = params.get("param1", "")
        param2 = params.get("param2", 10)
        
        # Do something...
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message="It worked!",
            data={"result": "some data"}
        )
```
"""

import os
import sys
import logging
import importlib
import importlib.util
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from main import SIPAIAssistant

logger = logging.getLogger(__name__)


# =============================================================================
# Base Classes (exported for plugins to use)
# =============================================================================

class ToolStatus(str, Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    status: ToolStatus
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_speech(self) -> str:
        """Convert result to speech-friendly text."""
        return self.message


@dataclass
class ToolCall:
    """Parsed tool call from LLM output."""
    name: str
    params: Dict[str, Any]
    raw_match: str = ""


class BaseTool(ABC):
    """
    Base class for all tools.
    
    Subclasses must:
    - Set `name` class attribute (uppercase, e.g., "WEATHER")
    - Set `description` class attribute
    - Implement `execute()` method
    
    Optional:
    - Set `parameters` dict to define accepted parameters
    - Set `enabled` to False to disable the tool
    - Override `validate_params()` for custom validation
    """
    
    name: str = "UNNAMED_TOOL"
    description: str = "No description provided"
    parameters: Dict[str, Dict[str, Any]] = {}
    enabled: bool = True
    
    def __init__(self, assistant: 'SIPAIAssistant'):
        self.assistant = assistant
        self.config = assistant.config if assistant else None
        
    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Validate parameters. Returns error message if invalid, None if valid.
        Override this for custom validation logic.
        """
        for param_name, param_spec in self.parameters.items():
            if param_spec.get("required", False) and param_name not in params:
                return f"Missing required parameter: {param_name}"
                
            if param_name in params:
                expected_type = param_spec.get("type", "string")
                value = params[param_name]
                
                # Basic type checking
                if expected_type == "integer" and not isinstance(value, int):
                    try:
                        params[param_name] = int(value)
                    except (ValueError, TypeError):
                        return f"Parameter {param_name} must be an integer"
                        
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    try:
                        params[param_name] = float(value)
                    except (ValueError, TypeError):
                        return f"Parameter {param_name} must be a number"
                        
                elif expected_type == "boolean" and not isinstance(value, bool):
                    if isinstance(value, str):
                        params[param_name] = value.lower() in ("true", "yes", "1")
                    else:
                        return f"Parameter {param_name} must be a boolean"
                        
        # Apply defaults for missing optional params
        for param_name, param_spec in self.parameters.items():
            if param_name not in params and "default" in param_spec:
                params[param_name] = param_spec["default"]
                
        return None
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def get_prompt_description(self) -> str:
        """Get description for the system prompt."""
        if not self.parameters:
            return f"- {self.name}: {self.description}"
            
        param_strs = []
        for param_name, param_spec in self.parameters.items():
            required = param_spec.get("required", False)
            param_type = param_spec.get("type", "string")
            desc = param_spec.get("description", "")
            
            if required:
                param_strs.append(f"{param_name}={param_type.upper()}")
            else:
                default = param_spec.get("default", "")
                param_strs.append(f"{param_name}={param_type.upper()} (optional, default={default})")
                
        params_str = ", ".join(param_strs)
        return f"- {self.name}: [TOOL:{self.name}:{params_str}] - {self.description}"


# =============================================================================
# Plugin Discovery and Loading
# =============================================================================

class PluginLoader:
    """Discovers and loads tool plugins from a directory."""
    
    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """
        Initialize the plugin loader.
        
        Args:
            plugin_dirs: List of directories to search for plugins.
                        Defaults to ./plugins and ./src/plugins
        """
        if plugin_dirs is None:
            # Default plugin directories
            base_dir = Path(__file__).parent
            plugin_dirs = [
                base_dir / "plugins",
                base_dir.parent / "plugins",
                Path("./plugins"),
            ]
            
        self.plugin_dirs = [Path(d) for d in plugin_dirs]
        self._discovered_tools: Dict[str, Type[BaseTool]] = {}
        
    def discover_plugins(self) -> Dict[str, Type[BaseTool]]:
        """
        Discover all tool plugins in the plugin directories.
        
        Returns:
            Dict mapping tool names to tool classes
        """
        self._discovered_tools = {}
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
                
            logger.info(f"Scanning plugin directory: {plugin_dir}")
            
            # Find all Python files in the plugin directory
            for plugin_file in plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue  # Skip __init__.py, etc.
                    
                try:
                    self._load_plugin_file(plugin_file)
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_file}: {e}")
                    
        logger.info(f"Discovered {len(self._discovered_tools)} tool plugins")
        return self._discovered_tools
        
    def _load_plugin_file(self, plugin_file: Path):
        """Load a single plugin file and extract tool classes."""
        module_name = f"tool_plugin_{plugin_file.stem}"
        
        # Add parent directory (src/) to sys.path so plugins can import tool_plugins, etc.
        parent_dir = str(plugin_file.parent.parent)
        path_added = False
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            path_added = True
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load spec for {plugin_file}")
                return
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"Error executing plugin module {plugin_file}: {e}", exc_info=True)
                return
                
            # Find all BaseTool subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if it's a class that inherits from BaseTool
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseTool) and 
                    attr is not BaseTool and
                    hasattr(attr, 'name') and
                    attr.name != "UNNAMED_TOOL"):
                    
                    tool_name = attr.name.upper()
                    
                    if not attr.enabled:
                        logger.info(f"Skipping disabled plugin tool: {tool_name}")
                        continue
                        
                    if tool_name in self._discovered_tools:
                        logger.warning(f"Duplicate tool name: {tool_name} (keeping first)")
                        continue
                        
                    self._discovered_tools[tool_name] = attr
                    logger.info(f"Loaded plugin tool: {tool_name} from {plugin_file.name}")
        finally:
            # Clean up sys.path if we added to it
            if path_added and parent_dir in sys.path:
                sys.path.remove(parent_dir)
                
    def get_tool_class(self, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool class by name."""
        return self._discovered_tools.get(name.upper())
        
    def list_tools(self) -> List[str]:
        """List all discovered tool names."""
        return list(self._discovered_tools.keys())


# =============================================================================
# Plugin-aware Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Registry for all tools (built-in and plugins).
    
    Usage:
        registry = ToolRegistry(assistant)
        registry.discover_plugins()
        registry.register_builtin_tools()
        
        tool = registry.get_tool("WEATHER")
        result = await tool.execute({"location": "NYC"})
    """
    
    def __init__(self, assistant: 'SIPAIAssistant', plugin_dirs: Optional[List[Path]] = None):
        self.assistant = assistant
        self.plugin_loader = PluginLoader(plugin_dirs)
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        
    def discover_plugins(self) -> int:
        """
        Discover and register all plugin tools.
        
        Returns:
            Number of plugins discovered
        """
        plugin_classes = self.plugin_loader.discover_plugins()
        
        for name, tool_class in plugin_classes.items():
            self._tool_classes[name] = tool_class
            try:
                self._tools[name] = tool_class(self.assistant)
                logger.debug(f"Instantiated plugin tool: {name}")
            except Exception as e:
                logger.error(f"Failed to instantiate plugin tool {name}: {e}")
                
        return len(plugin_classes)
        
    def register_tool(self, tool: BaseTool):
        """Register a tool instance."""
        name = tool.name.upper()
        self._tools[name] = tool
        self._tool_classes[name] = type(tool)
        logger.debug(f"Registered tool: {name}")
        
    def register_tool_class(self, tool_class: Type[BaseTool]):
        """Register a tool class (will be instantiated on first use)."""
        name = tool_class.name.upper()
        self._tool_classes[name] = tool_class
        logger.debug(f"Registered tool class: {name}")
        
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        name = name.upper()
        
        # Return existing instance
        if name in self._tools:
            return self._tools[name]
            
        # Try to instantiate from class
        if name in self._tool_classes:
            try:
                self._tools[name] = self._tool_classes[name](self.assistant)
                return self._tools[name]
            except Exception as e:
                logger.error(f"Failed to instantiate tool {name}: {e}")
                
        return None
        
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name.upper() in self._tools or name.upper() in self._tool_classes
        
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(set(self._tools.keys()) | set(self._tool_classes.keys()))
        
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all tool instances."""
        # Instantiate any classes that haven't been instantiated yet
        for name in self._tool_classes:
            if name not in self._tools:
                self.get_tool(name)
        return self._tools.copy()
        
    def get_prompt_descriptions(self) -> str:
        """Get tool descriptions for the system prompt."""
        descriptions = []
        for tool in self.get_all_tools().values():
            if tool.enabled:
                descriptions.append(tool.get_prompt_description())
        return "\n".join(descriptions)
        
    def unregister_tool(self, name: str):
        """Unregister a tool."""
        name = name.upper()
        self._tools.pop(name, None)
        self._tool_classes.pop(name, None)


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    "BaseTool",
    "ToolResult", 
    "ToolStatus",
    "ToolCall",
    "ToolRegistry",
    "PluginLoader",
]
