"""
Tool registry for managing available tools.

The registry provides a central place to:
- Register tools
- Look up tools by name
- List available tool specs for the planner
"""

from audio_agent.core.schemas import ToolSpec
from audio_agent.core.errors import ToolRegistryError
from audio_agent.tools.base import BaseTool


class ToolRegistry:
    """
    Registry for managing tools.
    
    Thread-safety note: This implementation is not thread-safe.
    For concurrent access, wrap with appropriate locks.
    """
    
    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
        
        Raises:
            ToolRegistryError: If tool name is empty or already registered
        """
        if not isinstance(tool, BaseTool):
            raise ToolRegistryError(
                f"Cannot register non-tool object: {type(tool).__name__}",
                details={"type": type(tool).__name__}
            )
        
        raw_name = tool.spec.name
        name = raw_name.strip() if raw_name else ""
        
        if not name:
            raise ToolRegistryError(
                "Cannot register tool with empty name",
                details={"tool_type": type(tool).__name__, "raw_name": repr(raw_name)}
            )
        
        if name in self._tools:
            raise ToolRegistryError(
                f"Tool '{name}' is already registered",
                details={"existing_tool": type(self._tools[name]).__name__}
            )
        
        self._tools[name] = tool
    
    def get(self, tool_name: str) -> BaseTool:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
        
        Returns:
            The registered tool
        
        Raises:
            ToolRegistryError: If tool is not found
        """
        if not tool_name:
            raise ToolRegistryError("Cannot look up tool with empty name")
        
        if tool_name not in self._tools:
            available = list(self._tools.keys())
            raise ToolRegistryError(
                f"Unknown tool: '{tool_name}'",
                details={"available_tools": available}
            )
        
        return self._tools[tool_name]
    
    def list_specs(self) -> list[ToolSpec]:
        """
        List specifications of all registered tools.
        
        Returns:
            List of ToolSpec objects for the planner
        """
        return [tool.spec for tool in self._tools.values()]
    
    def list_names(self) -> list[str]:
        """
        List names of all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
