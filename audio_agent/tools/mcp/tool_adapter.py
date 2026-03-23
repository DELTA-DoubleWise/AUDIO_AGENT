"""
MCPToolAdapter - Adapts MCP tools to the BaseTool interface.
"""

from __future__ import annotations

from audio_agent.core.schemas import ToolSpec, ToolCallRequest, ToolResult
from audio_agent.core.errors import ToolExecutionError
from audio_agent.tools.base import BaseTool
from audio_agent.tools.mcp.schemas import MCPToolInfo
from audio_agent.tools.mcp.server_manager import MCPServerManager


class MCPToolAdapter(BaseTool):
    """
    Adapts an MCP tool to the BaseTool interface.
    
    This allows MCP tools to be used seamlessly with the existing
    tool registry and executor.
    """
    
    def __init__(
        self,
        server_name: str,
        tool_info: MCPToolInfo,
        server_manager: MCPServerManager,
    ):
        """
        Initialize MCP tool adapter.
        
        Args:
            server_name: Name of the MCP server hosting this tool
            tool_info: MCP tool information from server
            server_manager: Server manager for getting client
        """
        self._server_name = server_name
        self._tool_info = tool_info
        self._server_manager = server_manager
        self._spec = self._build_spec()
    
    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        return self._spec
    
    def _build_spec(self) -> ToolSpec:
        """
        Convert MCP tool info to ToolSpec.
        
        The MCP inputSchema maps directly to our input_schema.
        Output is always a dict with content.
        """
        return ToolSpec(
            name=self._tool_info.name,
            description=self._tool_info.description,
            input_schema=self._tool_info.inputSchema,
            output_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "text": {"type": "string"},
                            }
                        }
                    },
                    "isError": {"type": "boolean"},
                    "error": {"type": "string"},
                }
            },
            tags=["mcp", self._server_name],
        )
    
    async def invoke(self, request: ToolCallRequest) -> ToolResult:
        """
        Invoke the MCP tool via the server.
        
        This is an async method that communicates with the MCP server.
        
        Args:
            request: Tool call request with arguments
            
        Returns:
            ToolResult with the tool output
        """
        # Validate request
        self.validate_request(request)
        
        try:
            # Get client for server
            client = await self._server_manager.get_client(self._server_name)
            
            # Call the tool
            result = await client.call_tool(
                name=self._tool_info.name,
                arguments=request.args
            )
            
            # Convert MCP result to ToolResult
            if result.isError:
                return ToolResult(
                    tool_name=self.spec.name,
                    success=False,
                    output={},
                    error_message=result.error or "MCP tool returned error",
                )
            
            # Extract text content from result
            output = {"content": []}
            for item in result.content:
                if hasattr(item, "model_dump"):
                    output["content"].append(item.model_dump())
                else:
                    output["content"].append(item)
            
            # Combine text for convenience
            text_parts = []
            for item in result.content:
                if item.type == "text":
                    text_parts.append(item.text)
            output["text"] = "\n".join(text_parts)
            
            return ToolResult(
                tool_name=self.spec.name,
                success=True,
                output=output,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                output={},
                error_message=f"MCP tool invocation failed: {e}",
            )
    
    def validate_request(self, request: ToolCallRequest) -> None:
        """
        Validate a request before execution.
        
        Raises:
            ToolExecutionError: If request is invalid
        """
        if request.tool_name != self.spec.name:
            raise ToolExecutionError(
                f"Tool name mismatch: expected '{self.spec.name}', got '{request.tool_name}'",
                details={"expected": self.spec.name, "actual": request.tool_name}
            )
