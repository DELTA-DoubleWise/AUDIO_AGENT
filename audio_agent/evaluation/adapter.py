"""
Adapter for integrating audio_agent tools with evaluation system.
"""

from __future__ import annotations

from audio_agent.core.schemas import ToolSpec, ToolCallRequest, ToolResult
from audio_agent.tools.base import BaseTool as AgentBaseTool
from audio_agent.evaluation.core.rps import ToolPerformance


class EvaluatedTool:
    """
    Wrapper that adds evaluation capabilities to an audio_agent tool.
    
    This adapter allows any audio_agent BaseTool to be evaluated
    and tracked with RPS scores.
    """
    
    def __init__(self, tool: AgentBaseTool) -> None:
        """
        Wrap an audio_agent tool.
        
        Args:
            tool: BaseTool instance from audio_agent.tools
        """
        self._tool = tool
        self._call_count = 0
        self._total_latency = 0.0
    
    @property
    def spec(self) -> ToolSpec:
        """Return tool specification."""
        return self._tool.spec
    
    def invoke(self, request: ToolCallRequest) -> ToolResult:
        """
        Invoke the wrapped tool and track metrics.
        
        Args:
            request: Tool call request
            
        Returns:
            Tool result
        """
        import time
        
        start = time.time()
        result = self._tool.invoke(request)
        latency = time.time() - start
        
        self._call_count += 1
        self._total_latency += latency
        
        return result
    
    def get_stats(self) -> dict:
        """Get tool usage statistics."""
        return {
            "calls": self._call_count,
            "total_latency": self._total_latency,
            "avg_latency": self._total_latency / max(1, self._call_count),
        }


def adapt_tool_for_evaluation(tool: AgentBaseTool) -> EvaluatedTool:
    """
    Adapt an audio_agent tool for evaluation.
    
    Args:
        tool: BaseTool instance
        
    Returns:
        EvaluatedTool wrapper
        
    Example:
        >>> from audio_agent.tools.dummy_tools import DummyASRTool
        >>> tool = DummyASRTool()
        >>> eval_tool = adapt_tool_for_evaluation(tool)
        >>> result = eval_tool.invoke(request)
    """
    return EvaluatedTool(tool)
