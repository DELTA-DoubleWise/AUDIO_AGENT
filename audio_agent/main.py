"""
Main entry point for the audio agent framework.

Provides convenience functions for running the agent.
"""

from audio_agent.core.state import AgentState, create_initial_state
from audio_agent.core.constants import AgentStatus
from audio_agent.core.schemas import FinalAnswer
from audio_agent.core.logging import setup_logger, set_debug_mode
from audio_agent.config.settings import AgentConfig
from audio_agent.graph.builder import build_graph
from audio_agent.frontend.base import BaseFrontend
from audio_agent.planner.base import BasePlanner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.fusion.base import BaseEvidenceFuser


class AudioAgent:
    """
    Main audio agent class.
    
    Encapsulates the LangGraph workflow and provides a clean interface
    for running the agent on audio queries.
    """
    
    def __init__(
        self,
        frontend: BaseFrontend,
        planner: BasePlanner,
        registry: ToolRegistry,
        fuser: BaseEvidenceFuser,
        config: AgentConfig | None = None,
    ) -> None:
        """
        Initialize the audio agent.
        
        Args:
            frontend: Frontend for initial audio processing
            planner: Planner for decision making
            registry: Tool registry with available tools
            fuser: Evidence fuser for tool results
            config: Optional configuration
        """
        if frontend is None:
            raise ValueError("frontend cannot be None")
        if planner is None:
            raise ValueError("planner cannot be None")
        if registry is None:
            raise ValueError("registry cannot be None")
        if fuser is None:
            raise ValueError("fuser cannot be None")
        
        self.frontend = frontend
        self.planner = planner
        self.registry = registry
        self.fuser = fuser
        self.config = config or AgentConfig()
        
        # Set up logging
        setup_logger()
        if self.config.debug:
            set_debug_mode(True)
        
        # Build the graph
        self._graph = build_graph(frontend, planner, registry, fuser)
    
    def run(
        self,
        question: str,
        audio_path_or_uri: str,
        max_steps: int | None = None,
    ) -> AgentState:
        """
        Run the agent on an audio query (synchronous).
        
        Note: If using MCP tools, use arun() instead.
        
        Args:
            question: User question about the audio
            audio_path_or_uri: Path or URI to audio file
            max_steps: Override default max_steps
        
        Returns:
            Final agent state with answer or error
        """
        effective_max_steps = max_steps if max_steps is not None else self.config.max_steps
        
        initial_state = create_initial_state(
            question=question,
            audio_path_or_uri=audio_path_or_uri,
            max_steps=effective_max_steps,
        )
        
        # Execute the graph
        final_state = self._graph.invoke(initial_state)
        
        return final_state
    
    async def arun(
        self,
        question: str,
        audio_path_or_uri: str,
        max_steps: int | None = None,
    ) -> AgentState:
        """
        Run the agent on an audio query (asynchronous).
        
        Required when using MCP tools or async tool execution.
        
        Args:
            question: User question about the audio
            audio_path_or_uri: Path or URI to audio file
            max_steps: Override default max_steps
        
        Returns:
            Final agent state with answer or error
        """
        effective_max_steps = max_steps if max_steps is not None else self.config.max_steps
        
        initial_state = create_initial_state(
            question=question,
            audio_path_or_uri=audio_path_or_uri,
            max_steps=effective_max_steps,
        )
        
        # Execute the graph asynchronously
        final_state = await self._graph.ainvoke(initial_state)
        
        return final_state
    
    def get_answer(self, state: AgentState) -> FinalAnswer | None:
        """Extract the final answer from a completed state."""
        return state.get("final_answer")
    
    def get_status(self, state: AgentState) -> AgentStatus:
        """Extract the status from a state."""
        return state.get("status", AgentStatus.RUNNING)
    
    def is_successful(self, state: AgentState) -> bool:
        """Check if the agent completed successfully with an answer."""
        return state.get("status") == AgentStatus.ANSWERED


def create_dummy_agent(config: AgentConfig | None = None) -> AudioAgent:
    """
    Create an audio agent with dummy components for testing.
    
    Args:
        config: Optional configuration
    
    Returns:
        AudioAgent instance with dummy components
    """
    from audio_agent.frontend.dummy_frontend import DummyFrontend
    from audio_agent.planner.dummy_planner import DummyPlanner
    from audio_agent.tools.dummy_tools import DummyASRTool, DummyAudioEventDetectorTool
    from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
    
    frontend = DummyFrontend()
    planner = DummyPlanner()
    registry = ToolRegistry()
    registry.register(DummyASRTool())
    registry.register(DummyAudioEventDetectorTool())
    fuser = DefaultEvidenceFuser()
    
    return AudioAgent(
        frontend=frontend,
        planner=planner,
        registry=registry,
        fuser=fuser,
        config=config,
    )


def create_openai_planner(
    model: str = "qwen3.5-plus",
    api_key: str | None = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    enable_thinking: bool = False,
    **kwargs,
) -> "OpenAICompatiblePlanner":
    """
    Create an OpenAI-compatible API planner.
    
    Works with qwen3.5-plus, kimi-k2.5, OpenAI models, and any other
    OpenAI-compatible API endpoint.
    
    Args:
        model: Model name (e.g., "qwen3.5-plus", "kimi-k2.5", "gpt-4")
        api_key: API key. If None, reads from DASHSCOPE_API_KEY or OPENAI_API_KEY env var.
        base_url: API base URL
        enable_thinking: Enable thinking mode for models that support it (qwen3.5-plus)
        **kwargs: Additional arguments passed to OpenAICompatiblePlanner
    
    Returns:
        Configured OpenAICompatiblePlanner instance
    
    Example:
        # qwen3.5-plus with thinking
        planner = create_openai_planner(
            model="qwen3.5-plus",
            api_key="sk-xxx",
            enable_thinking=True,
        )
        
        # kimi-k2.5
        planner = create_openai_planner(
            model="kimi-k2.5",
            api_key="sk-xxx",
        )
        
        # Using environment variable for API key
        import os
        os.environ["DASHSCOPE_API_KEY"] = "sk-xxx"
        planner = create_openai_planner(model="qwen3.5-plus")
    """
    from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner
    
    return OpenAICompatiblePlanner(
        model=model,
        api_key=api_key,
        base_url=base_url,
        enable_thinking=enable_thinking,
        **kwargs,
    )
