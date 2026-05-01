"""
Benchmark runner for executing the audio agent on benchmark datasets.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add AUDIO_AGENT to path - fix for imports
project_root = Path(__file__).parent.parent
audio_agent_path = project_root / "AUDIO_AGENT"
sys.path.insert(0, str(audio_agent_path))

from audio_agent.config.settings import AgentConfig
from audio_agent.core.constants import AgentStatus
from audio_agent.frontend.openai_compatible_frontend import OpenAICompatibleFrontend
from audio_agent.frontend.gemini_frontend import GeminiFrontend
from audio_agent.main import AudioAgent
from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner
from audio_agent.planner.gemini_planner import GeminiPlanner
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.tools.catalog import register_all_mcp_tools


class BenchmarkRunner:
    """
    Runner for executing audio agent on benchmark datasets.
    
    This class wraps the AudioAgent with API-based components and provides
    a simple interface for running inference on benchmark samples.
    
    Example:
        runner = BenchmarkRunner(
            frontend_model="qwen3-omni-flash",
            planner_model="qwen3.5-plus",
            api_key="sk-xxx",
            max_steps=10,
        )
        
        result = await runner.run_single(
            question="What is in the audio?",
            audio_paths=["/path/to/audio.wav"],
        )
    """
    
    def __init__(
        self,
        frontend_model: str = "qwen3-omni-flash",
        planner_model: str = "qwen3.5-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_steps: int = 10,
        enable_thinking: bool = False,
        enable_mcp_tools: bool = True,
        tool_names: Optional[list[str]] = None,
        debug: bool = False,
        enable_run_logging: bool = True,
        log_dir: str = "./logs",
        use_gemini: bool = False,
        gemini_api_key: Optional[str] = None,
        gemini_base_url: str = "https://runway.devops.rednote.life/openai/google/v1:generateContent",
        planner_backend: str = "openai",
        gemini_planner_api_key: Optional[str] = None,
        gemini_planner_base_url: Optional[str] = None,
        use_dual_frontend: bool = False,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            frontend_model: Model name for frontend (API-based).
            planner_model: Model name for planner (API-based).
            api_key: API key. If None, reads from DASHSCOPE_API_KEY or OPENAI_API_KEY env var.
            base_url: API base URL.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            max_steps: Maximum agent steps.
            enable_thinking: Enable thinking mode for planner.
            enable_mcp_tools: Whether to register MCP tools.
            tool_names: Specific tools to register (None = all available).
            debug: Enable debug logging.
            enable_run_logging: Whether to enable run logging to files.
            log_dir: Directory for run logs (default: "./logs").
            use_dual_frontend: Enable dual frontend calls (verifier + observer).
        """
        self.frontend_model = frontend_model
        self.planner_model = planner_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.enable_thinking = enable_thinking
        self.enable_mcp_tools = enable_mcp_tools
        self.tool_names = tool_names
        self.debug = debug
        self.enable_run_logging = enable_run_logging
        self.log_dir = log_dir
        self.use_gemini = use_gemini
        self.gemini_api_key = gemini_api_key
        self.gemini_base_url = gemini_base_url
        self.planner_backend = planner_backend
        self.gemini_planner_api_key = gemini_planner_api_key
        self.gemini_planner_base_url = gemini_planner_base_url
        self.use_dual_frontend = use_dual_frontend
        
        # Validate planner_backend
        if self.planner_backend not in ("openai", "gemini"):
            raise ValueError(
                f"Invalid planner_backend: {self.planner_backend}. "
                "Must be 'openai' or 'gemini'."
            )
        
        # Get API key for frontend
        if self.use_gemini:
            self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
            self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if not self.gemini_api_key:
                raise ValueError(
                    "Gemini API key required for Gemini frontend. "
                    "Provide via gemini_api_key parameter or set GEMINI_API_KEY environment variable."
                )
        else:
            self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "API key required. Provide via api_key parameter or set "
                    "DASHSCOPE_API_KEY / OPENAI_API_KEY environment variable."
                )
        
        # Get API key for Gemini planner (if used)
        if self.planner_backend == "gemini":
            self.gemini_planner_api_key = (
                gemini_planner_api_key
                or os.environ.get("GEMINI_API_KEY")
            )
            if not self.gemini_planner_api_key:
                raise ValueError(
                    "Gemini planner API key required. "
                    "Provide via gemini_planner_api_key parameter or set GEMINI_API_KEY environment variable."
                )
        
        self.base_url = base_url
        
        # Initialize components
        self.frontend: Optional[Any] = None
        self.planner: Optional[Any] = None
        self.registry: Optional[ToolRegistry] = None
        self.fuser: Optional[DefaultEvidenceFuser] = None
        self.agent: Optional[AudioAgent] = None
        self.server_manager: Optional[MCPServerManager] = None
        
    async def initialize(self):
        """
        Initialize the agent components.
        
        This must be called before running any samples.
        """
        # Create components
        if self.use_gemini:
            self.frontend = GeminiFrontend(
                api_key=self.gemini_api_key,
                base_url=self.gemini_base_url,
                temperature=0.01,
                max_tokens=40960,
            )
        else:
            self.frontend = OpenAICompatibleFrontend(
                model=self.frontend_model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        
        # Create planner based on backend selection
        if self.planner_backend == "gemini":
            self.planner = GeminiPlanner(
                api_key=self.gemini_planner_api_key,
                base_url=self.gemini_planner_base_url or self.gemini_base_url,
                model_name=self.planner_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            self.planner = OpenAICompatiblePlanner(
                model=self.planner_model,
                api_key=self.api_key,
                base_url=self.base_url,
                enable_thinking=self.enable_thinking,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        
        self.registry = ToolRegistry()
        self.fuser = DefaultEvidenceFuser()
        
        config = AgentConfig(
            max_steps=self.max_steps,
            debug=self.debug,
            enable_run_logging=self.enable_run_logging,
            log_dir=self.log_dir,
            use_dual_frontend=self.use_dual_frontend,
        )
        
        # Set up MCP tools if enabled
        if self.enable_mcp_tools:
            self.server_manager = MCPServerManager()
            await register_all_mcp_tools(
                registry=self.registry,
                server_manager=self.server_manager,
                tool_names=self.tool_names,
                verbose=False,
            )
        
        # Create agent
        self.agent = AudioAgent(
            frontend=self.frontend,
            planner=self.planner,
            registry=self.registry,
            fuser=self.fuser,
            config=config,
        )
    
    async def run_single(
        self,
        question: str,
        audio_paths: list[str],
        run_log_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run the agent on a single sample.
        
        Args:
            question: The question to ask.
            audio_paths: List of paths to audio files.
            
        Returns:
            Dictionary containing:
            - success: Whether the agent completed successfully
            - answer: The final answer (if successful)
            - status: The agent status
            - step_count: Number of steps taken
            - error: Error message (if failed)
            - tool_calls: Number of tool calls made
        """
        if self.agent is None:
            raise RuntimeError("Runner not initialized. Call initialize() first.")
        
        try:
            # Run the agent
            final_state = await self.agent.arun(
                question=question,
                audio_paths=audio_paths,
                max_steps=self.max_steps,
                run_log_name=run_log_name,
            )
            
            # Extract results
            status = final_state.get("status", AgentStatus.RUNNING)
            final_answer = final_state.get("final_answer")
            step_count = final_state.get("step_count", 0)
            tool_history = final_state.get("tool_call_history", [])
            error_message = final_state.get("error_message")
            
            result = {
                "success": status == AgentStatus.ANSWERED,
                "status": status.value,
                "answer": final_answer.answer if final_answer else None,
                "confidence": final_answer.confidence if final_answer else 0.0,
                "step_count": step_count,
                "tool_calls": len(tool_history),
                "error": error_message,
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "answer": None,
                "confidence": 0.0,
                "step_count": 0,
                "tool_calls": 0,
                "error": str(e),
            }
    
    async def cleanup(self):
        """
        Clean up resources.
        
        Call this when done with the runner.
        """
        if self.server_manager:
            await self.server_manager.shutdown_all()
        
        if self.agent:
            self.agent.cleanup()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
