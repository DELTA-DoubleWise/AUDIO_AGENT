#!/usr/bin/env python3
"""
Demo script with real ASR tool using Qwen3-ASR-1.7B.

This script demonstrates the complete workflow using:
- Qwen2-Audio frontend
- Qwen2.5 planner
- Real ASR tool (MCP-based Qwen3-ASR-1.7B)

Usage:
    python -m audio_agent.examples.demo_run_real_asr \
        --audio /path/to/audio.wav \
        --question "What is being said?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path for direct execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from audio_agent.config.settings import AgentConfig
from audio_agent.core.constants import AgentStatus
from audio_agent.core.logging import setup_logger, set_debug_mode
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
from audio_agent.frontend.qwen2_audio_frontend import Qwen2AudioFrontend
from audio_agent.main import AudioAgent
from audio_agent.planner.qwen25_planner import Qwen25Planner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.mcp import MCPServerManager, MCPToolAdapter
from audio_agent.tools.catalog import load_mcp_server_config
from audio_agent.utils.model_downloader import (
    DEFAULT_QWEN2_AUDIO_PATH,
    DEFAULT_QWEN25_PATH,
)


def print_separator(title: str = "") -> None:
    """Print a visual separator."""
    line = "=" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}")
    else:
        print(line)


def print_evidence_log(evidence_log: list) -> None:
    """Print the evidence log in a readable format."""
    print_separator("Evidence Log")
    for i, item in enumerate(evidence_log, 1):
        print(f"\n[{i}] Source: {item.source}")
        print(f"    Type: {item.evidence_type}")
        print(f"    Confidence: {item.confidence:.2f}")
        print(f"    Content: {item.content[:200]}...")


def print_tool_history(tool_history: list) -> None:
    """Print the tool call history."""
    print_separator("Tool Call History")
    for i, record in enumerate(tool_history, 1):
        req = record.request
        res = record.result
        print(f"\n[{i}] Tool: {req.tool_name}")
        print(f"    Step: {record.step_number}")
        print(f"    Args: {req.args}")
        print(f"    Success: {res.success}")
        print(f"    Execution time: {res.execution_time_ms:.2f}ms")


def print_planner_trace(planner_trace: list) -> None:
    """Print the planner decision trace."""
    print_separator("Planner Trace")
    for i, decision in enumerate(planner_trace, 1):
        print(f"\n[{i}] Action: {decision.action.value}")
        print(f"    Rationale: {decision.rationale}")
        if decision.selected_tool_name:
            print(f"    Tool: {decision.selected_tool_name}")
        print(f"    Confidence: {decision.confidence:.2f}")


def print_initial_plan(initial_plan) -> None:
    """Print the initial plan."""
    print_separator("Initial Plan")
    print(f"\nApproach: {initial_plan.approach}")
    if initial_plan.focus_points:
        print("Focus points:")
        for item in initial_plan.focus_points:
            print(f"  - {item}")
    if initial_plan.possible_tool_types:
        print("Possible tool types:")
        for item in initial_plan.possible_tool_types:
            print(f"  - {item}")
    if initial_plan.notes:
        print(f"Notes: {initial_plan.notes}")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the demo."""
    parser = argparse.ArgumentParser(
        description="Run the audio agent demo with real ASR tool.")
    parser.add_argument(
        "--audio",
        required=True,
        help="Path or URI to the input audio file.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the audio.",
    )
    parser.add_argument(
        "--frontend-model-path",
        default=DEFAULT_QWEN2_AUDIO_PATH,
        help=f"Model path for Qwen2-Audio frontend (default: {DEFAULT_QWEN2_AUDIO_PATH}).",
    )
    parser.add_argument(
        "--planner-model-path",
        default=DEFAULT_QWEN25_PATH,
        help=f"Model path for Qwen2.5 planner (default: {DEFAULT_QWEN25_PATH}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of agent steps.",
    )
    parser.add_argument(
        "--asr-device",
        default="auto",
        help="Device for ASR tool (auto, cuda, cpu).",
    )
    return parser


async def setup_asr_qwen3_tool(registry: ToolRegistry) -> MCPServerManager:
    """
    Set up the Qwen3-ASR-1.7B tool from catalog.
    
    Args:
        registry: Tool registry to register tools to
        
    Returns:
        Server manager for cleanup
    """
    # Create server manager
    server_manager = MCPServerManager()
    
    # Load ASR config from catalog
    asr_config = load_mcp_server_config("asr_qwen3")
    server_manager.register_config("asr_qwen3", asr_config)
    
    # Get client and discover tools
    print("Starting ASR MCP server and discovering tools...")
    client = await server_manager.get_client("asr_qwen3")
    tools = await client.list_tools()
    
    # Register each tool from the server
    for tool_info in tools:
        adapter = MCPToolAdapter(
            server_name="asr_qwen3",
            tool_info=tool_info,
            server_manager=server_manager,
        )
        registry.register_mcp(adapter)
        print(f"  Registered MCP tool: {tool_info.name}")
    
    return server_manager


async def amain() -> int:
    """Run the demo (async version)."""
    args = build_parser().parse_args()

    print_separator("Audio Agent Framework Demo (Real ASR)")
    print("\nThis demo runs the agent with:")
    print("- Qwen2-Audio frontend")
    print("- Qwen2.5 planner")
    print("- Qwen3-ASR-1.7B tool (MCP-based)\n")
    
    # Set up logging
    setup_logger()
    set_debug_mode(True)
    
    # Create configuration
    config = AgentConfig(
        max_steps=args.max_steps,
        debug=True,
    )
    
    # Create the agent components
    print("Creating audio agent with Qwen2-Audio frontend and Qwen2.5 planner...")
    try:
        frontend = Qwen2AudioFrontend(model_path=args.frontend_model_path)
        planner = Qwen25Planner(model_path=args.planner_model_path)
        registry = ToolRegistry()
        fuser = DefaultEvidenceFuser()
        
        # Set up ASR tool
        print("Setting up Qwen3-ASR-1.7B tool...")
        server_manager = await setup_asr_qwen3_tool(registry)
        
        # Create agent
        agent = AudioAgent(
            frontend=frontend,
            planner=planner,
            registry=registry,
            fuser=fuser,
            config=config,
        )
        
    except Exception as e:
        print(f"\nFailed to initialize agent: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    question = args.question
    # Resolve audio path to absolute path for tools
    audio_paths = [str(Path(args.audio).resolve())]

    print(f"\nQuestion: {question}")
    print(f"Audio path: {audio_paths[0]}")
    print(f"Frontend model: {args.frontend_model_path}")
    print(f"Planner model: {args.planner_model_path}")
    
    print_separator("Running Agent")
    
    # Run the agent asynchronously
    try:
        final_state = await agent.arun(
            question=question,
            audio_paths=audio_paths,
            max_steps=args.max_steps,
        )
    except Exception as e:
        print(f"\nAgent failed with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up MCP servers
        print("\nShutting down MCP servers...")
        await server_manager.shutdown_all()
     
    # Print results
    print_separator("Results")
    
    status = final_state.get("status", AgentStatus.RUNNING)
    print(f"\nFinal Status: {status.value}")
    print(f"Step Count: {final_state.get('step_count', 0)}")

    frontend_output = final_state.get("initial_frontend_output")
    if frontend_output:
        print_separator("Frontend Caption")
        print(f"\n{frontend_output.question_guided_caption}")

    initial_plan = final_state.get("initial_plan")
    if initial_plan:
        print_initial_plan(initial_plan)
    
    # Print evidence log
    evidence_log = final_state.get("evidence_log", [])
    if evidence_log:
        print_evidence_log(evidence_log)
    
    # Print tool history
    tool_history = final_state.get("tool_call_history", [])
    if tool_history:
        print_tool_history(tool_history)
    
    # Print planner trace
    planner_trace = final_state.get("planner_trace", [])
    if planner_trace:
        print_planner_trace(planner_trace)
    
    # Print final answer
    final_answer = final_state.get("final_answer")
    if final_answer:
        print_separator("Final Answer")
        print(f"\n{final_answer.answer}")
        print(f"\nConfidence: {final_answer.confidence:.2f}")
        
        # Print output audio if present
        if final_answer.output_audio:
            print(f"\nOutput Audio:")
            print(f"  ID: {final_answer.output_audio.audio_id}")
            print(f"  Path: {final_answer.output_audio.path}")
            print(f"  Description: {final_answer.output_audio.description}")
    
    # Print error if any
    error_message = final_state.get("error_message")
    if error_message:
        print_separator("Error")
        print(f"\n{error_message}")
    
    print_separator()
    
    if status == AgentStatus.ANSWERED:
        print("\nDemo completed successfully.")
        return 0
    else:
        print(f"\nDemo completed with status: {status.value}")
        return 1


def main() -> int:
    """Run the demo (entry point)."""
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
