#!/usr/bin/env python3
"""
Demo script with API-based planner and auto-registered MCP tools.

This script demonstrates the complete workflow using:
- Qwen2-Audio frontend (local)
- OpenAI-compatible API planner (qwen3.5-plus, kimi-k2.5, etc.)
- All available MCP tools auto-registered from catalog

Usage:
    # With explicit API key
    python -m audio_agent.examples.demo_run_api_planner \
        --audio /path/to/audio.wav \
        --question "What is being said?" \
        --api-key "sk-xxx"

    # With environment variable
    export DASHSCOPE_API_KEY="sk-xxx"
    python -m audio_agent.examples.demo_run_api_planner \
        --audio /path/to/audio.wav \
        --question "What is being said?"

    # Using kimi-k2.5
    python -m audio_agent.examples.demo_run_api_planner \
        --audio /path/to/audio.wav \
        --question "What is being said?" \
        --planner-model "kimi-k2.5"
"""

from __future__ import annotations

import argparse
import asyncio
import os
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
from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.tools.catalog import register_all_mcp_tools, list_available_tools
from audio_agent.utils.model_downloader import DEFAULT_QWEN2_AUDIO_PATH


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
        print(f"    Content: {item.content}")


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
    if initial_plan.clarified_intent:
        print(f"Clarified Intent: {initial_plan.clarified_intent}")
    if initial_plan.expected_output_format:
        print(f"Expected Output Format: {initial_plan.expected_output_format}")
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
        description="Run the audio agent demo with API-based planner and auto-registered MCP tools.")
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
        "--planner-model",
        default="qwen3.5-plus",
        help="API model name for planner (default: qwen3.5-plus).",
    )
    parser.add_argument(
        "--api-key",
        default="sk-f8ae3fc37bdd4953977e813f77b7324f",
        help="API key. If not provided, reads from DASHSCOPE_API_KEY or OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="API base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for models that support it (e.g., qwen3.5-plus).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of agent steps.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help="Specific tools to register (default: all available tools)",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit",
    )
    return parser


async def amain() -> int:
    """Run the demo (async version)."""
    parser = build_parser()
    
    # Handle --list-tools before parsing all args (since --audio and --question are required)
    import sys
    if "--list-tools" in sys.argv:
        print("Available MCP tools:")
        for tool in list_available_tools():
            print(f"  - {tool}")
        return 0
    
    args = parser.parse_args()
    
    print_separator("Audio Agent Framework Demo (API Planner + Auto-registered Tools)")
    print("\nThis demo runs the agent with:")
    print("- Qwen2-Audio frontend (local)")
    print(f"- {args.planner_model} planner (API-based)")
    print("- Auto-registered MCP tools from catalog\n")
    
    # Get API key from args or environment
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: API key required. Provide via --api-key or set DASHSCOPE_API_KEY / OPENAI_API_KEY environment variable.")
        return 1
    
    # Set up logging
    setup_logger()
    set_debug_mode(True)
    
    # Create configuration
    config = AgentConfig(
        max_steps=args.max_steps,
        debug=True,
    )
    
    # Create the agent components
    print(f"Creating audio agent with Qwen2-Audio frontend and {args.planner_model} planner...")
    try:
        frontend = Qwen2AudioFrontend(model_path=args.frontend_model_path)
        planner = OpenAICompatiblePlanner(
            model=args.planner_model,
            api_key=api_key,
            base_url=args.base_url,
            enable_thinking=args.enable_thinking,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        registry = ToolRegistry()
        fuser = DefaultEvidenceFuser()
        
        # Set up MCP server manager
        server_manager = MCPServerManager()
        
        # Auto-register all MCP tools from catalog
        print("\nAuto-registering MCP tools from catalog...")
        registered_tools = await register_all_mcp_tools(
            registry=registry,
            server_manager=server_manager,
            tool_names=args.tools,  # None = all tools, or specify list
            verbose=True,
        )
        
        if not registered_tools:
            print("\nWarning: No MCP tools were registered!")
            print("Make sure tools are set up:")
            print("  cd audio_agent/tools/catalog/<tool_name> && ./setup.sh")
        
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
    print(f"Planner model: {args.planner_model} (API)")
    print(f"Enable thinking: {args.enable_thinking}")
    
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
    
    # Print final clarified intent (may differ from initial if clarification occurred)
    clarified_intent = final_state.get("clarified_intent")
    expected_format = final_state.get("expected_output_format")
    if clarified_intent or expected_format:
        print_separator("Clarified Intent")
        if clarified_intent:
            print(f"\nIntent: {clarified_intent}")
        if expected_format:
            print(f"Expected Format: {expected_format}")
    
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
