"""
Demo script for evaluation module integration.

Shows how to:
1. Evaluate tools on benchmark datasets
2. Record RPS scores
3. Recommend best tools
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audio_agent.evaluation import RPSCalculator, RPSRegistry, ToolEvaluator
from audio_agent.tools.dummy_tools import DummyASRTool
from audio_agent.evaluation.adapter import adapt_tool_for_evaluation


def demo_rps_calculation() -> None:
    """Demo RPS calculation."""
    print("=" * 60)
    print("Demo 1: RPS Calculation")
    print("=" * 60)
    
    calc = RPSCalculator()
    
    # Simulate evaluation results
    datasets = ["aishell1", "aishell5", "librispeech_clean", "iemocap"]
    
    print("\nSOTA Baselines:")
    for dataset in datasets:
        sota = calc.get_sota(dataset)
        if sota:
            print(f"  {dataset}: {sota.score} {sota.metric}")
    
    print("\nTool A Performance:")
    tool_a_scores = {
        "aishell1": 0.85,  # Worse than SOTA
        "aishell5": 22.5,  # Better than SOTA
    }
    
    for dataset, score in tool_a_scores.items():
        rps = calc.calculate(dataset, score)
        sota = calc.get_sota(dataset)
        print(f"  {dataset}: {sota.metric}={score} -> RPS={rps:.3f}")
    
    print("\nInterpretation:")
    print("  RPS = 1.0: Matches SOTA")
    print("  RPS > 1.0: Beats SOTA")
    print("  RPS < 1.0: Below SOTA")


def demo_registry() -> None:
    """Demo RPS registry."""
    print("\n" + "=" * 60)
    print("Demo 2: RPS Registry")
    print("=" * 60)
    
    # Use temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        db_path = Path(f.name)
    
    try:
        registry = RPSRegistry(db_path)
        
        # Record some performances
        print("\nRecording performances:")
        
        perf1 = registry.record("gemini-2.5-pro", "aishell1", 0.85)
        print(f"  {perf1.tool_name} on {perf1.dataset}: RPS={perf1.rps:.3f}")
        
        perf2 = registry.record("whisper-large-v3", "aishell1", 0.78)
        print(f"  {perf2.tool_name} on {perf2.dataset}: RPS={perf2.rps:.3f}")
        
        perf3 = registry.record("gemini-2.5-pro", "librispeech_clean", 1.85)
        print(f"  {perf3.tool_name} on {perf3.dataset}: RPS={perf3.rps:.3f}")
        
        # Recommend best tool
        print("\nBest tool for aishell1:")
        best = registry.get_best_tool("aishell1")
        print(f"  {best}")
        
        # List tool performances
        print("\nGemini performances:")
        for perf in registry.list_tool_performances("gemini-2.5-pro"):
            print(f"  {perf.dataset}: RPS={perf.rps:.3f}")
    
    finally:
        os.unlink(db_path)


def demo_tool_adapter() -> None:
    """Demo tool adapter."""
    print("\n" + "=" * 60)
    print("Demo 3: Tool Adapter")
    print("=" * 60)
    
    # Create dummy tool
    dummy = DummyASRTool()
    
    # Wrap for evaluation
    evaluated = adapt_tool_for_evaluation(dummy)
    
    print(f"\nWrapped tool: {evaluated.spec.name}")
    print(f"Description: {evaluated.spec.description}")
    
    # Simulate invocation
    from audio_agent.core.schemas import ToolCallRequest
    
    request = ToolCallRequest(
        tool_name="dummy_asr",
        args={"audio_path": "test.wav"},
    )
    
    result = evaluated.invoke(request)
    
    print(f"\nInvocation result:")
    print(f"  Success: {result.success}")
    print(f"  Text: {result.output.get('text', '')[:50]}...")
    
    stats = evaluated.get_stats()
    print(f"\nStats: {stats}")


def main() -> int:
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" Audio Agent Evaluation Module Demo")
    print("=" * 70)
    
    try:
        demo_rps_calculation()
        demo_registry()
        demo_tool_adapter()
        
        print("\n" + "=" * 70)
        print(" All demos completed successfully!")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
