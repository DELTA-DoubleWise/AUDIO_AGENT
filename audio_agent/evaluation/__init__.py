"""
Tool evaluation module for AUDIO_AGENT.

Provides evaluation capabilities and RPS-based tool selection.
Integrates with audio_agent.tools for unified tool management.
"""

from audio_agent.evaluation.core.rps import RPSCalculator, RPSRegistry
from audio_agent.evaluation.core.evaluator import ToolEvaluator
from audio_agent.evaluation.adapter import adapt_tool_for_evaluation

__all__ = [
    "RPSCalculator",
    "RPSRegistry", 
    "ToolEvaluator",
    "adapt_tool_for_evaluation",
]
