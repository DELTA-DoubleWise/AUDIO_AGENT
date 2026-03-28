"""
Dummy planner implementation for testing and development.

Uses simple deterministic rules instead of an LLM.
"""

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import (
    InitialPlan,
    PlannerDecision,
    PlannerActionType,
    ToolSpec,
)
from audio_agent.core.errors import PlannerError
from audio_agent.planner.base import BasePlanner


class DummyPlanner(BasePlanner):
    """
    Dummy planner with deterministic behavior.
    
    Logic:
    - plan(question): produce a simple deterministic initial plan
    - decide(...):
      1. If no tools have been called yet, call "dummy_asr"
      2. After one tool call, ANSWER using accumulated evidence
      3. If step_count >= max_steps, FAIL
    """
    
    @property
    def name(self) -> str:
        return "dummy_planner"

    def plan(self, question: str) -> InitialPlan:
        """Produce deterministic initial plan from question only."""
        question = self.validate_question(question)
        return InitialPlan(
            approach="Start with speech transcription, then synthesize direct evidence for the question.",
            focus_points=[
                "Identify question-relevant speech content",
                "Keep answer grounded in observed audio evidence",
            ],
            possible_tool_types=["asr", "event_detection"],
            notes=f"[DummyPlanner] Initial plan generated for question: {question}",
        )
    
    def decide(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> PlannerDecision:
        """Make a deterministic decision based on simple rules."""
        self.validate_state(state)
        
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 10)
        tool_history = state.get("tool_call_history", [])
        evidence_log = state.get("evidence_log", [])
        initial_plan: InitialPlan = state["initial_plan"]
        
        # Check step limit first
        if step_count >= max_steps:
            return PlannerDecision(
                action=PlannerActionType.FAIL,
                rationale=f"Exceeded maximum steps ({max_steps})",
                confidence=1.0,
            )
        
        # Build tool name lookup
        available_tool_names = {t.name for t in available_tools}

        # Decision logic based on tool call count
        num_tools_called = len(tool_history)
        
        if num_tools_called == 0:
            # First iteration: call ASR tool if available
            target_tool = "dummy_asr"
            if target_tool not in available_tool_names:
                return PlannerDecision(
                    action=PlannerActionType.FAIL,
                    rationale=f"Required tool '{target_tool}' not available",
                    confidence=1.0,
                )
            return PlannerDecision(
                action=PlannerActionType.CALL_TOOL,
                rationale=(
                    "Initial plan indicates transcription-first strategy; "
                    "call ASR to gather direct textual evidence."
                ),
                selected_tool_name=target_tool,
                selected_tool_args={"audio_path": state.get("audio_path_or_uri", "")},
                confidence=0.8,
            )

        else:
            # After first tool call, answer
            return self._build_answer_decision(state, evidence_log, initial_plan)
    
    def _build_answer_decision(
        self,
        state: AgentState,
        evidence_log: list,
        initial_plan: InitialPlan,
    ) -> PlannerDecision:
        """Build an ANSWER decision from accumulated evidence."""
        question = state.get("question", "")
        
        # Collect evidence summaries
        evidence_texts = []
        for item in evidence_log:
            evidence_texts.append(f"- {item.source}: {item.content[:100]}...")
        
        evidence_summary = "\n".join(evidence_texts) if evidence_texts else "No evidence collected"
        
        draft_answer = (
            f"Based on the analysis of the audio:\n\n"
            f"Question: {question}\n\n"
            f"Initial plan approach: {initial_plan.approach}\n\n"
            f"Evidence collected:\n{evidence_summary}\n\n"
            f"[DummyPlanner] This is a placeholder answer. "
            f"A real planner would synthesize evidence into a coherent response."
        )
        
        return PlannerDecision(
            action=PlannerActionType.ANSWER,
            rationale="Sufficient evidence collected from multiple tools",
            draft_answer=draft_answer,
            confidence=0.75,
        )

    def answer(self, state: AgentState) -> str:
        """Generate dummy final answer from accumulated evidence."""
        question = state.get("question", "")
        evidence_count = len(state.get("evidence_log", []))
        return (
            f"[Dummy Answer] Based on {evidence_count} evidence items for question: {question}\n\n"
            "The audio has been analyzed but this is a placeholder answer."
        )

    def clarify_intent(self, state: AgentState) -> tuple[str, str | None]:
        """Clarify intent using dummy values."""
        question = state.get("question", "")
        return (
            f"Understand the audio content related to: {question}",
            "concise answer",
        )
