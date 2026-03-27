"""
Model-backed planner base classes.

This module mirrors the structure of the model-backed frontend:
- explicit unified input schema
- explicit input format dispatch
- template-method hooks for model initialization and invocation
- strict output parsing and normalization
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
import json
from typing import Any

from pydantic import BaseModel, Field

from audio_agent.core.errors import PlannerError
from audio_agent.core.schemas import InitialPlan, PlannerDecision, ToolSpec
from audio_agent.core.state import AgentState
from audio_agent.planner.base import BasePlanner
from audio_agent.utils.model_io import parse_json_object_text, validate_message_sequence


DEFAULT_PLAN_SYSTEM_PROMPT = (
    "You are the planning module for an audio agent. "
    "Given only the user question, produce an initial high-level plan. "
    "Do not answer the question yet. "
    "Return only a JSON object matching the required InitialPlan schema."
)

DEFAULT_DECISION_SYSTEM_PROMPT = (
    "You are the action-decision planner for an audio agent. "
    "Given the question, frontend caption, initial plan, accumulated evidence, "
    "tool history, and available tools, decide the next concrete action. "
    "Keep suspecting that the initial front-end caption may be hallucinated and adapt as needed based on evidence and tool results. "
    "Return only a JSON object matching the required PlannerDecision schema."
)

DEFAULT_CLARIFY_INTENT_SYSTEM_PROMPT = (
    "You are the intent clarification module for an audio agent. "
    "Given the original question and all accumulated evidence, "
    "clarify what the user is actually asking and what output format they expect. "
    "Use reasoning on the evidence to refine or clarify the intent. "
    "Do not call tools - if tools were needed, they should have been called before this step."
)


class PlannerInputFormat(str, Enum):
    """Supported planner backend input modes."""

    API_MODEL = "api_model"
    LOCAL_MODEL = "local_model"


class UnifiedPlannerInput(BaseModel):
    """Backend-agnostic planner input wrapper."""

    system_prompt: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    user_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class BaseModelPlanner(BasePlanner):
    """
    Template-method base for model-backed planners.

    Concrete subclasses mainly implement:
    - initialize_model()
    - call_model()
    """

    def __init__(
        self,
        plan_system_prompt: str | None = None,
        decision_system_prompt: str | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.plan_system_prompt = (plan_system_prompt or DEFAULT_PLAN_SYSTEM_PROMPT).strip()
        self.decision_system_prompt = (
            decision_system_prompt or DEFAULT_DECISION_SYSTEM_PROMPT
        ).strip()
        if not self.plan_system_prompt:
            raise PlannerError("plan_system_prompt must be non-empty")
        if not self.decision_system_prompt:
            raise PlannerError("decision_system_prompt must be non-empty")
        self.model_config = model_config or {}
        self.model_handle = self.initialize_model()

    @property
    def input_format(self) -> PlannerInputFormat:
        """Default planner input mode."""
        return PlannerInputFormat.API_MODEL

    @abstractmethod
    def initialize_model(self) -> Any:
        """Initialize and return provider/model handle."""
        raise NotImplementedError

    @abstractmethod
    def call_model(self, model_input: UnifiedPlannerInput) -> Any:
        """Invoke model/backend and return raw output."""
        raise NotImplementedError

    def build_plan_system_prompt(self) -> str:
        """Build system prompt for initial planning phase."""
        return self.plan_system_prompt

    def build_plan_user_instruction(self, question: str) -> str:
        """Build user instruction for initial planning phase."""
        return (
            f"Question: {question}\n"
            "Produce an InitialPlan JSON object with keys: "
            "`approach` (str), `focus_points` (list[str]), "
            "`possible_tool_types` (list[str]), optional `notes` (str), "
            "`clarified_intent` (str | null) - what the question is asking, "
            "`expected_output_format` (str | null) - expected answer format. "
            "If the intent is unclear, express uncertainty in focus_points or notes."
        )

    def build_decision_system_prompt(self) -> str:
        """Build system prompt for action decision phase."""
        return self.decision_system_prompt

    def build_decision_user_instruction(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> str:
        """Build user instruction for action decision phase."""
        frontend_output = state["initial_frontend_output"]
        initial_plan = state["initial_plan"]
        evidence_log = state.get("evidence_log", [])
        tool_history = state.get("tool_call_history", [])

        evidence_summary = [
            {
                "source": item.source,
                "type": item.evidence_type,
                "content": item.content,
            }
            for item in evidence_log
        ]
        tool_history_summary = [
            {
                "tool_name": record.request.tool_name,
                "success": record.result.success,
                "output_keys": list(record.result.output.keys()),
            }
            for record in tool_history
        ]
        tool_summary = [
            {
                "name": tool.name,
                "description": tool.description,
                "tags": tool.tags,
            }
            for tool in available_tools
        ]

        payload = {
            "question": state["question"],
            "frontend_caption": frontend_output.question_guided_caption,
            "initial_plan": initial_plan.model_dump(mode="json"),
            "evidence_log": evidence_summary,
            "tool_call_history": tool_history_summary,
            "available_tools": tool_summary,
            "step_count": state.get("step_count", 0),
            "max_steps": state.get("max_steps", 10),
            "decision_rules": [
                "If you have enough evidence to answer the question, use action='answer' and provide draft_answer.",
                "If you need more information, use action='call_tool' and specify which tool in selected_tool_name.",
                "If the intent or expected output format is unclear, use action='clarify_intent' to reason about it.",
                "action='call_tool' REQUIRES a non-empty selected_tool_name - never leave it null or empty.",
                "action='answer' REQUIRES a non-empty draft_answer.",
                "action='clarify_intent' uses reasoning only - do not call tools.",
                "Do NOT use action='call_tool' if you are ready to answer - use action='answer' instead.",
            ],
            "required_output": {
                "action": "answer | call_tool | clarify_intent | fail",
                "rationale": "str - explain your decision",
                "selected_tool_name": "str | null - REQUIRED for call_tool, must be a valid tool name",
                "selected_tool_args": "dict - arguments for the tool when using call_tool",
                "draft_answer": "str | null - REQUIRED for answer, your final response to the question",
                "confidence": "float - 0.0 to 1.0",
            },
        }
        return json.dumps(payload, ensure_ascii=True)

    def build_api_model_input_for_plan(self, question: str) -> UnifiedPlannerInput:
        """Build API-style planner input for initial planning."""
        system_prompt = self.build_plan_system_prompt()
        user_text = self.build_plan_user_instruction(question)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="initial_plan",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "initial_plan"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.API_MODEL.value},
        )

    def build_local_model_input_for_plan(self, question: str) -> UnifiedPlannerInput:
        """Build local-text-model input for initial planning."""
        system_prompt = self.build_plan_system_prompt()
        user_text = self.build_plan_user_instruction(question)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="initial_plan",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "initial_plan"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.LOCAL_MODEL.value},
        )

    def build_api_model_input_for_decision(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> UnifiedPlannerInput:
        """Build API-style planner input for action decision."""
        system_prompt = self.build_decision_system_prompt()
        user_text = self.build_decision_user_instruction(state, available_tools)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="decision",
            question=state["question"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": state["question"], "task": "decision"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.API_MODEL.value},
        )

    def build_local_model_input_for_decision(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> UnifiedPlannerInput:
        """Build local-text-model input for action decision."""
        system_prompt = self.build_decision_system_prompt()
        user_text = self.build_decision_user_instruction(state, available_tools)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="decision",
            question=state["question"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": state["question"], "task": "decision"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.LOCAL_MODEL.value},
        )

    def _validate_built_model_input(self, model_input: UnifiedPlannerInput) -> None:
        """Fail-fast validation for planner model input."""
        if not model_input.system_prompt.strip():
            raise PlannerError("Malformed planner model input: empty system_prompt")
        if not model_input.task_type.strip():
            raise PlannerError("Malformed planner model input: empty task_type")
        if not model_input.question.strip():
            raise PlannerError("Malformed planner model input: empty question")
        validate_message_sequence(
            model_input.messages,
            error_cls=PlannerError,
            context="Malformed planner model input",
        )

    def build_plan_model_input(self, question: str) -> UnifiedPlannerInput:
        """Dispatch planner initial-plan input build by backend mode."""
        question = self.validate_question(question)
        mode = self.input_format
        if isinstance(mode, str):
            try:
                mode = PlannerInputFormat(mode)
            except ValueError as e:
                raise PlannerError(
                    "Unsupported planner input format",
                    details={"input_format": mode},
                ) from e
        elif not isinstance(mode, PlannerInputFormat):
            raise PlannerError(
                "Unsupported planner input format type",
                details={"input_format_type": type(mode).__name__},
            )

        if mode == PlannerInputFormat.API_MODEL:
            model_input = self.build_api_model_input_for_plan(question)
        elif mode == PlannerInputFormat.LOCAL_MODEL:
            model_input = self.build_local_model_input_for_plan(question)
        else:
            raise PlannerError(
                "Unsupported planner input format",
                details={"input_format": mode.value},
            )

        self._validate_built_model_input(model_input)
        return model_input

    def build_decision_model_input(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> UnifiedPlannerInput:
        """Dispatch planner decision input build by backend mode."""
        self.validate_state(state)
        mode = self.input_format
        if isinstance(mode, str):
            try:
                mode = PlannerInputFormat(mode)
            except ValueError as e:
                raise PlannerError(
                    "Unsupported planner input format",
                    details={"input_format": mode},
                ) from e
        elif not isinstance(mode, PlannerInputFormat):
            raise PlannerError(
                "Unsupported planner input format type",
                details={"input_format_type": type(mode).__name__},
            )

        if mode == PlannerInputFormat.API_MODEL:
            model_input = self.build_api_model_input_for_decision(state, available_tools)
        elif mode == PlannerInputFormat.LOCAL_MODEL:
            model_input = self.build_local_model_input_for_decision(state, available_tools)
        else:
            raise PlannerError(
                "Unsupported planner input format",
                details={"input_format": mode.value},
            )

        self._validate_built_model_input(model_input)
        return model_input

    def normalize_plan_output(self, raw_output: Any) -> InitialPlan:
        """Normalize model output into InitialPlan."""
        if isinstance(raw_output, InitialPlan):
            return raw_output
        if isinstance(raw_output, str):
            raw_output = parse_json_object_text(
                raw_output,
                error_cls=PlannerError,
                subject="Planner",
            )
        if isinstance(raw_output, dict):
            required = {"approach", "focus_points", "possible_tool_types"}
            keys = set(raw_output.keys())
            missing = sorted(required - keys)
            if missing:
                raise PlannerError(
                    "Malformed initial plan output: missing required fields",
                    details={"missing_fields": missing, "output_keys": sorted(keys)},
                )
            try:
                return InitialPlan(**raw_output)
            except Exception as e:
                raise PlannerError(
                    "Malformed initial plan output: schema validation failed",
                    details={"error": str(e), "output_keys": sorted(keys)},
                ) from e
        raise PlannerError(
            "Malformed initial plan output: expected dict, JSON text, or InitialPlan",
            details={"output_type": type(raw_output).__name__},
        )

    def normalize_decision_output(self, raw_output: Any) -> PlannerDecision:
        """Normalize model output into PlannerDecision."""
        if isinstance(raw_output, PlannerDecision):
            return raw_output
        if isinstance(raw_output, str):
            raw_output = parse_json_object_text(
                raw_output,
                error_cls=PlannerError,
                subject="Planner",
            )
        if isinstance(raw_output, dict):
            required = {"action", "rationale"}
            keys = set(raw_output.keys())
            missing = sorted(required - keys)
            if missing:
                raise PlannerError(
                    "Malformed planner decision output: missing required fields",
                    details={"missing_fields": missing, "output_keys": sorted(keys)},
                )
            try:
                return PlannerDecision(**raw_output)
            except Exception as e:
                raise PlannerError(
                    "Malformed planner decision output: schema validation failed",
                    details={"error": str(e), "output_keys": sorted(keys)},
                ) from e
        raise PlannerError(
            "Malformed planner decision output: expected dict, JSON text, or PlannerDecision",
            details={"output_type": type(raw_output).__name__},
        )

    def plan(self, question: str) -> InitialPlan:
        """Question-only initial planning phase."""
        question = self.validate_question(question)
        model_input = self.build_plan_model_input(question)
        try:
            raw_output = self.call_model(model_input)
        except PlannerError:
            raise
        except Exception as e:
            raise PlannerError(
                f"Planner model call failed during initial planning: {type(e).__name__}: {e}",
                details={"planner": self.name},
            ) from e
        return self.normalize_plan_output(raw_output)

    def decide(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> PlannerDecision:
        """Action decision phase using state + available tools."""
        self.validate_state(state)
        model_input = self.build_decision_model_input(state, available_tools)
        try:
            raw_output = self.call_model(model_input)
        except PlannerError:
            raise
        except Exception as e:
            raise PlannerError(
                f"Planner model call failed during decision phase: {type(e).__name__}: {e}",
                details={"planner": self.name},
            ) from e
        return self.normalize_decision_output(raw_output)

    def answer(self, state: AgentState) -> str:
        """Generate final answer using the model."""
        model_input = self.build_answer_model_input(state)
        try:
            raw_output = self.call_model(model_input)
        except PlannerError:
            raise
        except Exception as e:
            raise PlannerError(
                f"Planner model call failed during answer generation: {type(e).__name__}: {e}",
                details={"planner": self.name},
            ) from e

        # Treat output as plain text answer (could be JSON or string)
        if isinstance(raw_output, str):
            # Try to parse as JSON first (for structured answer)
            try:
                parsed = json.loads(raw_output)
                if isinstance(parsed, dict) and "answer" in parsed:
                    return parsed["answer"].strip()
            except json.JSONDecodeError:
                pass
            # Return as plain text
            return raw_output.strip()

        return str(raw_output).strip()

    def build_answer_model_input(self, state: AgentState) -> UnifiedPlannerInput:
        """Build model input for final answer generation."""
        question = state["question"]
        evidence_log = state.get("evidence_log", [])

        # Build evidence summary
        evidence_text = "\n".join(
            f"[{item.source}] {item.content}"
            for item in evidence_log
        )

        system_prompt = (
            "You are the final answer generator for an audio agent. "
            "Given the original question and all accumulated evidence, "
            "provide a comprehensive final answer. "
            "Synthesize all evidence to directly answer the question. "
            "Be concise but complete."
        )

        user_text = (
            f"Original Question: {question}\n\n"
            f"Accumulated Evidence:\n{evidence_text}\n\n"
            "Based on all the evidence above, provide your final answer to the question."
        )

        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="final_answer",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "final_answer"},
            metadata={"planner_name": self.name, "task_type": "final_answer"},
        )

    def build_clarify_intent_model_input(self, state: AgentState) -> UnifiedPlannerInput:
        """Build model input for intent clarification."""
        question = state["question"]
        evidence_log = state.get("evidence_log", [])
        clarified_intent = state.get("clarified_intent")
        expected_format = state.get("expected_output_format")

        # Build evidence summary
        evidence_text = "\n".join(
            f"[{item.source}] {item.content}"
            for item in evidence_log
        )

        system_prompt = DEFAULT_CLARIFY_INTENT_SYSTEM_PROMPT

        user_text_parts = [
            f"Original Question: {question}",
            f"Current Clarified Intent: {clarified_intent or 'Not yet clarified'}",
            f"Current Expected Format: {expected_format or 'Not yet specified'}",
            "",
            "Accumulated Evidence:",
            evidence_text if evidence_text else "No evidence yet.",
            "",
            "Based on the question and accumulated evidence, clarify the intent and expected output format.",
            "Return a JSON object with keys:",
            "  `clarified_intent` (str): What the question is actually asking",
            "  `expected_output_format` (str | null): Expected format (e.g., 'bullet points', 'single sentence', 'detailed paragraph')",
        ]
        user_text = "\n".join(user_text_parts)

        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="clarify_intent",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "clarify_intent"},
            metadata={"planner_name": self.name, "task_type": "clarify_intent"},
        )

    def normalize_clarify_intent_output(self, raw_output: Any) -> tuple[str, str | None]:
        """Normalize model output into (clarified_intent, expected_output_format)."""
        if isinstance(raw_output, tuple) and len(raw_output) == 2:
            return raw_output[0], raw_output[1]
        if isinstance(raw_output, str):
            raw_output = parse_json_object_text(
                raw_output,
                error_cls=PlannerError,
                subject="Planner",
            )
        if isinstance(raw_output, dict):
            clarified_intent = raw_output.get("clarified_intent")
            expected_format = raw_output.get("expected_output_format")
            if clarified_intent is None:
                raise PlannerError(
                    "Malformed clarify_intent output: missing clarified_intent",
                    details={"output_keys": sorted(raw_output.keys())},
                )
            return clarified_intent, expected_format
        raise PlannerError(
            "Malformed clarify_intent output: expected dict, JSON text, or tuple",
            details={"output_type": type(raw_output).__name__},
        )

    def clarify_intent(self, state: AgentState) -> tuple[str, str | None]:
        """
        Clarify the user's intent and expected output format.
        
        Uses reasoning on accumulated evidence to refine or clarify intent.
        Does NOT call tools - evidence should already be accumulated.
        """
        model_input = self.build_clarify_intent_model_input(state)
        try:
            raw_output = self.call_model(model_input)
        except PlannerError:
            raise
        except Exception as e:
            raise PlannerError(
                f"Planner model call failed during intent clarification: {type(e).__name__}: {e}",
                details={"planner": self.name},
            ) from e
        return self.normalize_clarify_intent_output(raw_output)
