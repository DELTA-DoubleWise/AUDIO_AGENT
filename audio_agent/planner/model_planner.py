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
from audio_agent.utils.prompt_io import load_prompt


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
        model_config: dict[str, Any] | None = None,
    ) -> None:
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
        return load_prompt("plan_system")

    def build_plan_user_instruction(self, question: str) -> str:
        """Build user instruction for initial planning phase."""
        return load_prompt("plan_user").format(question=question)

    def build_decision_system_prompt(self) -> str:
        """Build system prompt for action decision phase."""
        return load_prompt("decide_system")

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
        audio_list = state.get("audio_list", [])
        verification_result = state.get("verification_result")
        verification_count = state.get("verification_count", 0)

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
        
        # Build audio list summary with descriptions
        audio_summary = [
            f"- {a.audio_id}: {a.description} (source: {a.source})"
            for a in audio_list
        ]

        # Load decision rules from markdown
        # Use raw rules text to preserve multi-line formatting and bullet points
        rules_text = load_prompt("decide_rules")

        # Build verification context if applicable
        verification_context = None
        if verification_result is not None:
            verification_context = {
                "passed": verification_result.passed,
                "confidence": verification_result.confidence,
                "critique": verification_result.critique,
            }

        # Build payload with decision rules FIRST so LLM sees them before evidence
        # This helps the model prioritize following the rules over getting distracted by evidence
        payload = {
            "question": state["question"],
            "decision_rules": rules_text,
            "expected_output_format": {
                "action": "answer | call_tool | clarify_intent | verify | fail",
                "rationale": "str - detailed rationale explaining: (a) Why this action was chosen, (b) What evidence supports it, (c) For VERIFY: why verification is needed (see Rule 11), (d) For ANSWER: why confident in the answer (see Rule 1)",
                "selected_tool_name": "str | null - REQUIRED for call_tool, must be a valid tool name",
                "selected_tool_args": "dict - arguments for the tool when using call_tool. MUST be {} (empty dict) for answer/verify/clarify_intent/fail actions, never null",
                "selected_audio_id": "str | null - REQUIRED for call_tool, must be a valid audio_id from Available Audio Files",
                "draft_answer": "str | null - REQUIRED for answer AND verify actions, your proposed answer",
                "confidence": "float - 0.0 to 1.0",
            },
            "frontend_caption": frontend_output.question_guided_caption,
            "initial_plan": initial_plan.model_dump(mode="json"),
            "evidence_log": evidence_summary,
            "tool_call_history": tool_history_summary,
            "audio_list": "\n".join(audio_summary) if audio_summary else "- audio_0: original input audio (source: original)",
            "available_tools": tool_summary,
            "step_count": state.get("step_count", 0),
            "max_steps": state.get("max_steps", 10),
            "verification_count": verification_count,
            "verification_result": verification_context,
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
        
        decision = self.normalize_decision_output(raw_output)
        return decision

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
        initial_plan = state.get("initial_plan")

        # Build evidence summary
        evidence_text = "\n".join(
            f"[{item.source}] {item.content}"
            for item in evidence_log
        )

        # Check if audio output is expected
        requires_audio_output = (
            initial_plan.requires_audio_output if initial_plan else False
        )

        system_prompt = load_prompt("answer_system")
        user_text = load_prompt("answer_user").format(
            question=question,
            evidence_text=evidence_text,
        )

        # Add audio output context if applicable
        if requires_audio_output:
            user_text += "\n\n**Task Type:** This task requires producing an audio file output."
            user_text += "\nPlease confirm the audio processing was completed successfully."

        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="final_answer",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "final_answer"},
            metadata={
                "planner_name": self.name,
                "task_type": "final_answer",
                "requires_audio_output": requires_audio_output,
            },
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

        system_prompt = load_prompt("clarify_system")
        user_text = load_prompt("clarify_user").format(
            question=question,
            clarified_intent=clarified_intent or "Not yet clarified",
            expected_format=expected_format or "Not yet specified",
            evidence_text=evidence_text if evidence_text else "No evidence yet.",
        )

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
