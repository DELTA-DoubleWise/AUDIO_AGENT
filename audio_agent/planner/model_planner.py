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
from audio_agent.core.logging import get_logger
from audio_agent.core.schemas import InitialPlan, PlannerDecision, ToolSpec, FormatCheckResult
from audio_agent.core.state import AgentState
from audio_agent.planner.base import BasePlanner
from audio_agent.utils.model_io import parse_json_object_text, validate_message_sequence
from audio_agent.utils.prompt_io import load_prompt
from audio_agent.utils.skill_io import render_skills_reference


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
        max_retries: int = 3,
    ) -> None:
        self.model_config = model_config or {}
        self.max_retries = max_retries
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

    def build_initial_prompt_system_prompt(self) -> str:
        """Build system prompt for question-oriented prompt generation."""
        return load_prompt("initial_prompt_system")

    def build_initial_prompt_user_instruction(self, question: str) -> str:
        """Build user instruction for question-oriented prompt generation."""
        try:
            caption_skills = load_prompt("task_oriented_caption_skill")
        except Exception:
            caption_skills = "No caption skills reference available."
        return load_prompt("initial_prompt_user").format(
            question=question,
            caption_skills_reference=caption_skills,
        )

    def build_api_model_input_for_initial_prompt(self, question: str) -> UnifiedPlannerInput:
        """Build API-style planner input for question-oriented prompt generation."""
        system_prompt = self.build_initial_prompt_system_prompt()
        user_text = self.build_initial_prompt_user_instruction(question)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="initial_prompt",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "initial_prompt"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.API_MODEL.value},
        )

    def build_local_model_input_for_initial_prompt(self, question: str) -> UnifiedPlannerInput:
        """Build local-text-model input for question-oriented prompt generation."""
        system_prompt = self.build_initial_prompt_system_prompt()
        user_text = self.build_initial_prompt_user_instruction(question)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="initial_prompt",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "initial_prompt"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.LOCAL_MODEL.value},
        )

    def build_plan_system_prompt(self) -> str:
        """Build system prompt for initial planning phase."""
        return load_prompt("plan_system")

    def build_plan_user_instruction(self, question: str, frontend_output: FrontendOutput | None = None) -> str:
        """Build user instruction for initial planning phase."""
        frontend_caption = (
            frontend_output.question_guided_caption
            if frontend_output else "No frontend caption available."
        )
        user_text = load_prompt("plan_user").format(
            question=question,
            frontend_caption=frontend_caption,
        )
        skills_ref = render_skills_reference()
        if skills_ref:
            user_text = f"{user_text}\n\n{skills_ref}"
        return user_text

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
                "input_schema": tool.input_schema,
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

        # Build payload with decision rules FIRST so LLM sees them before evidence
        # This helps the model prioritize following the rules over getting distracted by evidence
        payload = {
            "question": state["question"],
            "decision_rules": rules_text,
            "expected_output_format": {
                "action": "answer | call_tool | clarify_intent | fail",
                "rationale": "str - detailed rationale explaining: (a) Why this action was chosen, (b) What evidence supports it, (c) For ANSWER: why confident the frontend model can generate a correct answer (see Rule 1)",
                "selected_tool_name": "str | null - REQUIRED for call_tool, must be a valid tool name",
                "selected_tool_args": "dict - arguments for the tool when using call_tool. MUST be {} (empty dict) for answer/clarify_intent/fail actions, never null",
                "selected_audio_id": "str | null - REQUIRED for call_tool, must be a valid audio_id from Available Audio Files",
                "draft_answer": "str | null - you do NOT need to provide this for answer; the frontend model will generate the final answer",
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
        }
        
        return json.dumps(payload, ensure_ascii=True)

    def build_api_model_input_for_plan(self, question: str, frontend_output: FrontendOutput | None = None) -> UnifiedPlannerInput:
        """Build API-style planner input for initial planning."""
        system_prompt = self.build_plan_system_prompt()
        user_text = self.build_plan_user_instruction(question, frontend_output)
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

    def build_local_model_input_for_plan(self, question: str, frontend_output: FrontendOutput | None = None) -> UnifiedPlannerInput:
        """Build local-text-model input for initial planning."""
        system_prompt = self.build_plan_system_prompt()
        user_text = self.build_plan_user_instruction(question, frontend_output)
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

    def build_initial_prompt_model_input(self, question: str) -> UnifiedPlannerInput:
        """Dispatch question-oriented prompt input build by backend mode."""
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
            model_input = self.build_api_model_input_for_initial_prompt(question)
        elif mode == PlannerInputFormat.LOCAL_MODEL:
            model_input = self.build_local_model_input_for_initial_prompt(question)
        else:
            raise PlannerError(
                "Unsupported planner input format",
                details={"input_format": mode.value},
            )

        self._validate_built_model_input(model_input)
        return model_input

    def build_plan_model_input(self, question: str, frontend_output: FrontendOutput | None = None) -> UnifiedPlannerInput:
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
            model_input = self.build_api_model_input_for_plan(question, frontend_output)
        elif mode == PlannerInputFormat.LOCAL_MODEL:
            model_input = self.build_local_model_input_for_plan(question, frontend_output)
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
                    details={
                        "missing_fields": missing,
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                    },
                )
            # Sanitize: remove None values for fields that have defaults
            fields_with_defaults = {"notes", "clarified_intent", "expected_output_format", 
                                    "requires_audio_output", "detailed_plan"}
            sanitized_output = {
                k: v for k, v in raw_output.items() 
                if v is not None or k not in fields_with_defaults
            }
            try:
                return InitialPlan(**sanitized_output)
            except Exception as e:
                raise PlannerError(
                    "Malformed initial plan output: schema validation failed",
                    details={
                        "error": str(e),
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                        "sanitized_output": sanitized_output,
                    },
                ) from e
        raise PlannerError(
            "Malformed initial plan output: expected dict, JSON text, or InitialPlan",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
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
                    details={
                        "missing_fields": missing,
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                    },
                )
            # Sanitize: remove None values for fields that have defaults
            # This allows Pydantic to use the default values instead of failing validation
            fields_with_defaults = {"confidence", "selected_tool_args", "selected_tool_name", 
                                    "selected_audio_id", "draft_answer"}
            sanitized_output = {
                k: v for k, v in raw_output.items() 
                if v is not None or k not in fields_with_defaults
            }
            try:
                return PlannerDecision(**sanitized_output)
            except Exception as e:
                raise PlannerError(
                    "Malformed planner decision output: schema validation failed",
                    details={
                        "error": str(e),
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                        "sanitized_output": sanitized_output,
                    },
                ) from e
        raise PlannerError(
            "Malformed planner decision output: expected dict, JSON text, or PlannerDecision",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def _call_with_retries(self, callable, context: str):
        """Call model and normalize output with retries on PlannerError."""
        logger = get_logger()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return callable()
            except PlannerError as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"{context} failed (attempt {attempt + 1}/{self.max_retries + 1}), retrying: {e}"
                    )
                    import time
                    time.sleep(0.5 * (2 ** attempt))
                else:
                    logger.error(f"{context} exhausted all retries: {e}")
        raise PlannerError(
            f"{context} failed after {self.max_retries + 1} attempts",
            details={"last_error": str(last_error), "retries": self.max_retries},
        ) from last_error

    def generate_question_oriented_prompt(self, question: str) -> str:
        """Generate a question-oriented prompt for the frontend model."""
        question = self.validate_question(question)
        model_input = self.build_initial_prompt_model_input(question)

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner model call failed during question-oriented prompt generation: {type(e).__name__}: {e}",
                    details={"planner": self.name},
                ) from e
            return self.normalize_question_oriented_prompt_output(raw_output)

        return self._call_with_retries(_call, "generate_question_oriented_prompt()")

    def normalize_question_oriented_prompt_output(self, raw_output: Any) -> str:
        """Normalize model output into a plain string prompt."""
        if isinstance(raw_output, str):
            stripped = raw_output.strip()
            if not stripped:
                raise PlannerError(
                    "Question-oriented prompt output is empty",
                    details={"raw_output": raw_output},
                )
            return stripped
        if isinstance(raw_output, dict):
            prompt = raw_output.get("question_oriented_prompt") or raw_output.get("prompt")
            if not prompt or not str(prompt).strip():
                raise PlannerError(
                    "Malformed question-oriented prompt output: missing prompt text",
                    details={"output_keys": sorted(raw_output.keys()), "raw_output": raw_output},
                )
            return str(prompt).strip()
        raise PlannerError(
            "Malformed question-oriented prompt output: expected str or dict",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def plan(self, question: str, frontend_output: FrontendOutput | None = None) -> InitialPlan:
        """Question-and-caption initial planning phase."""
        question = self.validate_question(question)
        model_input = self.build_plan_model_input(question, frontend_output)

        def _call():
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

        return self._call_with_retries(_call, "plan()")

    def decide(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> PlannerDecision:
        """Action decision phase using state + available tools."""
        self.validate_state(state)
        model_input = self.build_decision_model_input(state, available_tools)

        def _call():
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

        return self._call_with_retries(_call, "decide()")

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
                    details={
                        "output_keys": sorted(raw_output.keys()),
                        "raw_output": raw_output,
                    },
                )
            return clarified_intent, expected_format
        raise PlannerError(
            "Malformed clarify_intent output: expected dict, JSON text, or tuple",
            details={
                "output_type": type(raw_output).__name__,
                "raw_output": str(raw_output)[:1000],
            },
        )

    def clarify_intent(self, state: AgentState) -> tuple[str, str | None]:
        """
        Clarify the user's intent and expected output format.
        
        Uses reasoning on accumulated evidence to refine or clarify intent.
        Does NOT call tools - evidence should already be accumulated.
        """
        model_input = self.build_clarify_intent_model_input(state)

        def _call():
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

        return self._call_with_retries(_call, "clarify_intent()")

    # =============================================================================
    # Format Check Methods
    # =============================================================================

    def build_format_check_system_prompt(self) -> str:
        """Build system prompt for format checking phase."""
        return load_prompt("format_check_system")

    def build_format_check_user_instruction(
        self,
        proposed_answer: str,
        expected_format: str | None,
        question: str,
        requires_audio_output: bool = False,
    ) -> str:
        """Build user instruction for format checking phase."""
        return load_prompt("format_check_user").format(
            question=question,
            expected_format=expected_format or "No specific format required",
            proposed_answer=proposed_answer,
            is_audio_output_task="Yes" if requires_audio_output else "No",
        )

    def build_format_check_model_input(
        self,
        proposed_answer: str,
        expected_format: str | None,
        question: str,
        requires_audio_output: bool = False,
    ) -> UnifiedPlannerInput:
        """Build model input for format checking."""
        system_prompt = self.build_format_check_system_prompt()
        user_text = self.build_format_check_user_instruction(
            proposed_answer, expected_format, question, requires_audio_output
        )
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="format_check",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={
                "question": question,
                "task": "format_check",
                "expected_format": expected_format,
            },
            metadata={
                "planner_name": self.name,
                "task_type": "format_check",
            },
        )

    def normalize_format_check_output(self, raw_output: Any) -> FormatCheckResult:
        """Normalize model output into FormatCheckResult."""
        if isinstance(raw_output, FormatCheckResult):
            return raw_output
        if isinstance(raw_output, str):
            raw_output = parse_json_object_text(
                raw_output,
                error_cls=PlannerError,
                subject="Planner",
            )
        if isinstance(raw_output, dict):
            required = {"passed"}
            keys = set(raw_output.keys())
            missing = sorted(required - keys)
            if missing:
                raise PlannerError(
                    "Malformed format check output: missing required fields",
                    details={
                        "missing_fields": missing,
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                    },
                )
            # Sanitize: remove None values for fields that have defaults
            fields_with_defaults = {"critique", "confidence"}
            sanitized_output = {
                k: v for k, v in raw_output.items() 
                if v is not None or k not in fields_with_defaults
            }
            try:
                return FormatCheckResult(**sanitized_output)
            except Exception as e:
                raise PlannerError(
                    "Malformed format check output: schema validation failed",
                    details={
                        "error": str(e),
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                        "sanitized_output": sanitized_output,
                    },
                ) from e
        raise PlannerError(
            "Malformed format check output: expected dict, JSON text, or FormatCheckResult",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def check_format(
        self,
        proposed_answer: str,
        expected_format: str | None,
        question: str,
        requires_audio_output: bool = False,
    ) -> FormatCheckResult:
        """
        Check if the proposed answer follows the expected output format.
        
        Validates format compliance only - does NOT check content correctness.
        """
        # If no format specified and answer is non-empty, auto-pass
        if not expected_format and proposed_answer and proposed_answer.strip():
            return FormatCheckResult(
                passed=True,
                critique=None,
                confidence=1.0,
            )
        
        model_input = self.build_format_check_model_input(
            proposed_answer, expected_format, question, requires_audio_output
        )

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner model call failed during format check: {type(e).__name__}: {e}",
                    details={"planner": self.name},
                ) from e
            return self.normalize_format_check_output(raw_output)

        return self._call_with_retries(_call, "check_format()")

    # =============================================================================
    # Evidence Summary Methods
    # =============================================================================

    def build_evidence_summary_system_prompt(self) -> str:
        """Build system prompt for evidence summarization phase."""
        return load_prompt("evidence_summary_system")

    def build_evidence_summary_user_instruction(self, state: AgentState) -> str:
        """Build user instruction for evidence summarization phase."""
        question = state["question"]
        evidence_log = state.get("evidence_log", [])
        planner_trace = state.get("planner_trace", [])
        tool_history = state.get("tool_call_history", [])
        frontend_output = state.get("initial_frontend_output")
        clarified_intent = state.get("clarified_intent")
        expected_output_format = state.get("expected_output_format")

        evidence_text = "\n".join(
            f"[{item.source}] {item.content}"
            for item in evidence_log
        ) if evidence_log else "No evidence yet."

        planner_trace_text = "\n".join(
            f"Step {i+1}: {d.action.value} - {d.rationale}"
            for i, d in enumerate(planner_trace)
        ) if planner_trace else "No planner decisions yet."

        tool_history_text = "\n".join(
            f"- {record.request.tool_name}: success={record.result.success}"
            for record in tool_history
        ) if tool_history else "No tools called."

        frontend_caption = (
            frontend_output.question_guided_caption
            if frontend_output else "No frontend output yet."
        )

        return load_prompt("evidence_summary_user").format(
            question=question,
            frontend_caption=frontend_caption,
            evidence_text=evidence_text,
            planner_trace_text=planner_trace_text,
            tool_history_text=tool_history_text,
            clarified_intent=clarified_intent or "Not yet clarified",
            expected_output_format=expected_output_format or "Not yet specified",
        )

    def build_evidence_summary_model_input(self, state: AgentState) -> UnifiedPlannerInput:
        """Build model input for evidence summarization."""
        system_prompt = self.build_evidence_summary_system_prompt()
        user_text = self.build_evidence_summary_user_instruction(state)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="evidence_summary",
            question=state["question"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": state["question"], "task": "evidence_summary"},
            metadata={"planner_name": self.name, "task_type": "evidence_summary"},
        )

    def normalize_evidence_summary_output(self, raw_output: Any) -> str:
        """Normalize model output into a plain string summary."""
        if isinstance(raw_output, str):
            stripped = raw_output.strip()
            if not stripped:
                raise PlannerError(
                    "Evidence summary output is empty",
                    details={"raw_output": raw_output},
                )
            return stripped
        if isinstance(raw_output, dict):
            summary = raw_output.get("summary") or raw_output.get("evidence_summary")
            if not summary or not str(summary).strip():
                raise PlannerError(
                    "Malformed evidence summary output: missing summary text",
                    details={"output_keys": sorted(raw_output.keys()), "raw_output": raw_output},
                )
            return str(summary).strip()
        raise PlannerError(
            "Malformed evidence summary output: expected str or dict",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def summarize_evidence(self, state: AgentState) -> str:
        """
        Summarize accumulated evidence into a concise narrative.
        
        Uses the planner (text LLM) to compress evidence_log, planner_trace,
        and tool_call_history into a single neutral summary.
        """
        model_input = self.build_evidence_summary_model_input(state)

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner model call failed during evidence summarization: {type(e).__name__}: {e}",
                    details={"planner": self.name},
                ) from e
            return self.normalize_evidence_summary_output(raw_output)

        return self._call_with_retries(_call, "summarize_evidence()")
