"""Tests for the model-backed planner base class."""

import json

import pytest

from audio_agent.core.errors import PlannerError
from audio_agent.core.schemas import FrontendOutput, InitialPlan, PlannerActionType, PlannerDecision, ToolSpec
from audio_agent.core.state import create_initial_state
from audio_agent.planner.model_planner import (
    BaseModelPlanner,
    PlannerInputFormat,
    UnifiedPlannerInput,
)


class EchoModelPlanner(BaseModelPlanner):
    """Minimal test planner for BaseModelPlanner behavior."""

    @property
    def name(self) -> str:
        return "echo_model_planner"

    def initialize_model(self) -> dict:
        return {"ready": True}

    def call_model(self, model_input: UnifiedPlannerInput):
        if model_input.task_type == "initial_prompt":
            return "Focus on speech content and identify any unclear words."
        if model_input.task_type == "initial_plan":
            return {
                "approach": "Use the question to decide what evidence matters first.",
                "focus_points": ["speech content"],
                "possible_tool_types": ["asr"],
            }
        if model_input.task_type == "evidence_summary":
            return "Summary of evidence for testing."
        return {
            "action": "call_tool",
            "rationale": "Need ASR evidence first.",
            "selected_tool_name": "dummy_asr",
            "selected_tool_args": {"audio_path": "/tmp/audio.wav"},
            "selected_audio_id": "audio_0",
            "confidence": 0.8,
        }


class TestBaseModelPlanner:
    """Tests for model-backed planner flow and strict parsing."""

    def test_build_initial_prompt_model_input_has_expected_shape(self):
        planner = EchoModelPlanner()
        model_input = planner.build_initial_prompt_model_input("What is in this audio?")

        assert model_input.task_type == "initial_prompt"
        assert model_input.metadata["input_format"] == PlannerInputFormat.API_MODEL.value
        assert len(model_input.messages) == 2
        assert model_input.messages[0]["role"] == "system"
        assert model_input.messages[1]["role"] == "user"

    def test_build_plan_model_input_has_expected_shape(self):
        planner = EchoModelPlanner()
        model_input = planner.build_plan_model_input("What is in this audio?")

        assert model_input.task_type == "initial_plan"
        assert model_input.metadata["input_format"] == PlannerInputFormat.API_MODEL.value
        assert len(model_input.messages) == 2
        assert model_input.messages[0]["role"] == "system"
        assert model_input.messages[1]["role"] == "user"

    def test_build_plan_model_input_includes_skills_reference(self):
        planner = EchoModelPlanner()
        model_input = planner.build_plan_model_input("What is in this audio?")

        user_content = model_input.messages[1]["content"]
        # If task_skills.yaml exists, the reference should be appended
        # If it does not exist, the prompt should still be valid
        from audio_agent.utils.skill_io import TASK_SKILLS_PATH
        if TASK_SKILLS_PATH.exists():
            assert "Task Skills Reference" in user_content
        else:
            assert "Question:" in user_content

    def test_build_decision_model_input_local_mode(self):
        class LocalPlanner(EchoModelPlanner):
            @property
            def input_format(self) -> PlannerInputFormat:
                return PlannerInputFormat.LOCAL_MODEL

        planner = LocalPlanner()
        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(question_guided_caption="caption")
        state["initial_plan"] = planner.plan("Question")
        tools = [ToolSpec(name="dummy_asr", description="ASR")]

        model_input = planner.build_decision_model_input(state, tools)

        assert model_input.task_type == "decision"
        assert model_input.metadata["input_format"] == PlannerInputFormat.LOCAL_MODEL.value
        assert len(model_input.messages) == 2

    def test_generate_question_oriented_prompt_returns_string(self):
        planner = EchoModelPlanner()
        result = planner.generate_question_oriented_prompt("What is in this audio?")

        assert isinstance(result, str)
        assert result.strip()
        assert "speech content" in result

    def test_plan_returns_initial_plan(self):
        planner = EchoModelPlanner()
        result = planner.plan("What is in this audio?")

        assert isinstance(result, InitialPlan)
        assert result.approach
        assert result.possible_tool_types == ["asr"]

    def test_decide_returns_planner_decision(self):
        planner = EchoModelPlanner()
        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(question_guided_caption="caption")
        state["initial_plan"] = planner.plan("Question")
        tools = [ToolSpec(name="dummy_asr", description="ASR")]

        result = planner.decide(state, tools)

        assert isinstance(result, PlannerDecision)
        assert result.action == PlannerActionType.CALL_TOOL
        assert result.selected_tool_name == "dummy_asr"

    def test_invalid_plan_output_raises(self):
        class BadPlanner(EchoModelPlanner):
            def call_model(self, model_input: UnifiedPlannerInput):
                if model_input.task_type == "initial_plan":
                    return {"focus_points": [], "possible_tool_types": []}
                return super().call_model(model_input)

        planner = BadPlanner()
        with pytest.raises(PlannerError, match="missing required fields"):
            planner.plan("Question")

    def test_invalid_decision_output_raises(self):
        class BadPlanner(EchoModelPlanner):
            def call_model(self, model_input: UnifiedPlannerInput):
                if model_input.task_type == "decision":
                    return {"action": "call_tool", "rationale": "Need tool"}
                return super().call_model(model_input)

        planner = BadPlanner()
        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(question_guided_caption="caption")
        state["initial_plan"] = planner.plan("Question")
        tools = [ToolSpec(name="dummy_asr", description="ASR")]

        with pytest.raises(PlannerError, match="schema validation failed"):
            planner.decide(state, tools)

    def test_json_text_output_is_parsed(self):
        class JsonTextPlanner(EchoModelPlanner):
            def call_model(self, model_input: UnifiedPlannerInput):
                if model_input.task_type == "initial_plan":
                    return json.dumps(
                        {
                            "approach": "Use transcription first.",
                            "focus_points": ["speech"],
                            "possible_tool_types": ["asr"],
                        }
                    )
                return json.dumps(
                    {
                        "action": "fail",
                        "rationale": "Insufficient evidence.",
                        "confidence": 1.0,
                    }
                )

        planner = JsonTextPlanner()
        plan = planner.plan("Question")
        assert plan.approach == "Use transcription first."

        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(question_guided_caption="caption")
        state["initial_plan"] = plan
        decision = planner.decide(state, [])
        assert decision.action == PlannerActionType.FAIL

    def test_invalid_backend_mode_raises(self):
        class BadModePlanner(EchoModelPlanner):
            @property
            def input_format(self):
                return "bad_mode"

        planner = BadModePlanner()
        with pytest.raises(PlannerError, match="Unsupported planner input format"):
            planner.build_plan_model_input("Question")


class TestBaseModelPlannerRetries:
    """Tests for retry behavior on model output parsing errors."""

    def test_retry_recovers_after_transient_failure(self):
        """A planner that fails once then succeeds should return the correct result."""
        class FlakyPlanner(EchoModelPlanner):
            def __init__(self, fail_count: int = 1):
                self._fail_count = fail_count
                self._call_count = 0
                super().__init__()

            def call_model(self, model_input: UnifiedPlannerInput):
                self._call_count += 1
                if model_input.task_type == "initial_plan" and self._call_count <= self._fail_count:
                    return {"focus_points": [], "possible_tool_types": []}  # Missing 'approach'
                return super().call_model(model_input)

        planner = FlakyPlanner(fail_count=1)
        result = planner.plan("What is in this audio?")
        assert isinstance(result, InitialPlan)
        assert result.approach
        assert planner._call_count == 2  # 1 failure + 1 success

    def test_retry_exhausts_and_raises(self):
        """A planner that always fails should raise after max_retries + 1 attempts."""
        class AlwaysBadPlanner(EchoModelPlanner):
            def __init__(self):
                self._call_count = 0
                super().__init__(max_retries=2)

            def call_model(self, model_input: UnifiedPlannerInput):
                self._call_count += 1
                if model_input.task_type == "initial_plan":
                    return {"focus_points": [], "possible_tool_types": []}
                return super().call_model(model_input)

        planner = AlwaysBadPlanner()
        with pytest.raises(PlannerError, match="failed after 3 attempts"):
            planner.plan("What is in this audio?")
        assert planner._call_count == 3  # initial + 2 retries

    def test_zero_retries_raises_immediately(self):
        """With max_retries=0, the first failure should raise immediately."""
        class AlwaysBadPlanner(EchoModelPlanner):
            def __init__(self):
                self._call_count = 0
                super().__init__(max_retries=0)

            def call_model(self, model_input: UnifiedPlannerInput):
                self._call_count += 1
                if model_input.task_type == "initial_plan":
                    return {"focus_points": [], "possible_tool_types": []}
                return super().call_model(model_input)

        planner = AlwaysBadPlanner()
        with pytest.raises(PlannerError, match="failed after 1 attempt"):
            planner.plan("What is in this audio?")
        assert planner._call_count == 1
