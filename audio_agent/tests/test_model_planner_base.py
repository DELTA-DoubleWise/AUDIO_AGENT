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
        if model_input.task_type == "initial_plan":
            return {
                "approach": "Use the question to decide what evidence matters first.",
                "focus_points": ["speech content"],
                "possible_tool_types": ["asr"],
            }
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

    def test_build_plan_model_input_has_expected_shape(self):
        planner = EchoModelPlanner()
        model_input = planner.build_plan_model_input("What is in this audio?")

        assert model_input.task_type == "initial_plan"
        assert model_input.metadata["input_format"] == PlannerInputFormat.API_MODEL.value
        assert len(model_input.messages) == 2
        assert model_input.messages[0]["role"] == "system"
        assert model_input.messages[1]["role"] == "user"

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
