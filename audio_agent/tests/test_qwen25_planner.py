"""Tests for Qwen2.5 planner adapter."""

import json
import os

import pytest

from audio_agent.core.errors import PlannerError
from audio_agent.core.schemas import FrontendOutput, InitialPlan, PlannerActionType, PlannerDecision, ToolSpec
from audio_agent.core.state import create_initial_state
from audio_agent.planner.model_planner import PlannerInputFormat, UnifiedPlannerInput
from audio_agent.planner.qwen25_planner import Qwen25Planner


class MockQwen25Planner(Qwen25Planner):
    """Lightweight test double that avoids loading large model weights."""

    def initialize_model(self):
        return {"model": object(), "tokenizer": object()}

    def call_model(self, model_input: UnifiedPlannerInput) -> str:
        if model_input.task_type == "initial_prompt":
            return "Focus on speech content and identify any unclear words."
        if model_input.task_type == "initial_plan":
            return json.dumps(
                {
                    "approach": "Inspect the question first, then gather missing evidence.",
                    "focus_points": ["speech content"],
                    "possible_tool_types": ["asr"],
                }
            )
        return json.dumps(
            {
                "action": "fail",
                "rationale": "Need a real backend for execution in this mock.",
                "confidence": 1.0,
            }
        )


class TestQwen25PlannerShape:
    """Unit tests for planner wiring and input format behavior."""

    def test_uses_local_model_input_format(self):
        planner = MockQwen25Planner()
        assert planner.input_format == PlannerInputFormat.LOCAL_MODEL

    def test_build_initial_prompt_model_input_contains_chat_messages(self):
        planner = MockQwen25Planner()
        model_input = planner.build_initial_prompt_model_input("What is in this audio?")

        assert model_input.task_type == "initial_prompt"
        assert model_input.messages[0]["role"] == "system"
        assert model_input.messages[1]["role"] == "user"
        assert isinstance(model_input.messages[1]["content"], str)

    def test_generate_question_oriented_prompt_returns_string(self):
        planner = MockQwen25Planner()
        result = planner.generate_question_oriented_prompt("What is in this audio?")

        assert isinstance(result, str)
        assert result.strip()
        assert "speech content" in result

    def test_build_plan_model_input_contains_chat_messages(self):
        planner = MockQwen25Planner()
        model_input = planner.build_plan_model_input("What is in this audio?")

        assert model_input.task_type == "initial_plan"
        assert model_input.messages[0]["role"] == "system"
        assert model_input.messages[1]["role"] == "user"
        assert isinstance(model_input.messages[1]["content"], str)

    def test_build_decision_model_input_contains_chat_messages(self):
        planner = MockQwen25Planner()
        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(question_guided_caption="caption")
        state["initial_plan"] = InitialPlan(
            approach="Use ASR first.",
            focus_points=["speech"],
            possible_tool_types=["asr"],
        )
        tools = [ToolSpec(name="dummy_asr", description="ASR")]

        model_input = planner.build_decision_model_input(state, tools)

        assert model_input.task_type == "decision"
        assert model_input.messages[0]["role"] == "system"
        assert model_input.messages[1]["role"] == "user"
        assert isinstance(model_input.messages[1]["content"], str)

    def test_plan_returns_initial_plan_from_raw_json_text(self):
        planner = MockQwen25Planner()
        result = planner.plan("What is in this audio?")

        assert isinstance(result, InitialPlan)
        assert result.approach.startswith("Inspect the question first")

    def test_decide_returns_planner_decision_from_raw_json_text(self):
        planner = MockQwen25Planner()
        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(question_guided_caption="caption")
        state["initial_plan"] = planner.plan("Question")

        result = planner.decide(state, [])

        assert isinstance(result, PlannerDecision)
        assert result.action == PlannerActionType.FAIL

    def test_empty_decoded_response_raises(self):
        class EmptyResponsePlanner(MockQwen25Planner):
            def call_model(self, model_input: UnifiedPlannerInput) -> str:
                return "   "

        planner = EmptyResponsePlanner()
        with pytest.raises(PlannerError, match="Planner text output is empty"):
            planner.plan("Question")


def test_qwen25_planner_cluster_smoke():
    """
    Optional integration test for cluster use.

    Enable by setting:
    - RUN_QWEN25_PLANNER_TEST=1
    - optional QWEN25_MODEL_PATH
    """
    if os.getenv("RUN_QWEN25_PLANNER_TEST") != "1":
        pytest.skip("Set RUN_QWEN25_PLANNER_TEST=1 to run Qwen2.5 planner integration test")

    planner = Qwen25Planner(
        model_path=os.getenv("QWEN25_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
    )

    plan = planner.plan("What evidence would help determine whether the audio contains speech?")
    assert plan.approach.strip()

    state = create_initial_state(
        "What is being said in the audio?",
        "/tmp/cluster_test_audio.wav",
    )
    state["initial_frontend_output"] = FrontendOutput(
        question_guided_caption="The audio appears to contain speech, but the words are unclear."
    )
    state["initial_plan"] = plan

    decision = planner.decide(state, [ToolSpec(name="dummy_asr", description="ASR")])
    assert decision.rationale.strip()
