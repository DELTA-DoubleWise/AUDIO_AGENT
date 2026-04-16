"""Tests for two-stage planner node flow (initial plan + action decision)."""

import pytest

from audio_agent.core.errors import StateValidationError
from audio_agent.core.schemas import FrontendOutput, PlannerActionType
from audio_agent.core.state import create_initial_state
from audio_agent.graph.nodes import create_initial_plan_node, create_planner_decision_node
from audio_agent.planner.dummy_planner import DummyPlanner
from audio_agent.tools.dummy_tools import DummyASRTool
from audio_agent.tools.registry import ToolRegistry


class TestPlannerStages:
    """Tests for initial planning and decision-stage fail-fast behavior."""

    def test_initial_prompt_node_generates_prompt_from_question(self):
        from audio_agent.graph.nodes import create_initial_prompt_node
        planner = DummyPlanner()
        node = create_initial_prompt_node(planner)

        result = node({"question": "What is discussed in this audio?"})

        assert result["question_oriented_prompt"] is not None
        assert "What is discussed in this audio?" in result["question_oriented_prompt"]

    def test_initial_plan_node_generates_plan_from_question_and_frontend_output(self):
        from audio_agent.core.schemas import FrontendOutput
        planner = DummyPlanner()
        node = create_initial_plan_node(planner)

        result = node({
            "question": "What is discussed in this audio?",
            "initial_frontend_output": FrontendOutput(
                question_guided_caption="A discussion about technology."
            ),
        })

        assert result["initial_plan"] is not None
        assert result["initial_plan"].approach
        assert len(result["initial_plan_trace"]) == 1

    def test_planner_decision_node_raises_if_initial_plan_missing(self):
        planner = DummyPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_planner_decision_node(planner, registry)

        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(
            question_guided_caption="A short caption."
        )

        with pytest.raises(StateValidationError, match="Missing required state fields"):
            node(state)

    def test_planner_decision_node_raises_if_frontend_output_missing(self):
        planner = DummyPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_planner_decision_node(planner, registry)

        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_plan"] = planner.plan("Question")

        with pytest.raises(StateValidationError, match="Missing required state fields"):
            node(state)

    def test_planner_decision_node_runs_after_stage1_outputs_exist(self):
        planner = DummyPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_planner_decision_node(planner, registry)

        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(
            question_guided_caption="A short caption."
        )
        state["initial_plan"] = planner.plan("Question")

        result = node(state)

        assert result["current_decision"] is not None
        assert result["current_decision"].action == PlannerActionType.CALL_TOOL
        assert len(result["planner_trace"]) == 1
