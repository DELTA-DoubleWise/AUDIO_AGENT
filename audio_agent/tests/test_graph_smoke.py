"""Smoke tests for the LangGraph workflow."""

import pytest

from audio_agent.main import create_dummy_agent, AudioAgent
from audio_agent.config.settings import AgentConfig
from audio_agent.core.state import create_initial_state
from audio_agent.core.constants import AgentStatus
from audio_agent.graph.builder import build_graph
from audio_agent.frontend.dummy_frontend import DummyFrontend
from audio_agent.planner.dummy_planner import DummyPlanner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.dummy_tools import DummyASRTool, DummyAudioEventDetectorTool
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser


class TestGraphSmoke:
    """Smoke tests for the agent graph."""
    
    def test_dummy_agent_runs_to_completion(self):
        """Test that dummy agent runs without errors and produces answer."""
        agent = create_dummy_agent()
        
        final_state = agent.run(
            question="What is in this audio?",
            audio_path_or_uri="/fake/audio.wav",
            max_steps=10,
        )
        
        assert final_state is not None
        assert final_state["status"] == AgentStatus.ANSWERED
        assert final_state["final_answer"] is not None
        assert final_state["initial_plan"] is not None
        assert len(final_state["initial_plan_trace"]) >= 1
        assert len(final_state["evidence_log"]) > 0
        assert len(final_state["tool_call_history"]) > 0
    
    def test_graph_respects_max_steps(self):
        """Test that graph stops at max_steps."""
        # Create a planner that always calls tools
        class AlwaysCallToolPlanner(DummyPlanner):
            def decide(self, state, available_tools):
                from audio_agent.core.schemas import PlannerDecision, PlannerActionType
                
                step_count = state.get("step_count", 0)
                max_steps = state.get("max_steps", 10)
                
                if step_count >= max_steps:
                    return PlannerDecision(
                        action=PlannerActionType.FAIL,
                        rationale=f"Exceeded max steps ({max_steps})",
                        confidence=1.0,
                    )
                
                return PlannerDecision(
                    action=PlannerActionType.CALL_TOOL,
                    rationale="Always call tool",
                    selected_tool_name="dummy_asr",
                    selected_tool_args={},
                    confidence=0.5,
                )
        
        # Build custom agent
        frontend = DummyFrontend()
        planner = AlwaysCallToolPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        fuser = DefaultEvidenceFuser()
        
        agent = AudioAgent(
            frontend=frontend,
            planner=planner,
            registry=registry,
            fuser=fuser,
        )
        
        final_state = agent.run(
            question="Test question",
            audio_path_or_uri="/test.wav",
            max_steps=3,
        )
        
        # Should fail due to max steps
        assert final_state["status"] in (AgentStatus.FAILED, AgentStatus.EXHAUSTED)
        assert final_state["step_count"] <= 3
    
    def test_graph_builds_with_all_components(self):
        """Test that graph builds successfully with all components."""
        frontend = DummyFrontend()
        planner = DummyPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        registry.register(DummyAudioEventDetectorTool())
        fuser = DefaultEvidenceFuser()
        
        graph = build_graph(frontend, planner, registry, fuser)
        
        assert graph is not None
    
    def test_graph_raises_on_none_components(self):
        """Test that graph builder raises on None components."""
        frontend = DummyFrontend()
        planner = DummyPlanner()
        registry = ToolRegistry()
        fuser = DefaultEvidenceFuser()
        
        with pytest.raises(ValueError, match="frontend cannot be None"):
            build_graph(None, planner, registry, fuser)
        
        with pytest.raises(ValueError, match="planner cannot be None"):
            build_graph(frontend, None, registry, fuser)
        
        with pytest.raises(ValueError, match="registry cannot be None"):
            build_graph(frontend, planner, None, fuser)
        
        with pytest.raises(ValueError, match="fuser cannot be None"):
            build_graph(frontend, planner, registry, None)
    
    def test_evidence_accumulates(self):
        """Test that evidence is properly accumulated."""
        agent = create_dummy_agent()
        
        final_state = agent.run(
            question="What sounds are in this audio?",
            audio_path_or_uri="/test/audio.mp3",
        )
        
        evidence_log = final_state["evidence_log"]
        
        # Should have initial evidence from frontend
        frontend_evidence = [e for e in evidence_log if "frontend" in e.source]
        assert len(frontend_evidence) >= 1
        
        # Should have evidence from tools
        tool_evidence = [e for e in evidence_log if "frontend" not in e.source]
        assert len(tool_evidence) >= 1
    
    def test_tool_history_recorded(self):
        """Test that tool calls are properly recorded."""
        agent = create_dummy_agent()
        
        final_state = agent.run(
            question="Describe the audio",
            audio_path_or_uri="/sample.wav",
        )
        
        tool_history = final_state["tool_call_history"]
        
        # Dummy planner calls 2 tools before answering
        assert len(tool_history) >= 1
        
        for record in tool_history:
            assert record.request is not None
            assert record.result is not None
            assert record.result.success is True


class TestAgentInterface:
    """Tests for AudioAgent interface methods."""
    
    def test_is_successful(self):
        """Test is_successful method."""
        agent = create_dummy_agent()
        
        final_state = agent.run(
            question="Test",
            audio_path_or_uri="/test.wav",
        )
        
        assert agent.is_successful(final_state) is True
    
    def test_get_answer(self):
        """Test get_answer method."""
        agent = create_dummy_agent()
        
        final_state = agent.run(
            question="What is this?",
            audio_path_or_uri="/test.wav",
        )
        
        answer = agent.get_answer(final_state)
        
        assert answer is not None
        assert answer.answer is not None
        assert len(answer.answer) > 0
    
    def test_get_status(self):
        """Test get_status method."""
        agent = create_dummy_agent()
        
        final_state = agent.run(
            question="Test",
            audio_path_or_uri="/test.wav",
        )
        
        status = agent.get_status(final_state)
        
        assert status == AgentStatus.ANSWERED
