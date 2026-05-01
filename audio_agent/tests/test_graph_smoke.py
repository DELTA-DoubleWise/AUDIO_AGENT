"""Smoke tests for the LangGraph workflow."""

import os
import tempfile

import pytest

from audio_agent.main import create_dummy_agent, AudioAgent
from audio_agent.config.settings import AgentConfig
from audio_agent.core.state import create_initial_state
from audio_agent.core.constants import AgentStatus
from audio_agent.core.schemas import AudioItem, PlannerDecision, PlannerActionType
from audio_agent.graph.builder import build_graph
from audio_agent.frontend.dummy_frontend import DummyFrontend
from audio_agent.planner.dummy_planner import DummyPlanner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.dummy_tools import DummyASRTool, DummyAudioEventDetectorTool
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser


def create_test_audio_file():
    """Create a temporary audio file for testing."""
    # Create an empty file with .wav extension for testing
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return path


class TestGraphSmoke:
    """Smoke tests for the agent graph."""
    
    def test_dummy_agent_runs_to_completion(self):
        """Test that dummy agent runs without errors and produces answer."""
        audio_path = create_test_audio_file()
        try:
            agent = create_dummy_agent()
            
            final_state = agent.run(
                question="What is in this audio?",
                audio_paths=[audio_path],
                max_steps=10,
            )
            
            assert final_state is not None
            assert final_state["status"] == AgentStatus.ANSWERED
            assert final_state["final_answer"] is not None
            assert final_state["initial_plan"] is not None
            assert len(final_state["initial_plan_trace"]) >= 1
            assert len(final_state["evidence_log"]) > 0
            assert len(final_state["tool_call_history"]) > 0
            assert final_state["evidence_summary"] is not None
            assert len(final_state["evidence_summary"]) > 0
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_graph_respects_max_steps(self):
        """Test that graph stops at max_steps with final answer."""
        audio_path = create_test_audio_file()
        try:
            # Create a planner that always calls tools
            class AlwaysCallToolPlanner(DummyPlanner):
                def decide(self, state, available_tools):
                    from audio_agent.core.schemas import PlannerDecision, PlannerActionType
                    
                    step_count = state.get("step_count", 0)
                    max_steps = state.get("max_steps", 10)
                    
                    # Get first audio from audio_list
                    audio_list = state.get("audio_list", [])
                    selected_audio_id = audio_list[0].audio_id if audio_list else "audio_0"
                    
                    if step_count >= max_steps - 1:
                        # On final step, planner_decision_node will generate answer
                        return PlannerDecision(
                            action=PlannerActionType.CALL_TOOL,
                            rationale="Always call tool",
                            selected_tool_name="dummy_asr",
                            selected_tool_args={},
                            selected_audio_id=selected_audio_id,
                            confidence=0.5,
                        )
                    
                    return PlannerDecision(
                        action=PlannerActionType.CALL_TOOL,
                        rationale="Always call tool",
                        selected_tool_name="dummy_asr",
                        selected_tool_args={},
                        selected_audio_id=selected_audio_id,
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
                audio_paths=[audio_path],
                max_steps=3,
            )
            
            # Should complete (either answered or exhausted)
            assert final_state["status"] in (AgentStatus.ANSWERED, AgentStatus.EXHAUSTED)
            assert final_state["step_count"] <= 3
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
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
        audio_path = create_test_audio_file()
        try:
            agent = create_dummy_agent()
            
            final_state = agent.run(
                question="What sounds are in this audio?",
                audio_paths=[audio_path],
            )
            
            evidence_log = final_state["evidence_log"]
            
            # Should have initial evidence from frontend
            frontend_evidence = [e for e in evidence_log if "frontend" in e.source]
            assert len(frontend_evidence) >= 1
            
            # Should have evidence from tools
            tool_evidence = [e for e in evidence_log if "frontend" not in e.source]
            assert len(tool_evidence) >= 1
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_tool_history_recorded(self):
        """Test that tool calls are properly recorded."""
        audio_path = create_test_audio_file()
        try:
            agent = create_dummy_agent()
            
            final_state = agent.run(
                question="Describe the audio",
                audio_paths=[audio_path],
            )
            
            tool_history = final_state["tool_call_history"]
            
            # Dummy planner calls 2 tools before answering
            assert len(tool_history) >= 1
            
            for record in tool_history:
                assert record.request is not None
                assert record.result is not None
                assert record.result.success is True
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)


class TestAgentInterface:
    """Tests for AudioAgent interface methods."""
    
    def test_is_successful(self):
        """Test is_successful method."""
        audio_path = create_test_audio_file()
        try:
            agent = create_dummy_agent()
            
            final_state = agent.run(
                question="Test",
                audio_paths=[audio_path],
            )
            
            assert agent.is_successful(final_state) is True
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_get_answer(self):
        """Test get_answer method."""
        audio_path = create_test_audio_file()
        try:
            agent = create_dummy_agent()
            
            final_state = agent.run(
                question="What is this?",
                audio_paths=[audio_path],
            )
            
            answer = agent.get_answer(final_state)
            
            assert answer is not None
            assert answer.answer is not None
            assert len(answer.answer) > 0
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_get_status(self):
        """Test get_status method."""
        audio_path = create_test_audio_file()
        try:
            agent = create_dummy_agent()
            
            final_state = agent.run(
                question="Test",
                audio_paths=[audio_path],
            )
            
            status = agent.get_status(final_state)
            
            assert status == AgentStatus.ANSWERED
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)



class TestFrontendFollowupRouting:
    """Tests for frontend follow-up routing."""
    
    def test_route_after_planner_decision_call_frontend(self):
        """Test that CALL_FRONTEND routes to frontend_followup_node."""
        from audio_agent.graph.routing import route_after_planner_decision, NODE_FRONTEND_FOLLOWUP
        from audio_agent.core.schemas import PlannerDecision, PlannerActionType
        
        state = create_initial_state(
            question="Test",
            audio_paths=["/fake/audio.wav"],
        )
        state["current_decision"] = PlannerDecision(
            action=PlannerActionType.CALL_FRONTEND,
            rationale="Re-perceive",
            selected_audio_ids=["audio_0"],
            frontend_followup_prompt="What emotion?",
            confidence=0.8,
        )
        
        result = route_after_planner_decision(state)
        assert result == NODE_FRONTEND_FOLLOWUP
    
    def test_route_after_frontend_followup(self):
        """Test that route_after_frontend_followup goes to evidence_fusion."""
        from audio_agent.graph.routing import route_after_frontend_followup, NODE_EVIDENCE_FUSION
        from audio_agent.core.schemas import FrontendOutput
        
        state = create_initial_state(
            question="Test",
            audio_paths=["/fake/audio.wav"],
        )
        state["latest_frontend_followup_output"] = FrontendOutput(
            question_guided_caption="Mock follow-up output"
        )
        
        result = route_after_frontend_followup(state)
        assert result == NODE_EVIDENCE_FUSION
    
    def test_route_after_frontend_followup_none_raises(self):
        """Test that routing fails when latest_frontend_followup_output is None."""
        from audio_agent.graph.routing import route_after_frontend_followup
        from audio_agent.core.errors import GraphRoutingError
        
        state = create_initial_state(
            question="Test",
            audio_paths=["/fake/audio.wav"],
        )
        
        with pytest.raises(GraphRoutingError, match="latest_frontend_followup_output is None"):
            route_after_frontend_followup(state)


class TestFrontendFollowupSmoke:
    """Smoke tests for the frontend follow-up path."""
    
    def test_call_frontend_path_end_to_end(self):
        """Test that CALL_FRONTEND path runs and produces evidence."""
        audio_path = create_test_audio_file()
        try:
            # Create a planner that emits CALL_FRONTEND on the original audio
            class FrontendFollowupPlanner(DummyPlanner):
                def decide(self, state, available_tools):
                    tool_history = state.get("tool_call_history", [])
                    planner_trace = state.get("planner_trace", [])
                    audio_list = state.get("audio_list", [])
                    
                    has_done_followup = any(
                        d.action == PlannerActionType.CALL_FRONTEND
                        for d in planner_trace
                    )
                    
                    if not has_done_followup and len(tool_history) >= 1:
                        return PlannerDecision(
                            action=PlannerActionType.CALL_FRONTEND,
                            rationale="Re-perceive the audio with a targeted prompt",
                            selected_audio_ids=[audio_list[0].audio_id],
                            frontend_followup_prompt="What emotion does the speaker express?",
                            frontend_followup_goal="Identify speaker emotion",
                            confidence=0.8,
                        )
                    
                    if len(tool_history) == 0:
                        return PlannerDecision(
                            action=PlannerActionType.CALL_TOOL,
                            rationale="Call tool first",
                            selected_tool_name="dummy_asr",
                            selected_audio_id=audio_list[0].audio_id if audio_list else "audio_0",
                            confidence=0.8,
                        )
                    
                    return PlannerDecision(
                        action=PlannerActionType.ANSWER,
                        rationale="Enough evidence",
                        confidence=0.8,
                    )
            
            frontend = DummyFrontend()
            planner = FrontendFollowupPlanner()
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
                question="What is the emotion in this audio?",
                audio_paths=[audio_path],
                max_steps=10,
            )
            
            assert final_state["status"] == AgentStatus.ANSWERED
            assert final_state["final_answer"] is not None
            
            # Verify follow-up evidence exists
            evidence_log = final_state["evidence_log"]
            followup_evidence = [
                e for e in evidence_log
                if e.evidence_type == "frontend_followup"
            ]
            assert len(followup_evidence) >= 1
            
            # Verify planner trace contains CALL_FRONTEND
            planner_trace = final_state["planner_trace"]
            followup_decisions = [
                d for d in planner_trace
                if d.action == PlannerActionType.CALL_FRONTEND
            ]
            assert len(followup_decisions) >= 1
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_call_frontend_on_generated_audio(self):
        """Test CALL_FRONTEND on a tool-generated audio artifact."""
        import tempfile
        
        audio_path = create_test_audio_file()
        try:
            # Build with default DummyPlanner which now emits CALL_FRONTEND
            # when non-original audio exists. We need a tool that generates audio.
            agent = create_dummy_agent()
            
            # Manually add a generated audio to the agent's registry
            # We can't easily do this through the public API, so instead
            # we verify that the DummyPlanner logic is correct by checking
            # its decision on a state with non-original audio.
            from audio_agent.core.schemas import AudioItem
            
            state = create_initial_state(
                question="Test",
                audio_paths=[audio_path],
                audio_list=[
                    AudioItem(audio_id="audio_0", path=audio_path, source="original", description="original"),
                    AudioItem(audio_id="audio_1", path=audio_path, source="dummy_trim", description="trimmed"),
                ],
            )
            state["initial_plan"] = agent.planner.plan("Test")
            state["initial_frontend_output"] = DummyFrontend().run("Test", [audio_path])
            
            decision = agent.planner.decide(state, agent.registry.list_specs())
            assert decision.action == PlannerActionType.CALL_FRONTEND
            assert decision.selected_audio_ids == ["audio_1"]
            assert decision.frontend_followup_prompt is not None
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
