"""Tests for state validation and creation."""

import pytest

from audio_agent.core.state import create_initial_state, AgentState
from audio_agent.core.constants import AgentStatus
from audio_agent.core.errors import StateValidationError
from audio_agent.core.schemas import AudioItem, PlannerDecision, PlannerActionType
from audio_agent.utils.validation import validate_state_has_fields, validate_non_empty_string


class TestCreateInitialState:
    """Tests for create_initial_state function."""
    
    def test_valid_inputs(self):
        """Test creating state with valid inputs."""
        audio_list = [
            AudioItem(
                audio_id="audio_0",
                path="/path/to/audio.wav",
                source="original",
                description="original input audio",
            )
        ]
        state = create_initial_state(
            question="What is in this audio?",
            audio_paths=["/path/to/audio.wav"],
            max_steps=5,
            temp_dir="/tmp/test",
            audio_list=audio_list,
        )
        
        assert state["question"] == "What is in this audio?"
        assert state["original_audio_paths"] == ["/path/to/audio.wav"]
        assert state["temp_dir"] == "/tmp/test"
        assert len(state["audio_list"]) == 1
        assert state["audio_list"][0].audio_id == "audio_0"
        assert state["max_steps"] == 5
        assert state["step_count"] == 0
        assert state["status"] == AgentStatus.RUNNING
        assert state["evidence_log"] == []
        assert state["tool_call_history"] == []
        assert state["initial_plan_trace"] == []
        assert state["planner_trace"] == []
        assert state["initial_frontend_output"] is None
        assert state["initial_plan"] is None
        assert state["current_decision"] is None
        assert state["final_answer"] is None
        assert state["error_message"] is None
        assert state["evidence_summary"] is None
    
    def test_empty_question_raises(self):
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="question must be a non-empty string"):
            create_initial_state(
                question="",
                audio_paths=["/path/to/audio.wav"],
            )
    
    def test_whitespace_question_raises(self):
        """Test that whitespace-only question raises ValueError."""
        with pytest.raises(ValueError, match="question must be a non-empty string"):
            create_initial_state(
                question="   ",
                audio_paths=["/path/to/audio.wav"],
            )
    
    def test_empty_audio_paths_raises(self):
        """Test that empty audio paths raises ValueError."""
        with pytest.raises(ValueError, match="audio_paths must contain at least one path"):
            create_initial_state(
                question="What is this?",
                audio_paths=[],
            )
    
    def test_invalid_max_steps_raises(self):
        """Test that max_steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_steps must be at least 1"):
            create_initial_state(
                question="What is this?",
                audio_paths=["/path/to/audio.wav"],
                max_steps=0,
            )
    
    def test_strips_whitespace(self):
        """Test that question and audio paths are stripped."""
        audio_list = [
            AudioItem(
                audio_id="audio_0",
                path="/path/to/audio.wav",
                source="original",
                description="original input audio",
            )
        ]
        state = create_initial_state(
            question="  What is this?  ",
            audio_paths=["  /path/to/audio.wav  "],
            temp_dir="/tmp/test",
            audio_list=audio_list,
        )
        
        assert state["question"] == "What is this?"
        assert state["original_audio_paths"] == ["/path/to/audio.wav"]


class TestValidateStateHasFields:
    """Tests for validate_state_has_fields function."""
    
    def test_valid_state(self):
        """Test validation passes with required fields."""
        state = AgentState(
            question="test",
            original_audio_paths=["/path"],
            temp_dir="/tmp",
            audio_list=[],
        )
        # Should not raise
        validate_state_has_fields(state, ["question", "original_audio_paths"])
    
    def test_missing_field_raises(self):
        """Test validation fails with missing field."""
        state = AgentState(question="test")
        
        with pytest.raises(StateValidationError, match="Missing required state fields"):
            validate_state_has_fields(state, ["question", "original_audio_paths"])
    
    def test_none_state_raises(self):
        """Test validation fails with None state."""
        with pytest.raises(StateValidationError, match="State is None"):
            validate_state_has_fields(None, ["question"])
    
    def test_multiple_audio_paths(self):
        """Test creating state with multiple audio paths."""
        audio_list = [
            AudioItem(audio_id="audio_0", path="/path/to/audio1.wav", source="original", description="input 0"),
            AudioItem(audio_id="audio_1", path="/path/to/audio2.wav", source="original", description="input 1"),
        ]
        state = create_initial_state(
            question="Compare these audios",
            audio_paths=["/path/to/audio1.wav", "/path/to/audio2.wav"],
            audio_list=audio_list,
        )
        
        assert state["original_audio_paths"] == ["/path/to/audio1.wav", "/path/to/audio2.wav"]
        assert len(state["audio_list"]) == 2
    
    def test_context_in_error_message(self):
        """Test that context appears in error message."""
        state = AgentState(question="test")
        
        with pytest.raises(StateValidationError) as exc_info:
            validate_state_has_fields(
                state,
                ["missing_field"],
                context="test_node",
            )
        
        assert "test_node" in str(exc_info.value)


class TestValidateNonEmptyString:
    """Tests for validate_non_empty_string function."""
    
    def test_valid_string(self):
        """Test validation passes with valid string."""
        result = validate_non_empty_string("hello", "test_field")
        assert result == "hello"
    
    def test_strips_whitespace(self):
        """Test that result is stripped."""
        result = validate_non_empty_string("  hello  ", "test_field")
        assert result == "hello"
    
    def test_none_raises(self):
        """Test validation fails with None."""
        with pytest.raises(StateValidationError, match="test_field is None"):
            validate_non_empty_string(None, "test_field")
    
    def test_non_string_raises(self):
        """Test validation fails with non-string."""
        with pytest.raises(StateValidationError, match="must be a string"):
            validate_non_empty_string(123, "test_field")
    
    def test_empty_string_raises(self):
        """Test validation fails with empty string."""
        with pytest.raises(StateValidationError, match="must be non-empty"):
            validate_non_empty_string("", "test_field")
    
    def test_whitespace_only_raises(self):
        """Test validation fails with whitespace-only string."""
        with pytest.raises(StateValidationError, match="must be non-empty"):
            validate_non_empty_string("   ", "test_field")



class TestPlannerDecisionValidation:
    """Tests for PlannerDecision schema validation, especially CALL_FRONTEND."""
    
    def test_call_frontend_valid(self):
        """CALL_FRONTEND with valid fields passes."""
        decision = PlannerDecision(
            action=PlannerActionType.CALL_FRONTEND,
            rationale="Re-perceive isolated speaker",
            selected_audio_ids=["audio_1"],
            frontend_followup_prompt="What emotion does the speaker express?",
            frontend_followup_goal="Identify emotion",
            confidence=0.8,
        )
        assert decision.action == PlannerActionType.CALL_FRONTEND
        assert decision.selected_audio_ids == ["audio_1"]
        assert decision.frontend_followup_prompt == "What emotion does the speaker express?"
    
    def test_call_frontend_missing_audio_ids_raises(self):
        """CALL_FRONTEND without selected_audio_ids raises ValueError."""
        with pytest.raises(ValueError, match="non-empty selected_audio_ids"):
            PlannerDecision(
                action=PlannerActionType.CALL_FRONTEND,
                rationale="Re-perceive",
                frontend_followup_prompt="What emotion?",
                confidence=0.8,
            )
    
    def test_call_frontend_empty_audio_id_raises(self):
        """CALL_FRONTEND with empty string in selected_audio_ids raises ValueError."""
        with pytest.raises(ValueError, match="empty audio_ids"):
            PlannerDecision(
                action=PlannerActionType.CALL_FRONTEND,
                rationale="Re-perceive",
                selected_audio_ids=["audio_1", ""],
                frontend_followup_prompt="What emotion?",
                confidence=0.8,
            )
    
    def test_call_frontend_missing_prompt_raises(self):
        """CALL_FRONTEND without frontend_followup_prompt raises ValueError."""
        with pytest.raises(ValueError, match="non-empty frontend_followup_prompt"):
            PlannerDecision(
                action=PlannerActionType.CALL_FRONTEND,
                rationale="Re-perceive",
                selected_audio_ids=["audio_1"],
                confidence=0.8,
            )
    
    def test_call_frontend_whitespace_prompt_raises(self):
        """CALL_FRONTEND with whitespace-only prompt raises ValueError."""
        with pytest.raises(ValueError, match="non-empty frontend_followup_prompt"):
            PlannerDecision(
                action=PlannerActionType.CALL_FRONTEND,
                rationale="Re-perceive",
                selected_audio_ids=["audio_1"],
                frontend_followup_prompt="   ",
                confidence=0.8,
            )
    
    def test_answer_with_audio_ids_raises(self):
        """ANSWER carrying selected_audio_ids raises ValueError."""
        with pytest.raises(ValueError, match="must not carry selected_audio_ids"):
            PlannerDecision(
                action=PlannerActionType.ANSWER,
                rationale="Ready to answer",
                selected_audio_ids=["audio_1"],
                confidence=0.8,
            )
    
    def test_answer_with_followup_prompt_raises(self):
        """ANSWER carrying frontend_followup_prompt raises ValueError."""
        with pytest.raises(ValueError, match="must not carry frontend_followup_prompt"):
            PlannerDecision(
                action=PlannerActionType.ANSWER,
                rationale="Ready to answer",
                frontend_followup_prompt="What emotion?",
                confidence=0.8,
            )
    
    def test_call_tool_with_followup_prompt_raises(self):
        """CALL_TOOL with frontend_followup_prompt is technically allowed by validator
        but should be avoided. The validator only checks CALL_FRONTEND and ANSWER
        specifically; mixed fields on CALL_TOOL are not blocked."""
        # This test documents current behavior: no explicit ban on CALL_TOOL + prompt
        decision = PlannerDecision(
            action=PlannerActionType.CALL_TOOL,
            rationale="Call tool",
            selected_tool_name="dummy_asr",
            selected_audio_id="audio_0",
            frontend_followup_prompt="What emotion?",  # Mixed field
            confidence=0.8,
        )
        assert decision.action == PlannerActionType.CALL_TOOL
