"""Tests for the model-backed planner base class."""

import json

import pytest

from audio_agent.core.errors import PlannerError
from audio_agent.core.schemas import (
    EvidenceItem,
    FrontendOutput,
    InitialPlan,
    PlannerActionType,
    PlannerDecision,
    ToolCallRecord,
    ToolCallRequest,
    ToolResult,
    ToolSpec,
)
from audio_agent.core.state import create_initial_state
from audio_agent.config.settings import AgentConfig
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

    def test_build_decision_model_input_separates_static_and_dynamic(self):
        """Static identity/contract content lives in the system message;
        iteration-volatile state lives in the user message.

        System message (rendered from decide_system.md) carries:
          - the role preamble
          - full decide_rules.md text
          - tool_category_definitions (definition + guideline per category)
          - the available_tools catalog
          - the Required Output Format JSON schema

        User message (rendered from decide_user.md) carries only:
          - question, initial plan, loop budget, audio list,
            evidence log, tool call history, and a terminal "Decide" anchor.
        """
        planner = EchoModelPlanner()
        state = create_initial_state(
            "Question",
            "/tmp/audio.wav",
            config=AgentConfig().model_dump(),
        )
        state["initial_frontend_output"] = FrontendOutput(question_guided_caption="caption")
        state["initial_plan"] = planner.plan("Question")
        tools = [
            ToolSpec(
                name="trim_audio",
                description=(
                    "Function: Cut a selected time range into a derived audio clip.\n"
                    "Category: audio_derivation"
                ),
            )
        ]

        model_input = planner.build_decision_model_input(state, tools)
        system_content = model_input.messages[0]["content"]
        user_content = model_input.messages[1]["content"]

        # --- System message: 4 static blocks present ---
        # decide_rules.md inlined
        assert "## Decision Rules" in system_content
        assert "Rationale Requirement" in system_content
        assert "Tool Parameter Rule" in system_content
        # tool category definitions inlined
        assert "## Tool Categories" in system_content
        assert '"category": "audio_derivation"' in system_content
        assert '"definition":' in system_content
        assert '"guideline":' in system_content
        # tool catalog inlined
        assert "## Available Tools" in system_content
        assert '"name": "trim_audio"' in system_content
        # output contract / schema inlined
        assert "## Output Contract" in system_content
        assert '"action": "answer | call_tool | call_frontend | fail"' in system_content

        # --- User message: only iteration-volatile state ---
        assert "## Question" in user_content
        assert "Question" in user_content  # the question string itself
        assert "## Initial Plan" in user_content
        assert "## Loop Budget" in user_content
        assert "Step 0 of" in user_content
        assert "## Available Audio Files" in user_content
        assert "## Evidence Ledger" in user_content
        # Evidence Log + Tool Call History were merged into the Evidence
        # Ledger; their old section headers should be gone.
        assert "## Evidence Log" not in user_content
        assert "## Tool Call History" not in user_content
        assert "## Decide" in user_content
        # The user message is not a JSON envelope.
        with pytest.raises(json.JSONDecodeError):
            json.loads(user_content)

        # --- The 4 statics are NOT duplicated into the user message ---
        assert "Rationale Requirement" not in user_content
        assert "Tool Parameter Rule" not in user_content
        assert "## Tool Categories" not in user_content
        assert "## Available Tools" not in user_content
        assert "## Output Contract" not in user_content

        # --- No leftover placeholders in either message ---
        import re
        for label, content in (("system", system_content), ("user", user_content)):
            leftover = re.findall(r"\{([a-z_][a-z_0-9]*)\}", content)
            assert leftover == [], f"Unsubstituted placeholders in {label}: {leftover}"

    def test_build_evidence_ledger_merges_frontend_and_tool_entries(self):
        """The unified ledger interleaves frontend / tool / followup
        evidence in chronological order. Tool-derived entries (evidence_type
        ``"tool_output"`` or ``"error"``) pick up their matching ToolCallRecord
        by FIFO order and gain ``step``, ``args``, ``success``, ``output_keys``.
        Non-tool entries pass through unchanged.
        """
        evidence_log = [
            EvidenceItem(
                source="frontend:qwen3.5-omni-plus",
                content="<frontend caption>",
                evidence_type="question_guided_caption",
                confidence=0.5,
            ),
            EvidenceItem(
                source="trim_audio",
                content="Trimmed audio_0 → audio_1",
                evidence_type="tool_output",
                confidence=0.9,
            ),
            EvidenceItem(
                source="transcribe_qwenasr",
                content="Tool failed: model offline",
                evidence_type="error",
                confidence=0.0,
            ),
            EvidenceItem(
                source="frontend:qwen3.5-omni-plus:followup",
                content="<refined caption on audio_1>",
                evidence_type="frontend_followup",
                confidence=0.7,
            ),
        ]
        tool_history = [
            ToolCallRecord(
                request=ToolCallRequest(
                    tool_name="trim_audio",
                    args={"audio_path": "audio_0", "start_sec": 5.0, "end_sec": 10.0},
                ),
                result=ToolResult(
                    tool_name="trim_audio",
                    success=True,
                    output={"generated_audio_path": "/tmp/audio_1.wav", "duration_sec": 5.0},
                ),
                step_number=1,
            ),
            ToolCallRecord(
                request=ToolCallRequest(
                    tool_name="transcribe_qwenasr",
                    args={"audio_path": "audio_1"},
                ),
                result=ToolResult(
                    tool_name="transcribe_qwenasr",
                    success=False,
                    output={},
                    error_message="model offline",
                ),
                step_number=2,
            ),
        ]

        ledger = BaseModelPlanner._build_evidence_ledger(evidence_log, tool_history)

        assert len(ledger) == 4

        # Entry 0: frontend caption — no tool fields
        assert ledger[0]["source"] == "frontend:qwen3.5-omni-plus"
        assert ledger[0]["type"] == "question_guided_caption"
        assert "step" not in ledger[0]
        assert "args" not in ledger[0]

        # Entry 1: tool_output → paired with tool_history[0]
        assert ledger[1]["source"] == "trim_audio"
        assert ledger[1]["type"] == "tool_output"
        assert ledger[1]["step"] == 1
        assert ledger[1]["args"] == {"audio_path": "audio_0", "start_sec": 5.0, "end_sec": 10.0}
        assert ledger[1]["success"] is True
        assert "generated_audio_path" in ledger[1]["output_keys"]

        # Entry 2: error → paired with tool_history[1]
        assert ledger[2]["source"] == "transcribe_qwenasr"
        assert ledger[2]["type"] == "error"
        assert ledger[2]["step"] == 2
        assert ledger[2]["args"] == {"audio_path": "audio_1"}
        assert ledger[2]["success"] is False
        assert ledger[2]["output_keys"] == []

        # Entry 3: followup — no tool fields
        assert ledger[3]["source"] == "frontend:qwen3.5-omni-plus:followup"
        assert ledger[3]["type"] == "frontend_followup"
        assert "step" not in ledger[3]

    def test_build_evidence_ledger_handles_empty_inputs(self):
        """Empty inputs return an empty ledger; lone evidence with no tool
        history still passes through (just without tool-side enrichment)."""
        assert BaseModelPlanner._build_evidence_ledger([], []) == []
        only_frontend = [
            EvidenceItem(
                source="frontend:x",
                content="cap",
                evidence_type="question_guided_caption",
            )
        ]
        out = BaseModelPlanner._build_evidence_ledger(only_frontend, [])
        assert len(out) == 1
        assert out[0]["source"] == "frontend:x"
        assert "step" not in out[0]

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
