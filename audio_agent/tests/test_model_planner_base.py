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


class _FakeToolCall:
    """Minimal stand-in for an OpenAI SDK ``tool_call`` object.

    The parser only reads ``.function.name`` and ``.function.arguments``
    (a JSON string), so we mirror just that surface.
    """

    class _Function:
        def __init__(self, name: str, arguments_dict: dict | None):
            self.name = name
            self.arguments = json.dumps(arguments_dict or {})

    def __init__(self, name: str, args: dict | None):
        self.function = self._Function(name, args)


class _FakeMessage:
    """Minimal stand-in for an OpenAI SDK ``message`` object."""

    def __init__(self, content: str, tool_calls: list[_FakeToolCall]):
        self.content = content
        self.tool_calls = tool_calls


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
            "selected_tool_calls": [
                {"tool_name": "dummy_asr", "args": {"audio_path": "/tmp/audio.wav"}, "context": {}},
            ],
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

        After the native function-calling migration, the system message
        (rendered from ``decide_system.md``) carries:
          - the Role preamble
          - the full ``decide_rules.md`` text
          - ``tool_category_definitions`` (definition + guideline per
            category present in available_tools)
          - the How-To-Decide guidance for emitting tool calls

        The tool catalog itself is delivered via the ``tools=`` API
        parameter (see ``_to_openai_tools``), NOT rendered into the system
        text. The Output Contract is enforced by the tools' JSON schemas.

        The user message (rendered from ``decide_user.md``) carries only:
          - question, initial plan, planner reasoning trace, loop budget,
            audio list, unified evidence ledger, and a terminal "Decide"
            anchor.
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

        # --- System message: identity + rules + categories + how-to-decide ---
        assert "## Role" in system_content
        # decide_rules.md inlined
        assert "## Decision Rules" in system_content
        assert "Rationale Requirement" in system_content
        assert "Tool Parameter Rule" in system_content
        # tool category definitions inlined
        assert "## Tool Categories" in system_content
        assert '"category": "audio_derivation"' in system_content
        assert '"definition":' in system_content
        assert '"guideline":' in system_content
        # How-To-Decide guidance present
        assert "## How To Decide" in system_content
        assert "emit_final_answer" in system_content
        assert "ask_frontend" in system_content
        assert "give_up" in system_content

        # The catalog itself moves to the `tools=` API parameter —
        # it should NOT be rendered into the system text. Likewise the
        # legacy Output Contract section is gone.
        assert "## Available Tools" not in system_content
        assert "## Output Contract" not in system_content

        # --- User message: only iteration-volatile state ---
        assert "## Question" in user_content
        assert "Question" in user_content  # the question string itself
        assert "## Initial Plan" in user_content
        assert "## Planner Reasoning Trace" in user_content
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

        # --- Statics are NOT duplicated into the user message ---
        assert "Rationale Requirement" not in user_content
        assert "Tool Parameter Rule" not in user_content
        assert "## Tool Categories" not in user_content
        assert "## How To Decide" not in user_content

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

    # =========================================================================
    # Native function-calling (Option 2) tests
    # =========================================================================

    def test_to_openai_tools_wraps_real_and_appends_synthetic_actions(self):
        """``_to_openai_tools`` wraps each real ToolSpec into OpenAI tool
        format and appends the three synthetic action-tools at the end.
        """
        real = [
            ToolSpec(
                name="trim_audio",
                description="Trim a clip.",
                input_schema={
                    "type": "object",
                    "properties": {"audio_path": {"type": "string"}},
                    "required": ["audio_path"],
                },
            ),
        ]
        tools = BaseModelPlanner._to_openai_tools(real)
        assert len(tools) == 1 + 3  # one real + three synthetic

        names = [t["function"]["name"] for t in tools]
        assert names[0] == "trim_audio"
        assert "emit_final_answer" in names
        assert "ask_frontend" in names
        assert "give_up" in names

        # Real tool's input schema is passed through verbatim as parameters.
        assert tools[0]["function"]["parameters"]["properties"]["audio_path"]["type"] == "string"

        # Synthetic action tools each have a usable parameters schema.
        action_specs = {t["function"]["name"]: t["function"] for t in tools if t["function"]["name"] in {
            "emit_final_answer", "ask_frontend", "give_up"
        }}
        assert "rationale" in action_specs["emit_final_answer"]["parameters"]["properties"]
        assert "selected_audio_ids" in action_specs["ask_frontend"]["parameters"]["properties"]
        assert "reason" in action_specs["give_up"]["parameters"]["properties"]

    def test_parse_tool_calls_emit_final_answer_maps_to_answer_action(self):
        """A single emit_final_answer tool_call → PlannerDecision(ANSWER)."""
        msg = _FakeMessage(
            content="The evidence is sufficient — sample rate and duration are clear.",
            tool_calls=[
                _FakeToolCall("emit_final_answer", {"rationale": "all evidence in", "confidence": 0.9}),
            ],
        )
        decision = BaseModelPlanner._parse_tool_calls_to_decision(msg, available_tool_names=None)
        assert decision.action == PlannerActionType.ANSWER
        assert decision.selected_tool_calls == []
        assert "evidence is sufficient" in decision.rationale
        assert decision.confidence == 0.9

    def test_parse_tool_calls_action_tool_exclusivity_drops_other_calls(self):
        """When an action tool appears alongside other calls, the action
        wins and the other calls are discarded with a warning (caller-side
        behavior; here we only verify the resulting decision shape)."""
        msg = _FakeMessage(
            content="Done.",
            tool_calls=[
                _FakeToolCall("trim_audio", {"audio_path": "audio_0", "start": 0, "end": 5}),
                _FakeToolCall("emit_final_answer", {"rationale": "got it"}),
            ],
        )
        decision = BaseModelPlanner._parse_tool_calls_to_decision(msg, available_tool_names=None)
        assert decision.action == PlannerActionType.ANSWER
        # The other (real) tool call was discarded.
        assert decision.selected_tool_calls == []

    def test_parse_tool_calls_parallel_real_tools_become_call_tool(self):
        """Multiple real tool calls in one round → CALL_TOOL with all of
        them in ``selected_tool_calls`` (parallel emission)."""
        msg = _FakeMessage(
            content="Three independent analyses on the same audio.",
            tool_calls=[
                _FakeToolCall("get_audio_info", {"audio_path": "audio_0"}),
                _FakeToolCall("vad_predict", {"audio_path": "audio_0"}),
                _FakeToolCall("analyze_onsets", {"audio_path": "audio_0"}),
            ],
        )
        decision = BaseModelPlanner._parse_tool_calls_to_decision(
            msg,
            available_tool_names={"get_audio_info", "vad_predict", "analyze_onsets"},
        )
        assert decision.action == PlannerActionType.CALL_TOOL
        assert len(decision.selected_tool_calls) == 3
        names = [tc.tool_name for tc in decision.selected_tool_calls]
        assert names == ["get_audio_info", "vad_predict", "analyze_onsets"]
        assert decision.selected_audio_id == "audio_0"  # extracted from first call's args
        assert "Three independent" in decision.rationale

    def test_parse_tool_calls_ask_frontend_maps_to_call_frontend(self):
        """``ask_frontend`` tool_call → PlannerDecision(CALL_FRONTEND)."""
        msg = _FakeMessage(
            content="Re-perceive the trimmed clip.",
            tool_calls=[
                _FakeToolCall(
                    "ask_frontend",
                    {
                        "selected_audio_ids": ["audio_1"],
                        "frontend_followup_prompt": "What chord is being played?",
                        "frontend_followup_goal": "identify chord",
                    },
                ),
            ],
        )
        decision = BaseModelPlanner._parse_tool_calls_to_decision(msg, available_tool_names=None)
        assert decision.action == PlannerActionType.CALL_FRONTEND
        assert decision.selected_audio_ids == ["audio_1"]
        assert decision.frontend_followup_prompt == "What chord is being played?"
        assert decision.frontend_followup_goal == "identify chord"
        assert decision.selected_tool_calls == []

    def test_parse_tool_calls_give_up_maps_to_fail(self):
        """``give_up`` tool_call → PlannerDecision(FAIL)."""
        msg = _FakeMessage(
            content="",
            tool_calls=[
                _FakeToolCall("give_up", {"reason": "no ASR model can transcribe this audio"}),
            ],
        )
        decision = BaseModelPlanner._parse_tool_calls_to_decision(msg, available_tool_names=None)
        assert decision.action == PlannerActionType.FAIL
        assert "no ASR" in decision.rationale

    def test_parse_tool_calls_drops_unknown_tool_names(self):
        """Tool calls referencing names not in available_tool_names are
        dropped with a warning. If the remaining list is empty, raise."""
        msg = _FakeMessage(
            content="x",
            tool_calls=[
                _FakeToolCall("nonexistent_tool", {}),
                _FakeToolCall("get_audio_info", {"audio_path": "audio_0"}),
            ],
        )
        decision = BaseModelPlanner._parse_tool_calls_to_decision(
            msg, available_tool_names={"get_audio_info"}
        )
        # nonexistent_tool dropped; get_audio_info kept.
        assert decision.action == PlannerActionType.CALL_TOOL
        assert len(decision.selected_tool_calls) == 1
        assert decision.selected_tool_calls[0].tool_name == "get_audio_info"

    def test_parse_tool_calls_empty_tool_calls_raises(self):
        """No tool_calls at all → PlannerError (the tool_choice='required'
        contract was violated)."""
        msg = _FakeMessage(content="hello", tool_calls=[])
        with pytest.raises(PlannerError):
            BaseModelPlanner._parse_tool_calls_to_decision(msg, available_tool_names=None)

    def test_format_planner_reasoning_trace_renders_rounds(self):
        """The reasoning trace renders one numbered line per past decision;
        empty rationales show as ``(no reasoning recorded)``."""
        trace = [
            PlannerDecision(
                action=PlannerActionType.CALL_TOOL,
                rationale="Need ASR on this clip.",
                selected_tool_calls=[ToolCallRequest(tool_name="dummy", args={}, context={})],
            ),
            PlannerDecision(
                action=PlannerActionType.CALL_TOOL,
                rationale="",  # model didn't emit content this round
                selected_tool_calls=[ToolCallRequest(tool_name="dummy", args={}, context={})],
            ),
            PlannerDecision(
                action=PlannerActionType.ANSWER,
                rationale="Evidence sufficient.",
            ),
        ]
        rendered = BaseModelPlanner._format_planner_reasoning_trace(trace)
        assert "Round 1: Need ASR on this clip." in rendered
        assert "Round 2: (no reasoning recorded)" in rendered
        assert "Round 3: Evidence sufficient." in rendered

    def test_format_planner_reasoning_trace_empty(self):
        rendered = BaseModelPlanner._format_planner_reasoning_trace([])
        assert "no prior rounds" in rendered

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
        assert result.selected_tool_calls[0].tool_name == "dummy_asr"

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
