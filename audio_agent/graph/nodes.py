"""
LangGraph node functions for the audio agent.

Each node is a pure function that takes state and returns partial state updates.
Nodes follow fail-fast principles with explicit validation.
"""

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import (
    EvidenceItem,
    InitialPlan,
    PlannerDecision,
    PlannerActionType,
    ToolCallRequest,
    ToolCallRecord,
    FinalAnswer,
)
from audio_agent.core.constants import AgentStatus
from audio_agent.core.errors import (
    StateValidationError,
    FrontendError,
    PlannerError,
    ToolExecutionError,
    FusionError,
)
from audio_agent.core.logging import (
    log_node_start,
    log_node_end,
    log_planner_decision,
    log_error,
    log_state_transition,
)
from audio_agent.utils.validation import validate_state_has_fields
from audio_agent.frontend.base import BaseFrontend
from audio_agent.planner.base import BasePlanner
from audio_agent.tools.executor import ToolExecutor
from audio_agent.tools.registry import ToolRegistry
from audio_agent.fusion.base import BaseEvidenceFuser


def create_frontend_evidence_node(frontend: BaseFrontend):
    """
    Factory to create a frontend evidence node with the given frontend.
    
    Args:
        frontend: Frontend instance to use for audio processing
    
    Returns:
        Node function compatible with LangGraph
    """
    def frontend_evidence_node(state: AgentState) -> dict:
        """
        Process audio through frontend and generate initial evidence.
        
        Validates:
        - question and audio_path_or_uri are present
        
        Updates:
        - initial_frontend_output
        - evidence_log (appends initial question-guided caption evidence)
        """
        log_node_start("frontend_evidence_node", {
            "question": state.get("question", "")[:50],
            "audio": state.get("audio_path_or_uri", "")[:50],
        })
        
        validate_state_has_fields(
            state,
            ["question", "audio_path_or_uri"],
            context="frontend_evidence_node",
        )
        
        question = state["question"]
        audio_path = state["audio_path_or_uri"]
        
        # Run frontend
        try:
            output = frontend.run(question, audio_path)
        except FrontendError:
            raise
        except Exception as e:
            log_error("frontend_evidence_node", e)
            raise FrontendError(
                f"Frontend failed: {e}",
                details={"frontend": frontend.name}
            ) from e
        
        if output is None:
            raise FrontendError(
                "Frontend returned None",
                details={"frontend": frontend.name}
            )
        
        # Create initial evidence from frontend output
        initial_evidence = EvidenceItem(
            source=f"frontend:{frontend.name}",
            content=output.question_guided_caption,
            evidence_type="question_guided_caption",
            confidence=0.5,
            metadata={},
        )

        log_node_end(
            "frontend_evidence_node",
            {
                "caption_length": len(output.question_guided_caption),
            },
        )

        return {
            "initial_frontend_output": output,
            "evidence_log": [initial_evidence],
        }
    
    return frontend_evidence_node


def create_initial_plan_node(planner: BasePlanner):
    """
    Factory to create an initial planning node.

    Initial planning must depend only on question (not frontend caption).
    """

    def initial_plan_node(state: AgentState) -> dict:
        """
        Generate initial plan from question only.

        Validates:
        - question exists

        Updates:
        - initial_plan
        - initial_plan_trace (appends plan)
        """
        log_node_start("initial_plan_node", {
            "question": state.get("question", "")[:50],
        })

        validate_state_has_fields(
            state,
            ["question"],
            context="initial_plan_node",
        )

        question = state["question"]

        try:
            plan = planner.plan(question)
        except PlannerError:
            raise
        except Exception as e:
            log_error("initial_plan_node", e)
            raise PlannerError(
                f"Initial planning failed: {e}",
                details={"planner": planner.name},
            ) from e

        if plan is None:
            raise PlannerError(
                "Planner returned None for initial plan",
                details={"planner": planner.name},
            )
        if not isinstance(plan, InitialPlan):
            raise PlannerError(
                "Planner returned malformed initial plan type",
                details={"actual_type": type(plan).__name__},
            )

        log_node_end("initial_plan_node", {
            "focus_points": len(plan.focus_points),
            "possible_tool_types": len(plan.possible_tool_types),
            "clarified_intent": plan.clarified_intent,
            "expected_output_format": plan.expected_output_format,
        })

        return {
            "initial_plan": plan,
            "initial_plan_trace": [plan],
            "clarified_intent": plan.clarified_intent,
            "expected_output_format": plan.expected_output_format,
        }

    return initial_plan_node


def create_planner_decision_node(planner: BasePlanner, registry: ToolRegistry):
    """
    Factory to create an action decision planner node.
    
    Args:
        planner: Planner instance for action decision
        registry: Tool registry for available tools
    
    Returns:
        Node function compatible with LangGraph
    """
    def planner_decision_node(state: AgentState) -> dict:
        """
        Make an action decision based on current state.
        On final step, generates final answer instead of making decision.
        
        Validates:
        - question exists
        - initial_frontend_output exists
        - initial_plan exists
        
        Updates:
        - current_decision
        - planner_trace (appends decision)
        """
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 10)
        is_final_step = step_count >= max_steps - 1

        # Final step: generate answer directly
        if is_final_step:
            log_node_start("planner_decision_node", {
                "step_count": step_count,
                "mode": "final_answer",
            })

            try:
                answer_text = planner.answer(state)
            except PlannerError:
                raise
            except Exception as e:
                log_error("planner_decision_node", e)
                raise PlannerError(
                    f"Final answer generation failed: {e}",
                    details={"planner": planner.name}
                ) from e

            # Create ANSWER decision with generated answer
            decision = PlannerDecision(
                action=PlannerActionType.ANSWER,
                rationale=f"Maximum steps ({max_steps}) reached. Providing final answer based on accumulated evidence.",
                draft_answer=answer_text,
                confidence=0.7,
            )

            log_planner_decision("answer", decision.rationale, None)
            log_node_end("planner_decision_node", {"action": "answer", "mode": "final_answer"})

            return {
                "current_decision": decision,
                "planner_trace": [decision],
            }

        # Normal decision flow
        log_node_start("planner_decision_node", {
            "step_count": step_count,
            "evidence_count": len(state.get("evidence_log", [])),
        })
        
        validate_state_has_fields(
            state,
            ["question", "initial_frontend_output", "initial_plan"],
            context="planner_decision_node",
        )
        
        available_tools = registry.list_specs()
        
        try:
            decision = planner.decide(state, available_tools)
        except PlannerError:
            raise
        except Exception as e:
            log_error("planner_decision_node", e)
            raise PlannerError(
                f"Planner decision failed: {e}",
                details={"planner": planner.name}
            ) from e
        
        if decision is None:
            raise PlannerError(
                "Planner returned None",
                details={"planner": planner.name}
            )
        
        log_planner_decision(
            decision.action.value,
            decision.rationale,
            decision.selected_tool_name,
        )
        
        log_node_end("planner_decision_node", {"action": decision.action.value})
        
        return {
            "current_decision": decision,
            "planner_trace": [decision],
        }
    
    return planner_decision_node


def create_tool_executor_node(executor: ToolExecutor):
    """
    Factory to create a tool executor node.
    
    Args:
        executor: ToolExecutor instance for running tools
    
    Returns:
        Async node function compatible with LangGraph
    """
    async def tool_executor_node(state: AgentState) -> dict:
        """
        Execute the tool specified in current_decision.
        
        Validates:
        - current_decision exists and is CALL_TOOL
        - selected_tool_name is present
        
        Updates:
        - latest_tool_result
        - tool_call_history (appends record)
        """
        log_node_start("tool_executor_node")
        
        validate_state_has_fields(
            state,
            ["current_decision"],
            context="tool_executor_node",
        )
        
        decision: PlannerDecision = state["current_decision"]
        
        if decision.action != PlannerActionType.CALL_TOOL:
            raise StateValidationError(
                f"tool_executor_node called with non-CALL_TOOL action: {decision.action}",
                details={"action": decision.action.value}
            )
        
        if not decision.selected_tool_name:
            raise StateValidationError(
                "CALL_TOOL decision has no selected_tool_name",
                details={"decision": decision.model_dump()}
            )
        
        # Build request
        # Get args from planner decision, inject audio_path if needed
        args = decision.selected_tool_args or {}
        
        # Inject audio_path for tools that need it (e.g., ASR tools)
        # The planner doesn't know the audio path, so we inject it from state
        if "audio_path" not in args and state.get("audio_path_or_uri"):
            args = {**args, "audio_path": state["audio_path_or_uri"]}
        
        request = ToolCallRequest(
            tool_name=decision.selected_tool_name,
            args=args,
            context={
                "question": state.get("question", ""),
                "step_count": state.get("step_count", 0),
            },
        )
        
        # Execute (async)
        try:
            result = await executor.execute(request)
        except ToolExecutionError:
            raise
        except Exception as e:
            log_error("tool_executor_node", e)
            raise ToolExecutionError(
                f"Tool execution failed: {e}",
                details={"tool_name": decision.selected_tool_name}
            ) from e
        
        # Create history record
        record = ToolCallRecord(
            request=request,
            result=result,
            step_number=state.get("step_count", 0),
        )
        
        log_node_end("tool_executor_node", {
            "tool": decision.selected_tool_name,
            "success": result.success,
        })
        
        return {
            "latest_tool_result": result,
            "tool_call_history": [record],
        }
    
    return tool_executor_node


def create_evidence_fusion_node(fuser: BaseEvidenceFuser):
    """
    Factory to create an evidence fusion node.
    
    Args:
        fuser: Evidence fuser instance
    
    Returns:
        Node function compatible with LangGraph
    """
    def evidence_fusion_node(state: AgentState) -> dict:
        """
        Fuse latest tool result into evidence items.
        
        Validates:
        - latest_tool_result exists
        
        Updates:
        - evidence_log (appends fused evidence)
        - step_count (increments)
        - latest_tool_result (clears to None)
        """
        log_node_start("evidence_fusion_node")
        
        validate_state_has_fields(
            state,
            ["latest_tool_result"],
            context="evidence_fusion_node",
        )
        
        tool_result = state["latest_tool_result"]
        
        try:
            evidence_items = fuser.fuse(state, tool_result)
        except FusionError:
            raise
        except Exception as e:
            log_error("evidence_fusion_node", e)
            raise FusionError(
                f"Evidence fusion failed: {e}",
                details={"fuser": fuser.name}
            ) from e
        
        if evidence_items is None:
            raise FusionError(
                "Fuser returned None",
                details={"fuser": fuser.name}
            )
        
        current_step = state.get("step_count", 0)
        
        log_node_end("evidence_fusion_node", {
            "new_evidence_count": len(evidence_items),
            "step_count": current_step + 1,
        })
        
        return {
            "evidence_log": evidence_items,
            "step_count": current_step + 1,
            "latest_tool_result": None,
        }
    
    return evidence_fusion_node


def create_intent_clarification_node(planner: BasePlanner):
    """
    Factory to create an intent clarification node.
    
    Args:
        planner: Planner instance for intent clarification
    
    Returns:
        Node function compatible with LangGraph
    """
    def intent_clarification_node(state: AgentState) -> dict:
        """
        Clarify the user's intent and expected output format.
        
        Uses reasoning on accumulated evidence to refine or clarify intent.
        Does NOT call tools - if tools are needed, planner should CALL_TOOL first.
        
        Validates:
        - current_decision exists and is CLARIFY_INTENT
        
        Updates:
        - clarified_intent
        - expected_output_format
        - evidence_log (appends clarification as evidence)
        """
        log_node_start("intent_clarification_node")
        
        validate_state_has_fields(
            state,
            ["current_decision"],
            context="intent_clarification_node",
        )
        
        decision: PlannerDecision = state["current_decision"]
        
        if decision.action != PlannerActionType.CLARIFY_INTENT:
            raise StateValidationError(
                f"intent_clarification_node called with non-CLARIFY_INTENT action: {decision.action}",
                details={"action": decision.action.value}
            )
        
        # Call planner to clarify intent
        try:
            clarified_intent, expected_format = planner.clarify_intent(state)
        except PlannerError:
            raise
        except Exception as e:
            log_error("intent_clarification_node", e)
            raise PlannerError(
                f"Intent clarification failed: {e}",
                details={"planner": planner.name}
            ) from e
        
        if clarified_intent is None:
            raise PlannerError(
                "Planner returned None for clarified_intent",
                details={"planner": planner.name}
            )
        
        # Create evidence item for the clarification
        clarification_evidence = EvidenceItem(
            source=f"planner:{planner.name}:intent_clarification",
            content=f"Clarified intent: {clarified_intent}. Expected format: {expected_format or 'not specified'}",
            evidence_type="intent_clarification",
            confidence=0.8,
            metadata={
                "clarified_intent": clarified_intent,
                "expected_output_format": expected_format,
                "rationale": decision.rationale,
            },
        )
        
        log_node_end("intent_clarification_node", {
            "clarified_intent": clarified_intent[:100] if clarified_intent else None,
            "expected_output_format": expected_format[:100] if expected_format else None,
        })
        
        return {
            "clarified_intent": clarified_intent,
            "expected_output_format": expected_format,
            "evidence_log": [clarification_evidence],
        }
    
    return intent_clarification_node


def answer_node(state: AgentState) -> dict:
    """
    Finalize the agent with an answer.
    
    Validates:
    - current_decision exists and is ANSWER
    - draft_answer is present
    
    Updates:
    - final_answer
    - status (to ANSWERED)
    """
    log_node_start("answer_node")
    
    validate_state_has_fields(
        state,
        ["current_decision"],
        context="answer_node",
    )
    
    decision: PlannerDecision = state["current_decision"]
    
    if decision.action != PlannerActionType.ANSWER:
        raise StateValidationError(
            f"answer_node called with non-ANSWER action: {decision.action}",
            details={"action": decision.action.value}
        )
    
    if not decision.draft_answer:
        raise StateValidationError(
            "ANSWER decision has no draft_answer",
            details={"decision": decision.model_dump()}
        )
    
    # Build final answer
    evidence_log = state.get("evidence_log", [])
    evidence_summary = "\n".join(
        f"- [{e.source}] {e.content[:100]}..."
        for e in evidence_log
    )
    
    planner_trace = state.get("planner_trace", [])
    reasoning_trace = "\n".join(
        f"Step {i+1}: {d.action.value} - {d.rationale}"
        for i, d in enumerate(planner_trace)
    )
    
    final_answer = FinalAnswer(
        answer=decision.draft_answer,
        confidence=decision.confidence,
        evidence_summary=evidence_summary,
        reasoning_trace=reasoning_trace,
    )
    
    log_state_transition(
        state.get("status", AgentStatus.RUNNING).value,
        AgentStatus.ANSWERED.value,
        "Planner provided final answer",
    )
    
    log_node_end("answer_node", {"answer_length": len(final_answer.answer)})
    
    return {
        "final_answer": final_answer,
        "status": AgentStatus.ANSWERED,
    }


def failure_node(state: AgentState) -> dict:
    """
    Handle agent failure.
    
    Updates:
    - error_message
    - status (to FAILED or EXHAUSTED)
    """
    log_node_start("failure_node")
    
    decision = state.get("current_decision")
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 10)
    
    # Determine failure reason
    if step_count >= max_steps:
        error_message = f"Agent exhausted: reached max_steps ({max_steps})"
        new_status = AgentStatus.EXHAUSTED
    elif decision and decision.action == PlannerActionType.FAIL:
        error_message = f"Planner requested failure: {decision.rationale}"
        new_status = AgentStatus.FAILED
    else:
        error_message = "Agent failed for unknown reason"
        new_status = AgentStatus.FAILED
    
    log_state_transition(
        state.get("status", AgentStatus.RUNNING).value,
        new_status.value,
        error_message,
    )
    
    log_node_end("failure_node", {"status": new_status.value})
    
    return {
        "error_message": error_message,
        "status": new_status,
    }


# Convenience aliases for node creation
frontend_evidence_node = create_frontend_evidence_node
initial_plan_node = create_initial_plan_node
planner_decision_node = create_planner_decision_node
planner_node = create_planner_decision_node  # Backward-compatible alias
tool_executor_node = create_tool_executor_node
evidence_fusion_node = create_evidence_fusion_node
intent_clarification_node = create_intent_clarification_node
