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
    AudioItem,
    AudioOutput,
    VerificationResult,
    FormatCheckResult,
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
    log_warning,
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
        - question and audio_list are present
        - audio_0 (original audio) exists in audio_list
        
        Updates:
        - initial_frontend_output
        - evidence_log (appends initial question-guided caption evidence)
        """
        log_node_start("frontend_evidence_node", {
            "question": state.get("question", "")[:50],
            "audio_count": len(state.get("audio_list", [])),
        })
        
        validate_state_has_fields(
            state,
            ["question", "audio_list"],
            context="frontend_evidence_node",
        )
        
        question = state["question"]
        audio_list = state["audio_list"]
        
        # Find audio_0 (original audio) in the list
        original_audio = next(
            (a for a in audio_list if a.audio_id == "audio_0"),
            None
        )
        if original_audio is None:
            raise StateValidationError(
                "audio_0 (original audio) not found in audio_list",
                details={"audio_ids": [a.audio_id for a in audio_list]}
            )
        
        audio_path = original_audio.path
        
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
        - selected_audio_id is present and valid
        
        Updates:
        - latest_tool_result
        - tool_call_history (appends record)
        - audio_list (if tool generates new audio)
        """
        log_node_start("tool_executor_node")
        
        validate_state_has_fields(
            state,
            ["current_decision", "audio_list"],
            context="tool_executor_node",
        )
        
        decision: PlannerDecision = state["current_decision"]
        audio_list: list[AudioItem] = state["audio_list"]
        
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
        
        if not decision.selected_audio_id:
            raise StateValidationError(
                "CALL_TOOL decision has no selected_audio_id",
                details={"decision": decision.model_dump()}
            )
        
        # Find the selected audio in the list
        selected_audio = next(
            (a for a in audio_list if a.audio_id == decision.selected_audio_id),
            None
        )
        if selected_audio is None:
            raise StateValidationError(
                f"Audio '{decision.selected_audio_id}' not found in audio_list",
                details={
                    "selected_audio_id": decision.selected_audio_id,
                    "available_ids": [a.audio_id for a in audio_list],
                }
            )
        
        # Build request with the selected audio path
        args = decision.selected_tool_args or {}
        args = {**args, "audio_path": selected_audio.path}
        
        request = ToolCallRequest(
            tool_name=decision.selected_tool_name,
            args=args,
            context={
                "question": state.get("question", ""),
                "step_count": state.get("step_count", 0),
                "selected_audio_id": decision.selected_audio_id,
                "selected_audio_description": selected_audio.description,
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
        
        # Prepare return updates
        updates: dict = {
            "latest_tool_result": result,
            "tool_call_history": [record],
        }
        
        # If tool generates new audio, add it to audio_list
        if result.output.get("generated_audio_path"):
            new_audio_id = f"audio_{len(audio_list)}"
            new_audio = AudioItem(
                audio_id=new_audio_id,
                path=result.output["generated_audio_path"],
                source=decision.selected_tool_name,
                description=result.output.get(
                    "audio_description",
                    f"Generated by {decision.selected_tool_name}"
                ),
                metadata=result.output.get("audio_metadata", {}),
            )
            updates["audio_list"] = audio_list + [new_audio]
            log_node_end("tool_executor_node", {
                "tool": decision.selected_tool_name,
                "success": result.success,
                "new_audio_id": new_audio_id,
            })
        else:
            log_node_end("tool_executor_node", {
                "tool": decision.selected_tool_name,
                "success": result.success,
            })
        
        return updates
    
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


def create_verification_node(frontend: BaseFrontend):
    """
    Factory to create a verification node.
    
    Uses the frontend (audio model) to verify a proposed answer by
    reviewing it against the audio content.
    
    Args:
        frontend: Frontend instance with verify_answer capability
    
    Returns:
        Node function compatible with LangGraph
    """
    def verification_node(state: AgentState) -> dict:
        """
        Verify the proposed answer in current_decision.
        
        Validates:
        - current_decision exists and is VERIFY
        - draft_answer is present (the answer to verify)
        - audio_0 (original audio) exists
        
        Updates:
        - verification_result
        - verification_count (increments)
        - evidence_log (appends verification result as evidence if failed)
        """
        log_node_start("verification_node")
        
        validate_state_has_fields(
            state,
            ["current_decision", "audio_list", "question"],
            context="verification_node",
        )
        
        decision: PlannerDecision = state["current_decision"]
        audio_list: list[AudioItem] = state["audio_list"]
        question: str = state["question"]
        
        if decision.action != PlannerActionType.VERIFY:
            raise StateValidationError(
                f"verification_node called with non-VERIFY action: {decision.action}",
                details={"action": decision.action.value}
            )
        
        if not decision.draft_answer:
            raise StateValidationError(
                "VERIFY decision has no draft_answer",
                details={"decision": decision.model_dump()}
            )
        
        # Find audio_0 (original audio) in the list
        original_audio = next(
            (a for a in audio_list if a.audio_id == "audio_0"),
            None
        )
        if original_audio is None:
            raise StateValidationError(
                "audio_0 (original audio) not found in audio_list",
                details={"audio_ids": [a.audio_id for a in audio_list]}
            )
        
        audio_path = original_audio.path
        proposed_answer = decision.draft_answer
        
        # Call frontend to verify the answer
        try:
            verification_result = frontend.verify_answer(
                question=question,
                audio_path_or_uri=audio_path,
                proposed_answer=proposed_answer,
            )
        except FrontendError:
            raise
        except Exception as e:
            log_error("verification_node", e)
            raise FrontendError(
                f"Verification failed: {e}",
                details={"frontend": frontend.name}
            ) from e
        
        if verification_result is None:
            raise FrontendError(
                "Frontend returned None for verification",
                details={"frontend": frontend.name}
            )
        
        # Update verification count
        current_count = state.get("verification_count", 0)
        new_count = current_count + 1
        
        # Prepare return updates
        updates: dict = {
            "verification_result": verification_result,
            "verification_count": new_count,
        }
        
        # If verification failed, add critique as evidence
        if not verification_result.passed and verification_result.critique:
            critique_evidence = EvidenceItem(
                source=f"verification:{frontend.name}",
                content=f"Verification failed: {verification_result.critique}",
                evidence_type="verification_critique",
                confidence=verification_result.confidence,
                metadata={
                    "proposed_answer": proposed_answer[:200],  # Truncate for metadata
                    "verification_passed": verification_result.passed,
                },
            )
            updates["evidence_log"] = [critique_evidence]
            log_node_end("verification_node", {
                "passed": verification_result.passed,
                "confidence": verification_result.confidence,
                "critique": verification_result.critique[:100],
            })
        else:
            log_node_end("verification_node", {
                "passed": verification_result.passed,
                "confidence": verification_result.confidence,
            })
        
        return updates
    
    return verification_node


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
    - final_answer (with output_audio if applicable)
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
    
    # Determine output audio
    output_audio = None
    initial_plan = state.get("initial_plan")
    audio_list = state.get("audio_list", [])
    
    # Check if audio output is expected and available
    if initial_plan and initial_plan.requires_audio_output and audio_list:
        # Find the last non-original audio (most likely the output)
        generated_audios = [a for a in audio_list if a.source != "original"]
        if generated_audios:
            last_audio = generated_audios[-1]
            output_audio = AudioOutput(
                audio_id=last_audio.audio_id,
                path=last_audio.path,
                description=last_audio.description,
                metadata=last_audio.metadata,
            )
    
    # Build final answer with output_audio
    final_answer = FinalAnswer(
        answer=decision.draft_answer,
        confidence=decision.confidence,
        evidence_summary=evidence_summary,
        reasoning_trace=reasoning_trace,
        output_audio=output_audio,
    )
    
    # Log warning if audio was expected but not found
    if initial_plan and initial_plan.requires_audio_output and not output_audio:
        log_warning(
            "answer_node",
            {"message": "Audio output was expected but not found in audio_list"}
        )
    
    log_state_transition(
        state.get("status", AgentStatus.RUNNING).value,
        AgentStatus.ANSWERED.value,
        "Planner provided final answer",
    )
    
    log_node_end("answer_node", {
        "answer_length": len(final_answer.answer),
        "has_output_audio": output_audio is not None,
    })
    
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


def create_format_check_node(planner: BasePlanner):
    """
    Factory to create a format check node.
    
    Uses the planner (text LLM) to check if the proposed answer follows
    the expected output format requirements.
    
    Args:
        planner: Planner instance with check_format capability
    
    Returns:
        Node function compatible with LangGraph
    """
    def format_check_node(state: AgentState) -> dict:
        """
        Check the format of the proposed answer in current_decision.
        
        Validates:
        - current_decision exists and is ANSWER
        - draft_answer is present (the answer to check)
        
        Updates:
        - format_check_result
        - format_check_count (increments)
        - evidence_log (appends format critique as evidence if failed)
        """
        log_node_start("format_check_node")
        
        validate_state_has_fields(
            state,
            ["current_decision", "question"],
            context="format_check_node",
        )
        
        decision: PlannerDecision = state["current_decision"]
        question: str = state["question"]
        
        if decision.action != PlannerActionType.ANSWER:
            raise StateValidationError(
                f"format_check_node called with non-ANSWER action: {decision.action}",
                details={"action": decision.action.value}
            )
        
        if not decision.draft_answer:
            raise StateValidationError(
                "ANSWER decision has no draft_answer for format check",
                details={"decision": decision.model_dump()}
            )
        
        proposed_answer = decision.draft_answer
        expected_format = state.get("expected_output_format")
        
        # Call planner to check format
        try:
            format_check_result = planner.check_format(
                proposed_answer=proposed_answer,
                expected_format=expected_format,
                question=question,
            )
        except PlannerError:
            raise
        except Exception as e:
            log_error("format_check_node", e)
            raise PlannerError(
                f"Format check failed: {e}",
                details={"planner": planner.name}
            ) from e
        
        if format_check_result is None:
            raise PlannerError(
                "Planner returned None for format check",
                details={"planner": planner.name}
            )
        
        # Update format check count
        current_count = state.get("format_check_count", 0)
        new_count = current_count + 1
        
        # Prepare return updates
        updates: dict = {
            "format_check_result": format_check_result,
            "format_check_count": new_count,
        }
        
        # If format check failed, add critique as evidence
        if not format_check_result.passed and format_check_result.critique:
            critique_evidence = EvidenceItem(
                source=f"format_check:{planner.name}",
                content=f"Format check failed: {format_check_result.critique}",
                evidence_type="format_critique",
                confidence=format_check_result.confidence,
                metadata={
                    "proposed_answer": proposed_answer[:200],  # Truncate for metadata
                    "format_check_passed": format_check_result.passed,
                    "expected_format": expected_format,
                },
            )
            updates["evidence_log"] = [critique_evidence]
            log_node_end("format_check_node", {
                "passed": format_check_result.passed,
                "confidence": format_check_result.confidence,
                "critique": format_check_result.critique[:100],
            })
        else:
            log_node_end("format_check_node", {
                "passed": format_check_result.passed,
                "confidence": format_check_result.confidence,
            })
        
        return updates
    
    return format_check_node


# Convenience aliases for node creation
frontend_evidence_node = create_frontend_evidence_node
initial_plan_node = create_initial_plan_node
planner_decision_node = create_planner_decision_node
planner_node = create_planner_decision_node  # Backward-compatible alias
tool_executor_node = create_tool_executor_node
evidence_fusion_node = create_evidence_fusion_node
verification_node = create_verification_node
format_check_node = create_format_check_node
intent_clarification_node = create_intent_clarification_node
