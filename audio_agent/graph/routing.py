"""
Routing logic for the LangGraph workflow.

Routers examine state and return the name of the next node.
All routing decisions are explicit and logged.
"""

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import PlannerActionType
from audio_agent.core.constants import AgentStatus
from audio_agent.core.errors import GraphRoutingError
from audio_agent.core.logging import get_logger


# Node name constants
NODE_INITIAL_PLAN = "initial_plan_node"
NODE_PLANNER_DECISION = "planner_decision_node"
NODE_ANSWER = "answer_node"
NODE_TOOL_EXECUTOR = "tool_executor_node"
NODE_FAILURE = "failure_node"
NODE_EVIDENCE_FUSION = "evidence_fusion_node"
NODE_PLANNER = NODE_PLANNER_DECISION  # Backward-compatible alias
END = "__end__"


def route_after_planner_decision(state: AgentState) -> str:
    """
    Route after the planner decision node based on its decision.
    
    Routes:
    - ANSWER -> answer_node
    - CALL_TOOL -> tool_executor_node
    - FAIL -> failure_node
    
    Also checks for max_steps exhaustion.
    
    Args:
        state: Current agent state
    
    Returns:
        Name of the next node
    
    Raises:
        GraphRoutingError: If routing cannot be determined
    """
    logger = get_logger()
    
    # Check for exhaustion first
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 10)
    
    if step_count >= max_steps:
        logger.info(f"ROUTING: step_count ({step_count}) >= max_steps ({max_steps}) -> {NODE_FAILURE}")
        return NODE_FAILURE
    
    # Get decision
    decision = state.get("current_decision")
    
    if decision is None:
        raise GraphRoutingError(
            "Cannot route: current_decision is None",
            details={"step_count": step_count}
        )
    
    action = decision.action
    
    if action == PlannerActionType.ANSWER:
        logger.info(f"ROUTING: action={action.value} -> {NODE_ANSWER}")
        return NODE_ANSWER
    
    elif action == PlannerActionType.CALL_TOOL:
        tool_name = decision.selected_tool_name
        logger.info(f"ROUTING: action={action.value}, tool={tool_name} -> {NODE_TOOL_EXECUTOR}")
        return NODE_TOOL_EXECUTOR
    
    elif action == PlannerActionType.FAIL:
        logger.info(f"ROUTING: action={action.value} -> {NODE_FAILURE}")
        return NODE_FAILURE
    
    else:
        raise GraphRoutingError(
            f"Unknown planner action: {action}",
            details={"action": str(action)}
        )


def route_after_planner(state: AgentState) -> str:
    """Backward-compatible alias for planner decision routing."""
    return route_after_planner_decision(state)


def route_after_tool(state: AgentState) -> str:
    """
    Route after tool execution.
    
    Always routes to evidence_fusion_node to process the result.
    
    Args:
        state: Current agent state
    
    Returns:
        Name of the next node
    """
    logger = get_logger()
    
    # Check if tool result exists
    if state.get("latest_tool_result") is None:
        raise GraphRoutingError(
            "Cannot route after tool: latest_tool_result is None"
        )
    
    logger.info(f"ROUTING: after tool -> {NODE_EVIDENCE_FUSION}")
    return NODE_EVIDENCE_FUSION


def route_after_fusion(state: AgentState) -> str:
    """
    Route after evidence fusion.
    
    Always loops back to planner for next decision.
    
    Args:
        state: Current agent state
    
    Returns:
        Name of the next node
    """
    logger = get_logger()
    logger.info(f"ROUTING: after fusion -> {NODE_PLANNER_DECISION}")
    return NODE_PLANNER_DECISION


def is_terminal_state(state: AgentState) -> bool:
    """
    Check if the agent has reached a terminal state.
    
    Terminal states:
    - status == ANSWERED
    - status == FAILED
    - status == EXHAUSTED
    
    Returns:
        True if agent should stop
    """
    status = state.get("status", AgentStatus.RUNNING)
    return status in (AgentStatus.ANSWERED, AgentStatus.FAILED, AgentStatus.EXHAUSTED)
