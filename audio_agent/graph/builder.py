"""
LangGraph graph builder for the audio agent.

Constructs the agent workflow with proper node connections and routing.
"""

from langgraph.graph import StateGraph, START, END

from audio_agent.core.state import AgentState
from audio_agent.graph.nodes import (
    create_initial_prompt_node,
    create_frontend_evidence_node,
    create_initial_plan_node,
    create_planner_decision_node,
    create_tool_executor_node,
    create_evidence_fusion_node,
    create_intent_clarification_node,
    create_evidence_summarization_node,
    create_final_answer_node,
    create_format_check_node,
    answer_node,
    failure_node,
)
from audio_agent.graph.routing import (
    route_after_planner_decision,
    route_after_format_check,
    NODE_ANSWER,
    NODE_TOOL_EXECUTOR,
    NODE_FAILURE,
    NODE_EVIDENCE_FUSION,
    NODE_INITIAL_PROMPT,
    NODE_INITIAL_PLAN,
    NODE_PLANNER_DECISION,
    NODE_INTENT_CLARIFICATION,
    NODE_EVIDENCE_SUMMARIZATION,
    NODE_FINAL_ANSWER,
    NODE_FORMAT_CHECK,
)
from audio_agent.frontend.base import BaseFrontend
from audio_agent.planner.base import BasePlanner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.executor import ToolExecutor
from audio_agent.fusion.base import BaseEvidenceFuser


def build_graph(
    frontend: BaseFrontend,
    planner: BasePlanner,
    registry: ToolRegistry,
    fuser: BaseEvidenceFuser,
) -> StateGraph:
    """
    Build the complete audio agent LangGraph workflow.
    
    Graph structure:

    START
      -> initial_prompt_node
      -> frontend_evidence_node
      -> initial_plan_node
      -> planner_decision_node
      -> [conditional routing based on decision]
         - ANSWER -> final_answer_node -> format_check_node -> [conditional]
            * Format OK -> answer_node -> END
            * Format Failed -> planner_decision_node (loop with critique)
         - CALL_TOOL -> tool_executor_node -> evidence_fusion_node -> planner_decision_node (loop)
         - CLARIFY_INTENT -> intent_clarification_node -> planner_decision_node (loop)
         - FAIL -> failure_node -> END
    
    Args:
        frontend: Frontend instance for initial audio processing
        planner: Planner instance for decision making
        registry: Tool registry containing available tools
        fuser: Evidence fuser for converting tool results
    
    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    if frontend is None:
        raise ValueError("frontend cannot be None")
    if planner is None:
        raise ValueError("planner cannot be None")
    if registry is None:
        raise ValueError("registry cannot be None")
    if fuser is None:
        raise ValueError("fuser cannot be None")
    
    # Create executor from registry
    executor = ToolExecutor(registry)
    
    # Create node functions with injected dependencies
    initial_prompt_node_fn = create_initial_prompt_node(planner)
    initial_prompt_node_fn = create_initial_prompt_node(planner)
    frontend_node = create_frontend_evidence_node(frontend)
    initial_plan_node_fn = create_initial_plan_node(planner)
    planner_decision_node_fn = create_planner_decision_node(planner, registry)
    tool_executor_node_fn = create_tool_executor_node(executor)
    evidence_fusion_node_fn = create_evidence_fusion_node(fuser)
    intent_clarification_node_fn = create_intent_clarification_node(planner)
    evidence_summarization_node_fn = create_evidence_summarization_node(planner)
    final_answer_node_fn = create_final_answer_node(frontend)
    format_check_node_fn = create_format_check_node(planner)
    
    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node(NODE_INITIAL_PROMPT, initial_prompt_node_fn)
    graph.add_node("frontend_evidence_node", frontend_node)
    graph.add_node(NODE_INITIAL_PLAN, initial_plan_node_fn)
    graph.add_node(NODE_PLANNER_DECISION, planner_decision_node_fn)
    graph.add_node(NODE_TOOL_EXECUTOR, tool_executor_node_fn)
    graph.add_node(NODE_EVIDENCE_FUSION, evidence_fusion_node_fn)
    graph.add_node(NODE_INTENT_CLARIFICATION, intent_clarification_node_fn)
    graph.add_node(NODE_EVIDENCE_SUMMARIZATION, evidence_summarization_node_fn)
    graph.add_node(NODE_FINAL_ANSWER, final_answer_node_fn)
    graph.add_node(NODE_FORMAT_CHECK, format_check_node_fn)
    graph.add_node(NODE_ANSWER, answer_node)
    graph.add_node(NODE_FAILURE, failure_node)
    
    # Add edges
    # START -> initial_prompt_node
    graph.add_edge(START, NODE_INITIAL_PROMPT)
    
    # initial_prompt_node -> frontend_evidence_node
    graph.add_edge(NODE_INITIAL_PROMPT, "frontend_evidence_node")
    
    # frontend_evidence_node -> initial_plan_node
    graph.add_edge("frontend_evidence_node", NODE_INITIAL_PLAN)
    
    # initial_plan_node -> planner_decision_node
    graph.add_edge(NODE_INITIAL_PLAN, NODE_PLANNER_DECISION)
    
    # planner_decision_node -> conditional routing
    # Note: ANSWER now routes to format_check_node first (mandatory format check)
    graph.add_conditional_edges(
        NODE_PLANNER_DECISION,
        route_after_planner_decision,
        {
            NODE_EVIDENCE_SUMMARIZATION: NODE_EVIDENCE_SUMMARIZATION,
            NODE_TOOL_EXECUTOR: NODE_TOOL_EXECUTOR,
            NODE_INTENT_CLARIFICATION: NODE_INTENT_CLARIFICATION,
            NODE_FAILURE: NODE_FAILURE,
        }
    )
    
    # format_check_node -> conditional routing based on result
    graph.add_conditional_edges(
        NODE_FORMAT_CHECK,
        route_after_format_check,
        {
            NODE_ANSWER: NODE_ANSWER,
            NODE_PLANNER_DECISION: NODE_PLANNER_DECISION,
        }
    )
    
    # tool_executor_node -> evidence_fusion_node
    graph.add_edge(NODE_TOOL_EXECUTOR, NODE_EVIDENCE_FUSION)
    
    # evidence_fusion_node -> planner_decision_node (loop back)
    graph.add_edge(NODE_EVIDENCE_FUSION, NODE_PLANNER_DECISION)
    
    # evidence_summarization_node -> final_answer_node
    graph.add_edge(NODE_EVIDENCE_SUMMARIZATION, NODE_FINAL_ANSWER)
    
    # final_answer_node -> format_check_node
    graph.add_edge(NODE_FINAL_ANSWER, NODE_FORMAT_CHECK)

    # intent_clarification_node -> planner_decision_node (loop back)
    graph.add_edge(NODE_INTENT_CLARIFICATION, NODE_PLANNER_DECISION)
    
    # Terminal nodes -> END
    graph.add_edge(NODE_ANSWER, END)
    graph.add_edge(NODE_FAILURE, END)
    
    return graph.compile()


def build_graph_with_config(
    frontend: BaseFrontend,
    planner: BasePlanner,
    registry: ToolRegistry,
    fuser: BaseEvidenceFuser,
    checkpointer=None,
):
    """
    Build graph with optional checkpointing support.
    
    This is an extended version for future use with:
    - State persistence
    - Resumable executions
    - Debugging with history
    
    Args:
        frontend: Frontend instance
        planner: Planner instance
        registry: Tool registry
        fuser: Evidence fuser
        checkpointer: Optional LangGraph checkpointer
    
    Returns:
        Compiled graph with checkpointing
    """
    if frontend is None:
        raise ValueError("frontend cannot be None")
    if planner is None:
        raise ValueError("planner cannot be None")
    if registry is None:
        raise ValueError("registry cannot be None")
    if fuser is None:
        raise ValueError("fuser cannot be None")
    
    executor = ToolExecutor(registry)
    
    initial_prompt_node_fn = create_initial_prompt_node(planner)
    frontend_node = create_frontend_evidence_node(frontend)
    initial_plan_node_fn = create_initial_plan_node(planner)
    planner_decision_node_fn = create_planner_decision_node(planner, registry)
    tool_executor_node_fn = create_tool_executor_node(executor)
    evidence_fusion_node_fn = create_evidence_fusion_node(fuser)
    intent_clarification_node_fn = create_intent_clarification_node(planner)
    evidence_summarization_node_fn = create_evidence_summarization_node(planner)
    final_answer_node_fn = create_final_answer_node(frontend)
    format_check_node_fn = create_format_check_node(planner)

    graph = StateGraph(AgentState)

    graph.add_node(NODE_INITIAL_PROMPT, initial_prompt_node_fn)
    graph.add_node("frontend_evidence_node", frontend_node)
    graph.add_node(NODE_INITIAL_PLAN, initial_plan_node_fn)
    graph.add_node(NODE_PLANNER_DECISION, planner_decision_node_fn)
    graph.add_node(NODE_TOOL_EXECUTOR, tool_executor_node_fn)
    graph.add_node(NODE_EVIDENCE_FUSION, evidence_fusion_node_fn)
    graph.add_node(NODE_INTENT_CLARIFICATION, intent_clarification_node_fn)
    graph.add_node(NODE_EVIDENCE_SUMMARIZATION, evidence_summarization_node_fn)
    graph.add_node(NODE_FINAL_ANSWER, final_answer_node_fn)
    graph.add_node(NODE_FORMAT_CHECK, format_check_node_fn)
    graph.add_node(NODE_ANSWER, answer_node)
    graph.add_node(NODE_FAILURE, failure_node)
    
    graph.add_edge(START, NODE_INITIAL_PROMPT)
    graph.add_edge(NODE_INITIAL_PROMPT, "frontend_evidence_node")
    graph.add_edge("frontend_evidence_node", NODE_INITIAL_PLAN)
    graph.add_edge(NODE_INITIAL_PLAN, NODE_PLANNER_DECISION)
    
    graph.add_conditional_edges(
        NODE_PLANNER_DECISION,
        route_after_planner_decision,
        {
            NODE_EVIDENCE_SUMMARIZATION: NODE_EVIDENCE_SUMMARIZATION,
            NODE_TOOL_EXECUTOR: NODE_TOOL_EXECUTOR,
            NODE_INTENT_CLARIFICATION: NODE_INTENT_CLARIFICATION,
            NODE_FAILURE: NODE_FAILURE,
        }
    )
    
    graph.add_conditional_edges(
        NODE_FORMAT_CHECK,
        route_after_format_check,
        {
            NODE_ANSWER: NODE_ANSWER,
            NODE_PLANNER_DECISION: NODE_PLANNER_DECISION,
        }
    )
    
    graph.add_edge(NODE_TOOL_EXECUTOR, NODE_EVIDENCE_FUSION)
    graph.add_edge(NODE_EVIDENCE_FUSION, NODE_PLANNER_DECISION)
    graph.add_edge(NODE_EVIDENCE_SUMMARIZATION, NODE_FINAL_ANSWER)
    graph.add_edge(NODE_FINAL_ANSWER, NODE_FORMAT_CHECK)

    graph.add_edge(NODE_INTENT_CLARIFICATION, NODE_PLANNER_DECISION)
    
    graph.add_edge(NODE_ANSWER, END)
    graph.add_edge(NODE_FAILURE, END)
    
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
