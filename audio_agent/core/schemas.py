"""
Pydantic schemas for structured data throughout the audio agent framework.

These schemas define explicit contracts between components.
All schemas use Pydantic v2 with strict validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Enums
# =============================================================================

class PlannerActionType(str, Enum):
    """
    Actions the planner can decide to take.
    
    - ANSWER: Provide final answer based on accumulated evidence
    - CALL_TOOL: Invoke a tool to gather more evidence
    - CLARIFY_INTENT: Clarify the user's intent and expected output format
    - FAIL: Stop with explicit failure (unrecoverable state)
    """
    ANSWER = "answer"
    CALL_TOOL = "call_tool"
    CLARIFY_INTENT = "clarify_intent"
    FAIL = "fail"


# =============================================================================
# Frontend Schemas
# =============================================================================

class FrontendInput(BaseModel):
    """Input to the frontend module."""
    question: str = Field(..., min_length=1, description="User question about the audio")
    audio_path_or_uri: str = Field(..., min_length=1, description="Path or URI to audio file")


class FrontendOutput(BaseModel):
    """
    Output from the frontend module.
    
    The frontend (LALM) produces question-guided initial evidence.
    This output is intentionally lightweight for downstream planning, not final answering.
    """
    question_guided_caption: str = Field(
        ...,
        min_length=1,
        description="Concise caption focused on information relevant to the user question",
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_frontend_output(self) -> "FrontendOutput":
        """Fail-fast validation for frontend output structure."""
        if not self.question_guided_caption.strip():
            raise ValueError("question_guided_caption must be non-empty after stripping")

        return self


# =============================================================================
# Evidence Schemas
# =============================================================================

class EvidenceItem(BaseModel):
    """
    A single piece of evidence accumulated during agent execution.
    
    Evidence can come from the frontend, tools, or fusion operations.
    """
    source: str = Field(..., description="Source of this evidence (frontend, tool name, etc.)")
    content: str = Field(..., description="The evidence content")
    evidence_type: str = Field(default="text", description="Type of evidence: text, structured, etc.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Tool Schemas
# =============================================================================

class ToolSpec(BaseModel):
    """
    Specification of a tool, exposed to the planner for decision-making.
    
    This is a planner-friendly view of a tool's capabilities.
    """
    name: str = Field(..., min_length=1, description="Unique tool identifier")
    description: str = Field(..., description="What this tool does")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="Expected input format")
    output_schema: dict[str, Any] = Field(default_factory=dict, description="Expected output format")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class ToolCallRequest(BaseModel):
    """Request to invoke a tool."""
    tool_name: str = Field(..., min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict, description="Optional context for the tool")


class ToolResult(BaseModel):
    """Result returned by a tool after execution."""
    tool_name: str = Field(...)
    success: bool = Field(...)
    output: dict[str, Any] = Field(default_factory=dict, description="Structured output from the tool")
    error_message: str | None = Field(default=None)
    execution_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolCallRecord(BaseModel):
    """
    Record of a tool call for history tracking.
    
    Stores both the request and result together.
    """
    request: ToolCallRequest
    result: ToolResult
    step_number: int = Field(ge=0)


# =============================================================================
# Planner Schemas
# =============================================================================

class PlannerInput(BaseModel):
    """
    Input provided to the planner for decision-making.
    
    This is a structured view of the current agent state.
    """
    question: str
    evidence_log: list[EvidenceItem]
    tool_call_history: list[ToolCallRecord]
    available_tools: list[ToolSpec]
    step_count: int
    max_steps: int


class InitialPlan(BaseModel):
    """
    Initial high-level plan generated from the question only.

    This plan guides downstream action decisions but is not itself a tool call decision.
    Includes clarified intent and expected output format extracted from the question.
    """

    approach: str = Field(..., min_length=1, description="High-level strategy for answering")
    focus_points: list[str] = Field(
        default_factory=list,
        description="Key aspects to focus on while gathering evidence",
    )
    possible_tool_types: list[str] = Field(
        default_factory=list,
        description="Possible tool categories that may help later",
    )
    notes: str | None = Field(default=None, description="Optional concise planning notes")
    clarified_intent: str | None = Field(
        default=None,
        description="What the question is actually asking (extracted from question)",
    )
    expected_output_format: str | None = Field(
        default=None,
        description="Expected format of the final answer (extracted from question)",
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_plan(self) -> "InitialPlan":
        """Fail-fast validation for planning output."""
        if not self.approach.strip():
            raise ValueError("InitialPlan.approach must be non-empty after stripping")
        for item in self.focus_points:
            if not item or not item.strip():
                raise ValueError("InitialPlan.focus_points cannot contain empty strings")
        for item in self.possible_tool_types:
            if not item or not item.strip():
                raise ValueError("InitialPlan.possible_tool_types cannot contain empty strings")
        return self


class PlannerDecision(BaseModel):
    """
    Decision made by the planner.
    
    Validation ensures consistency:
    - CALL_TOOL requires selected_tool_name
    - ANSWER requires draft_answer
    """
    action: PlannerActionType
    rationale: str = Field(..., min_length=1, description="Explanation for the decision")
    selected_tool_name: str | None = Field(default=None)
    selected_tool_args: dict[str, Any] = Field(default_factory=dict)
    draft_answer: str | None = Field(default=None)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_action_consistency(self) -> "PlannerDecision":
        """Ensure action-specific fields are populated."""
        if self.action == PlannerActionType.CALL_TOOL:
            if not self.selected_tool_name:
                raise ValueError(
                    "PlannerDecision with action=CALL_TOOL must have non-empty selected_tool_name"
                )
        if self.action == PlannerActionType.ANSWER:
            if not self.draft_answer:
                raise ValueError(
                    "PlannerDecision with action=ANSWER must have non-empty draft_answer"
                )
        # CLARIFY_INTENT requires no additional fields - uses rationale only
        return self


# =============================================================================
# Final Answer Schema
# =============================================================================

class FinalAnswer(BaseModel):
    """The final answer produced by the agent."""
    answer: str = Field(..., min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_summary: str = Field(default="")
    reasoning_trace: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.now)
