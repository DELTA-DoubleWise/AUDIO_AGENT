"""
Configuration settings for the audio agent.

Uses Pydantic for validation and environment variable support.
"""

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    Configuration for the audio agent.
    
    Attributes:
        max_steps: Maximum number of steps before exhaustion
        debug: Enable debug logging
        planner_name: Name of planner to use (for future dynamic selection)
        frontend_name: Name of frontend to use (for future dynamic selection)
        fail_on_tool_error: Whether to fail the agent on tool errors
        temp_dir_base: Base directory for temporary audio file storage
        cleanup_temp_on_exit: Whether to clean up temp files after run() completes
    """
    max_steps: int = Field(default=10, ge=1, le=100)
    debug: bool = Field(default=False)
    planner_name: str = Field(default="dummy_planner")
    frontend_name: str = Field(default="dummy_frontend")
    fail_on_tool_error: bool = Field(default=True)
    temp_dir_base: str = Field(default="./temp", description="Base directory for temp folders")
    cleanup_temp_on_exit: bool = Field(default=True, description="Clean up temp files after run()")
    
    model_config = {
        "frozen": False,  # Allow modification after creation
        "extra": "forbid",  # Reject unknown fields
    }


def get_default_config() -> AgentConfig:
    """Return a default configuration."""
    return AgentConfig()
