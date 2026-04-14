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
        output_dir: Directory for final output files (audio results)
        copy_output_to_dir: Copy final output audio to output_dir for easy access
        log_dir: Directory for run logs
        enable_run_logging: Enable run logging to markdown files
        enable_verification: Allow planner to use VERIFY action for answer verification
        max_verifications: Maximum number of verifications allowed per run
        enable_format_check: Enable mandatory format checking before final answer
        max_format_checks: Maximum number of format checks allowed per run
    """
    max_steps: int = Field(default=10, ge=1, le=100)
    debug: bool = Field(default=False)
    planner_name: str = Field(default="dummy_planner")
    frontend_name: str = Field(default="dummy_frontend")
    fail_on_tool_error: bool = Field(default=True)
    temp_dir_base: str = Field(default="./temp", description="Base directory for temp folders")
    cleanup_temp_on_exit: bool = Field(default=True, description="Clean up temp files after run()")
    output_dir: str = Field(default="./output", description="Directory for final output files")
    copy_output_to_dir: bool = Field(default=True, description="Copy output audio to output_dir")
    log_dir: str = Field(default="./logs", description="Directory for run logs")
    enable_run_logging: bool = Field(default=True, description="Enable run logging to markdown files")
    enable_verification: bool = Field(
        default=True,
        description="Allow planner to use VERIFY action for answer verification"
    )
    max_verifications: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum number of verifications allowed per run"
    )
    enable_format_check: bool = Field(
        default=True,
        description="Enable mandatory format checking before final answer"
    )
    max_format_checks: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum number of format checks allowed per run"
    )
    max_model_output_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for planner/frontend model output parsing errors"
    )
    
    model_config = {
        "frozen": False,  # Allow modification after creation
        "extra": "forbid",  # Reject unknown fields
    }


def get_default_config() -> AgentConfig:
    """Return a default configuration."""
    return AgentConfig()
