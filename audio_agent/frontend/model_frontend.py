"""
Model-backed frontend base classes.

This module mirrors the structure of the model-backed planner:
- explicit unified input schema
- explicit input format dispatch
- template-method hooks for model initialization and invocation
- strict output parsing and normalization
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from audio_agent.core.errors import FrontendError
from audio_agent.core.schemas import FrontendOutput
from audio_agent.frontend.base import BaseFrontend
from audio_agent.utils.model_io import parse_json_object_text, validate_message_sequence
from audio_agent.utils.prompt_io import load_prompt


class UnifiedFrontendInput(BaseModel):
    """
    Unified model-ready input for frontend calls.

    This contract keeps prompt and multimodal request shape consistent across providers.
    Provider-specific frontend implementations should consume this structure and only
    customize model initialization plus call mechanics.
    """

    system_prompt: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    audio_path_or_uri: str = Field(..., min_length=1)
    user_payload: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FrontendInputFormat(str, Enum):
    """Supported default frontend input composition modes."""

    API_MODEL = "api_model"
    LOCAL_MULTIMODAL = "local_multimodal"


class BaseModelFrontend(BaseFrontend):
    """
    Template-method base class for model-backed frontends.

    Shared behavior lives here:
    - input validation
    - model input composition
    - output parsing and normalization

    Concrete frontends mainly implement:
    - `initialize_model`
    - `call_model`
    """

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.model_config = model_config or {}
        self.model_handle = self.initialize_model()

    @abstractmethod
    def initialize_model(self) -> Any:
        """Initialize and return provider/model handle."""
        raise NotImplementedError

    @abstractmethod
    def call_model(self, model_input: UnifiedFrontendInput) -> Any:
        """Invoke provider/model with unified input and return raw model output."""
        raise NotImplementedError

    @property
    def input_format(self) -> FrontendInputFormat:
        """
        Input format mode used by build_model_input dispatcher.

        Subclasses can override this to switch default composition mode.
        """
        return FrontendInputFormat.API_MODEL

    def build_frontend_task_instruction(self, question: str, audio_path_or_uri: str) -> str:
        """Shared instruction text reused across input builders."""
        return load_prompt("frontend_user").format(
            question=question,
            audio_path_or_uri=audio_path_or_uri,
        )

    def _build_common_user_payload(self, question: str, audio_path_or_uri: str) -> dict[str, Any]:
        """Normalized provider-agnostic payload for adapters/logging."""
        return {
            "question": question,
            "audio": {
                "kind": "path_or_uri",
                "value": audio_path_or_uri,
            },
            "task": "question_guided_audio_captioning",
            "output_format": "plain_text_caption",
        }

    def build_api_model_input(self, question: str, audio_path_or_uri: str) -> UnifiedFrontendInput:
        """
        Build API-hosted chat style input:
        - one system message
        - one user message with readable task text + audio reference
        """
        user_payload = self._build_common_user_payload(question, audio_path_or_uri)
        system_prompt = load_prompt("frontend_system")
        user_text = self.build_frontend_task_instruction(question, audio_path_or_uri)

        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_path_or_uri=audio_path_or_uri,
            user_payload=user_payload,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.API_MODEL.value,
            },
        )

    def build_local_multimodal_model_input(
        self,
        question: str,
        audio_path_or_uri: str,
    ) -> UnifiedFrontendInput:
        """
        Build local multimodal style input:
        - system prompt
        - user content list with text instruction + audio reference
        """
        user_payload = self._build_common_user_payload(question, audio_path_or_uri)
        system_prompt = load_prompt("frontend_system")
        user_text = self.build_frontend_task_instruction(question, audio_path_or_uri)

        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_path_or_uri=audio_path_or_uri,
            user_payload=user_payload,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "audio", "audio": audio_path_or_uri},
                    ],
                },
            ],
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.LOCAL_MULTIMODAL.value,
            },
        )

    def _validate_built_model_input(self, model_input: UnifiedFrontendInput) -> None:
        """Fail-fast validation for builder outputs."""
        if not model_input.system_prompt.strip():
            raise FrontendError("Malformed model input: empty system_prompt")
        if not model_input.question.strip():
            raise FrontendError("Malformed model input: empty question")
        if not model_input.audio_path_or_uri.strip():
            raise FrontendError("Malformed model input: empty audio_path_or_uri")
        validate_message_sequence(
            model_input.messages,
            error_cls=FrontendError,
            context="Malformed model input",
        )

    def build_model_input(self, question: str, audio_path_or_uri: str) -> UnifiedFrontendInput:
        """Build model input via explicit format-mode dispatch."""
        mode = self.input_format
        if isinstance(mode, str):
            try:
                mode = FrontendInputFormat(mode)
            except ValueError as e:
                raise FrontendError(
                    "Unsupported frontend input format",
                    details={"input_format": mode},
                ) from e
        elif not isinstance(mode, FrontendInputFormat):
            raise FrontendError(
                "Unsupported frontend input format type",
                details={"input_format_type": type(mode).__name__},
            )

        if mode == FrontendInputFormat.API_MODEL:
            model_input = self.build_api_model_input(question, audio_path_or_uri)
        elif mode == FrontendInputFormat.LOCAL_MULTIMODAL:
            model_input = self.build_local_multimodal_model_input(question, audio_path_or_uri)
        else:
            raise FrontendError(
                "Unsupported frontend input format",
                details={"input_format": mode.value},
            )

        if not isinstance(model_input, UnifiedFrontendInput):
            raise FrontendError(
                "Malformed model input: builder must return UnifiedFrontendInput",
                details={"returned_type": type(model_input).__name__},
            )

        self._validate_built_model_input(model_input)
        return model_input

    def normalize_model_output(
        self,
        raw_output: Any,
        model_input: UnifiedFrontendInput,
    ) -> FrontendOutput:
        """
        Normalize model output into FrontendOutput.

        Supports:
        - FrontendOutput (returned directly)
        - Plain text string (treated as caption directly)
        - Dict (for backward compatibility with JSON outputs)

        Fail-fast requirement:
        - Empty outputs are rejected
        """
        if isinstance(raw_output, FrontendOutput):
            return raw_output

        # Treat string output as plain text caption (new default behavior)
        if isinstance(raw_output, str):
            caption = raw_output.strip()
            if not caption:
                raise FrontendError(
                    "Frontend returned empty caption",
                    details={"frontend": self.name},
                )
            return FrontendOutput(question_guided_caption=caption)

        # Dict output is not supported - model should return plain text
        if isinstance(raw_output, dict):
            raise FrontendError(
                "Frontend returned dict instead of plain text. "
                "The model should return plain text caption only.",
                details={"output_keys": list(raw_output.keys())},
            )

        raise FrontendError(
            "Malformed frontend output: expected str or FrontendOutput",
            details={"output_type": type(raw_output).__name__, "question": model_input.question},
        )

    def run(self, question: str, audio_path_or_uri: str) -> FrontendOutput:
        """
        Standardized frontend execution path:
        validate -> build unified input -> call model -> normalize output
        """
        self.validate_inputs(question, audio_path_or_uri)
        model_input = self.build_model_input(question.strip(), audio_path_or_uri.strip())
        try:
            raw_output = self.call_model(model_input)
        except FrontendError:
            raise
        except Exception as e:
            raise FrontendError(
                f"Model call failed: {type(e).__name__}: {e}",
                details={"frontend": self.name},
            ) from e
        return self.normalize_model_output(raw_output, model_input)
