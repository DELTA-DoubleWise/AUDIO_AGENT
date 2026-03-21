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


DEFAULT_FRONTEND_SYSTEM_PROMPT = (
    "You are the front-end perception model for an audio agent. "
    "Your job is to inspect the input audio and produce a question-guided textual caption "
    "for a downstream planner. "
    "You are not the final answering agent and not the main reasoner. "
    "Do not do final reasoning or final answering. "
    "Focus on information relevant to the user question. "
    "Do not guess unsupported details. "
    "State uncertainty explicitly when details are unclear. "
    "Keep the output concise, faithful, and useful for downstream tool planning. "
    "Return a JSON object with key: `question_guided_caption` (str)."
)


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
        system_prompt: str | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.system_prompt = (system_prompt or DEFAULT_FRONTEND_SYSTEM_PROMPT).strip()
        if not self.system_prompt:
            raise FrontendError("system_prompt must be non-empty")
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

    def build_frontend_task_instruction(self, question: str) -> str:
        """Shared instruction text reused across input builders."""
        return (
            f"User question: {question}\n"
            "Inspect the audio and produce a concise question-guided caption for a downstream planner. "
            "Do not do final reasoning or final answering. "
            "State uncertainty explicitly when details are unclear.\n"
            "Return a JSON object with key: `question_guided_caption` (str)."
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
            "output_schema": {
                "question_guided_caption": "str (non-empty)",
            },
        }

    def build_api_model_input(self, question: str, audio_path_or_uri: str) -> UnifiedFrontendInput:
        """
        Build API-hosted chat style input:
        - one system message
        - one user message with readable task text + audio reference
        """
        user_payload = self._build_common_user_payload(question, audio_path_or_uri)
        user_text = (
            f"{self.build_frontend_task_instruction(question)}\n"
            f"Audio reference: {audio_path_or_uri}"
        )

        return UnifiedFrontendInput(
            system_prompt=self.system_prompt,
            question=question,
            audio_path_or_uri=audio_path_or_uri,
            user_payload=user_payload,
            messages=[
                {"role": "system", "content": self.system_prompt},
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

        return UnifiedFrontendInput(
            system_prompt=self.system_prompt,
            question=question,
            audio_path_or_uri=audio_path_or_uri,
            user_payload=user_payload,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.build_frontend_task_instruction(question)},
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

        Supported raw output forms:
        - FrontendOutput (returned directly)
        - JSON text
        - dict with exact FrontendOutput-compatible fields

        Fail-fast requirement:
        - No silent field repair/filling for malformed model output
        """
        if isinstance(raw_output, FrontendOutput):
            return raw_output

        if isinstance(raw_output, str):
            parsed = parse_json_object_text(
                raw_output,
                error_cls=FrontendError,
                subject="Model",
            )
            return self.normalize_model_output(parsed, model_input)

        if isinstance(raw_output, dict):
            required = {"question_guided_caption"}
            keys = set(raw_output.keys())
            missing = sorted(required - keys)
            if missing:
                raise FrontendError(
                    "Malformed frontend output: missing required fields",
                    details={"missing_fields": missing, "output_keys": sorted(keys)},
                )
            try:
                return FrontendOutput(**raw_output)
            except Exception as e:
                raise FrontendError(
                    "Malformed frontend output: schema validation failed",
                    details={"error": str(e), "output_keys": sorted(keys)},
                ) from e

        raise FrontendError(
            "Malformed frontend output: expected dict or FrontendOutput",
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
