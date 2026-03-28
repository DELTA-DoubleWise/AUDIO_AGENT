"""
Abstract base class for frontend modules.

The frontend is responsible for initial audio understanding,
producing question-guided evidence/captions from raw audio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from audio_agent.core.errors import FrontendError
from audio_agent.core.schemas import FrontendOutput


class BaseFrontend(ABC):
    """
    Abstract base class for audio frontends.

    A frontend takes a question and audio path, and produces initial evidence.
    Concrete implementations might use:
    - Local LALM models
    - Remote API-based audio understanding services
    - Hybrid approaches
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this frontend for logging and identification."""
        raise NotImplementedError

    @abstractmethod
    def run(self, question: str, audio_path_or_uri: str) -> FrontendOutput:
        """
        Process audio with the given question and produce initial evidence.

        Args:
            question: The user's question about the audio
            audio_path_or_uri: Path or URI to the audio file

        Returns:
            FrontendOutput with question-guided caption

        Raises:
            FrontendError: If processing fails or input is invalid
        """
        raise NotImplementedError

    def validate_inputs(self, question: str, audio_path_or_uri: str) -> None:
        """
        Validate inputs before processing. Called by subclasses.

        Raises:
            FrontendError: If inputs are invalid
        """
        if not question or not question.strip():
            raise FrontendError(
                "Question must be a non-empty string",
                details={"question": question},
            )
        if not audio_path_or_uri or not audio_path_or_uri.strip():
            raise FrontendError(
                "Audio path/URI must be a non-empty string",
                details={"audio_path_or_uri": audio_path_or_uri},
            )


# Backward-compatible re-exports for existing imports.
from audio_agent.frontend.model_frontend import (  # noqa: E402
    BaseModelFrontend,
    FrontendInputFormat,
    UnifiedFrontendInput,
)

__all__ = [
    "BaseFrontend",
    "BaseModelFrontend",
    "FrontendInputFormat",
    "UnifiedFrontendInput",
]
