"""
Dummy frontend implementation for testing and development.

This frontend returns canned responses without real audio processing.
"""

from audio_agent.frontend.model_frontend import BaseModelFrontend, UnifiedFrontendInput


class DummyFrontend(BaseModelFrontend):
    """
    Dummy frontend that returns mock evidence.
    
    Useful for:
    - Testing the framework end-to-end
    - Development without GPU/API dependencies
    - Demonstrating the expected interface
    """
    
    @property
    def name(self) -> str:
        return "dummy_frontend"

    def initialize_model(self) -> dict:
        """
        Initialize dummy model handle.

        Real providers should initialize SDK clients or local model handles here.
        """
        return {"provider": "dummy", "ready": True}

    def call_model(self, model_input: UnifiedFrontendInput) -> dict:
        """
        Return mock model output using the unified input structure.
        """
        caption = (
            f"The audio appears to include speech content relevant to the question "
            f"'{model_input.question}'. "
            "There may also be background sounds, but exact lexical details are unclear "
            "from this initial frontend pass."
        )
        return {
            "question_guided_caption": caption,
        }
