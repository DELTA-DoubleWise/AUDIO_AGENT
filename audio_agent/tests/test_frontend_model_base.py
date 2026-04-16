"""Tests for the unified model frontend base class."""

import json

import pytest

from audio_agent.core.errors import FrontendError
from audio_agent.core.schemas import FrontendOutput
from audio_agent.frontend.model_frontend import (
    BaseModelFrontend,
    FrontendInputFormat,
    UnifiedFrontendInput,
)
from audio_agent.frontend.dummy_frontend import DummyFrontend
from audio_agent.utils.prompt_io import load_prompt


class EchoModelFrontend(BaseModelFrontend):
    """Minimal test frontend for BaseModelFrontend behavior."""

    @property
    def name(self) -> str:
        return "echo_frontend"

    def initialize_model(self) -> dict:
        return {"ready": True}

    def call_model(self, model_input: UnifiedFrontendInput):
        return f"Echo: {model_input.question}"


class TestBaseModelFrontend:
    """Tests for unified input building and output normalization."""

    def test_build_model_input_has_unified_structure(self):
        frontend = EchoModelFrontend()

        model_input = frontend.build_model_input(
            question="What is in this audio?",
            audio_paths=["/tmp/test.wav"],
        )

        assert model_input.question == "What is in this audio?"
        assert model_input.audio_paths == ["/tmp/test.wav"]
        assert len(model_input.messages) == 2
        assert model_input.messages[0]["role"] == "system"
        assert model_input.messages[1]["role"] == "user"
        assert model_input.metadata["input_format"] == FrontendInputFormat.API_MODEL.value
        assert isinstance(model_input.messages[1]["content"], str)
        assert "Audio files:" in model_input.messages[1]["content"]

    def test_build_model_input_local_multimodal_format(self):
        class LocalFrontend(EchoModelFrontend):
            @property
            def input_format(self) -> FrontendInputFormat:
                return FrontendInputFormat.LOCAL_MULTIMODAL

        frontend = LocalFrontend()
        model_input = frontend.build_model_input("What is in this audio?", ["/tmp/test.wav"])

        assert model_input.metadata["input_format"] == FrontendInputFormat.LOCAL_MULTIMODAL.value
        assert model_input.messages[1]["role"] == "user"
        assert isinstance(model_input.messages[1]["content"], list)
        # Content order: text instruction first, then audio(s)
        assert model_input.messages[1]["content"][0]["type"] == "text"
        assert model_input.messages[1]["content"][1]["type"] == "audio"
        
    def test_build_model_input_multiple_audios(self):
        """Test that multiple audio paths are properly included."""
        class LocalFrontend(EchoModelFrontend):
            @property
            def input_format(self) -> FrontendInputFormat:
                return FrontendInputFormat.LOCAL_MULTIMODAL

        frontend = LocalFrontend()
        model_input = frontend.build_model_input(
            "Compare these two audios", 
            ["/tmp/audio1.wav", "/tmp/audio2.wav"]
        )

        assert model_input.audio_paths == ["/tmp/audio1.wav", "/tmp/audio2.wav"]
        assert model_input.metadata["audio_count"] == 2
        # Content: text, audio1, audio2
        assert len(model_input.messages[1]["content"]) == 3
        assert model_input.messages[1]["content"][0]["type"] == "text"
        assert model_input.messages[1]["content"][1]["type"] == "audio"
        assert model_input.messages[1]["content"][2]["type"] == "audio"

    def test_unsupported_input_format_raises(self):
        class BadFormatFrontend(EchoModelFrontend):
            @property
            def input_format(self):
                return "unknown_format"

        frontend = BadFormatFrontend()
        with pytest.raises(FrontendError, match="Unsupported frontend input format"):
            frontend.build_model_input("Question", ["/tmp/audio.wav"])

    def test_malformed_builder_output_raises(self):
        class BadBuilderFrontend(EchoModelFrontend):
            def build_api_model_input(self, question: str, audio_paths: list[str], question_oriented_prompt: str | None = None):
                return UnifiedFrontendInput(
                    system_prompt=load_prompt("frontend_system"),
                    question=question,
                    audio_paths=audio_paths,
                    user_payload={},
                    messages=[],
                    metadata={},
                )

        frontend = BadBuilderFrontend()
        with pytest.raises(FrontendError, match="messages cannot be empty"):
            frontend.build_model_input("Question", ["/tmp/audio.wav"])

    def test_run_returns_frontend_output(self):
        frontend = EchoModelFrontend()
        output = frontend.run("Question", ["/tmp/audio.wav"])

        assert isinstance(output, FrontendOutput)
        assert output.question_guided_caption.startswith("Echo:")

    def test_normalize_model_output_rejects_empty_string(self):
        class EmptyFrontend(EchoModelFrontend):
            def call_model(self, model_input: UnifiedFrontendInput):
                return "   "

        frontend = EmptyFrontend()
        with pytest.raises(FrontendError, match="empty caption"):
            frontend.run("Question", ["/tmp/audio.wav"])

    def test_empty_inputs_raise_frontend_error(self):
        frontend = EchoModelFrontend()
        with pytest.raises(FrontendError, match="Question must be a non-empty string"):
            frontend.run("", ["/tmp/audio.wav"])
        with pytest.raises(FrontendError, match="Audio paths must contain at least one path"):
            frontend.run("Question", [])

    def test_plain_text_output_used_directly(self):
        class PlainTextFrontend(EchoModelFrontend):
            def call_model(self, model_input: UnifiedFrontendInput):
                return "Plain text caption from model"

        frontend = PlainTextFrontend()
        output = frontend.run("Question", ["/tmp/audio.wav"])
        assert output.question_guided_caption == "Plain text caption from model"

    def test_multiline_text_output_used_directly(self):
        class MultilineTextFrontend(EchoModelFrontend):
            def call_model(self, model_input: UnifiedFrontendInput):
                return """This is a multiline
caption with multiple
lines of text."""

        frontend = MultilineTextFrontend()
        output = frontend.run("Question", ["/tmp/audio.wav"])
        assert "multiline" in output.question_guided_caption
        assert "multiple" in output.question_guided_caption

    def test_dict_output_raises_error(self):
        class DictOutputFrontend(EchoModelFrontend):
            def call_model(self, model_input: UnifiedFrontendInput):
                return {"question_guided_caption": "should not work"}

        frontend = DictOutputFrontend()
        with pytest.raises(FrontendError, match="dict instead of plain text"):
            frontend.run("Question", ["/tmp/audio.wav"])


class TestDummyFrontendWithModelBase:
    """Regression tests for DummyFrontend using BaseModelFrontend."""

    def test_dummy_frontend_runs(self):
        frontend = DummyFrontend()
        output = frontend.run("Describe the audio", ["/fake/path.wav"])

        assert isinstance(output, FrontendOutput)
        assert "Describe the audio" in output.question_guided_caption

    def test_base_module_reexports_model_frontend_symbols(self):
        from audio_agent.frontend.base import BaseModelFrontend as ReexportedBaseModelFrontend

        assert ReexportedBaseModelFrontend is BaseModelFrontend


class TestBaseModelFrontendRetries:
    """Tests for retry behavior on frontend model output parsing errors."""

    def test_retry_recovers_after_transient_failure(self):
        """A frontend that fails once then succeeds should return the correct output."""
        class FlakyFrontend(EchoModelFrontend):
            def __init__(self, fail_count: int = 1):
                self._fail_count = fail_count
                self._call_count = 0
                super().__init__()

            def call_model(self, model_input: UnifiedFrontendInput):
                self._call_count += 1
                if self._call_count <= self._fail_count:
                    return "   "  # Empty caption triggers validation error
                return super().call_model(model_input)

        frontend = FlakyFrontend(fail_count=1)
        result = frontend.run("Question", ["/tmp/audio.wav"])
        assert isinstance(result, FrontendOutput)
        assert result.question_guided_caption.startswith("Echo:")
        assert frontend._call_count == 2  # 1 failure + 1 success

    def test_retry_exhausts_and_raises(self):
        """A frontend that always fails should raise after max_retries + 1 attempts."""
        class AlwaysBadFrontend(EchoModelFrontend):
            def __init__(self):
                self._call_count = 0
                super().__init__(max_retries=2)

            def call_model(self, model_input: UnifiedFrontendInput):
                self._call_count += 1
                return "   "  # Always empty

        frontend = AlwaysBadFrontend()
        with pytest.raises(FrontendError, match="failed after 3 attempts"):
            frontend.run("Question", ["/tmp/audio.wav"])
        assert frontend._call_count == 3  # initial + 2 retries

    def test_zero_retries_raises_immediately(self):
        """With max_retries=0, the first failure should raise immediately."""
        class AlwaysBadFrontend(EchoModelFrontend):
            def __init__(self):
                self._call_count = 0
                super().__init__(max_retries=0)

            def call_model(self, model_input: UnifiedFrontendInput):
                self._call_count += 1
                return "   "  # Always empty

        frontend = AlwaysBadFrontend()
        with pytest.raises(FrontendError, match="failed after 1 attempt"):
            frontend.run("Question", ["/tmp/audio.wav"])
        assert frontend._call_count == 1
