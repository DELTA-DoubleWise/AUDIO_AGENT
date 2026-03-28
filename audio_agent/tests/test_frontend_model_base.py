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
            audio_path_or_uri="/tmp/test.wav",
        )

        assert model_input.question == "What is in this audio?"
        assert model_input.audio_path_or_uri == "/tmp/test.wav"
        assert len(model_input.messages) == 2
        assert model_input.messages[0]["role"] == "system"
        assert model_input.messages[1]["role"] == "user"
        assert model_input.metadata["input_format"] == FrontendInputFormat.API_MODEL.value
        assert isinstance(model_input.messages[1]["content"], str)
        assert "Audio reference:" in model_input.messages[1]["content"]

    def test_build_model_input_local_multimodal_format(self):
        class LocalFrontend(EchoModelFrontend):
            @property
            def input_format(self) -> FrontendInputFormat:
                return FrontendInputFormat.LOCAL_MULTIMODAL

        frontend = LocalFrontend()
        model_input = frontend.build_model_input("What is in this audio?", "/tmp/test.wav")

        assert model_input.metadata["input_format"] == FrontendInputFormat.LOCAL_MULTIMODAL.value
        assert model_input.messages[1]["role"] == "user"
        assert isinstance(model_input.messages[1]["content"], list)
        assert model_input.messages[1]["content"][0]["type"] == "text"
        assert model_input.messages[1]["content"][1]["type"] == "audio"

    def test_unsupported_input_format_raises(self):
        class BadFormatFrontend(EchoModelFrontend):
            @property
            def input_format(self):
                return "unknown_format"

        frontend = BadFormatFrontend()
        with pytest.raises(FrontendError, match="Unsupported frontend input format"):
            frontend.build_model_input("Question", "/tmp/audio.wav")

    def test_malformed_builder_output_raises(self):
        class BadBuilderFrontend(EchoModelFrontend):
            def build_api_model_input(self, question: str, audio_path_or_uri: str):
                return UnifiedFrontendInput(
                    system_prompt=load_prompt("frontend_system"),
                    question=question,
                    audio_path_or_uri=audio_path_or_uri,
                    user_payload={},
                    messages=[],
                    metadata={},
                )

        frontend = BadBuilderFrontend()
        with pytest.raises(FrontendError, match="messages cannot be empty"):
            frontend.build_model_input("Question", "/tmp/audio.wav")

    def test_run_returns_frontend_output(self):
        frontend = EchoModelFrontend()
        output = frontend.run("Question", "/tmp/audio.wav")

        assert isinstance(output, FrontendOutput)
        assert output.question_guided_caption.startswith("Echo:")

    def test_normalize_model_output_rejects_empty_string(self):
        class EmptyFrontend(EchoModelFrontend):
            def call_model(self, model_input: UnifiedFrontendInput):
                return "   "

        frontend = EmptyFrontend()
        with pytest.raises(FrontendError, match="empty caption"):
            frontend.run("Question", "/tmp/audio.wav")

    def test_empty_inputs_raise_frontend_error(self):
        frontend = EchoModelFrontend()
        with pytest.raises(FrontendError, match="Question must be a non-empty string"):
            frontend.run("", "/tmp/audio.wav")
        with pytest.raises(FrontendError, match="Audio path/URI must be a non-empty string"):
            frontend.run("Question", "")

    def test_plain_text_output_used_directly(self):
        class PlainTextFrontend(EchoModelFrontend):
            def call_model(self, model_input: UnifiedFrontendInput):
                return "Plain text caption from model"

        frontend = PlainTextFrontend()
        output = frontend.run("Question", "/tmp/audio.wav")
        assert output.question_guided_caption == "Plain text caption from model"

    def test_multiline_text_output_used_directly(self):
        class MultilineTextFrontend(EchoModelFrontend):
            def call_model(self, model_input: UnifiedFrontendInput):
                return """This is a multiline
caption with multiple
lines of text."""

        frontend = MultilineTextFrontend()
        output = frontend.run("Question", "/tmp/audio.wav")
        assert "multiline" in output.question_guided_caption
        assert "multiple" in output.question_guided_caption

    def test_dict_output_raises_error(self):
        class DictOutputFrontend(EchoModelFrontend):
            def call_model(self, model_input: UnifiedFrontendInput):
                return {"question_guided_caption": "should not work"}

        frontend = DictOutputFrontend()
        with pytest.raises(FrontendError, match="dict instead of plain text"):
            frontend.run("Question", "/tmp/audio.wav")


class TestDummyFrontendWithModelBase:
    """Regression tests for DummyFrontend using BaseModelFrontend."""

    def test_dummy_frontend_runs(self):
        frontend = DummyFrontend()
        output = frontend.run("Describe the audio", "/fake/path.wav")

        assert isinstance(output, FrontendOutput)
        assert "Describe the audio" in output.question_guided_caption

    def test_base_module_reexports_model_frontend_symbols(self):
        from audio_agent.frontend.base import BaseModelFrontend as ReexportedBaseModelFrontend

        assert ReexportedBaseModelFrontend is BaseModelFrontend
