"""Tests for Qwen3-Omni frontend adapter."""

import os

import pytest

from audio_agent.core.schemas import FrontendOutput
from audio_agent.frontend.model_frontend import FrontendInputFormat, UnifiedFrontendInput
from audio_agent.frontend.qwen3_omni_frontend import Qwen3OmniFrontend


class MockQwen3OmniFrontend(Qwen3OmniFrontend):
    """Lightweight test double that avoids loading large model weights."""

    def initialize_model(self):
        return {"model": object(), "processor": object()}

    def call_model(self, model_input: UnifiedFrontendInput):
        return f"Mock caption for: {model_input.question}"


class TestQwen3OmniFrontendShape:
    """Unit tests for frontend wiring and input format behavior."""

    def test_uses_local_multimodal_input_format(self):
        frontend = MockQwen3OmniFrontend()
        assert frontend.input_format == FrontendInputFormat.LOCAL_MULTIMODAL

    def test_use_audio_in_video_is_disabled(self):
        frontend = MockQwen3OmniFrontend()
        assert frontend.use_audio_in_video is False

    def test_build_model_input_contains_audio_content_block(self):
        frontend = MockQwen3OmniFrontend()
        model_input = frontend.build_model_input(
            question="What is in the audio?",
            audio_paths=["/tmp/example.wav"],
        )
        assert model_input.messages[1]["role"] == "user"
        assert model_input.messages[1]["content"][1]["type"] == "audio"
        assert model_input.messages[1]["content"][1]["audio"] == "/tmp/example.wav"

    def test_run_returns_frontend_output(self):
        frontend = MockQwen3OmniFrontend()
        output = frontend.run("Question", ["/tmp/example.wav"])
        assert isinstance(output, FrontendOutput)
        assert output.question_guided_caption.startswith("Mock caption")


def test_qwen3_omni_cluster_smoke():
    """
    Optional integration test for cluster use.

    Enable by setting:
    - RUN_QWEN3_OMNI_TEST=1
    - QWEN3_OMNI_TEST_AUDIO=<path_or_uri>
    - optional QWEN3_OMNI_MODEL_PATH
    """
    if os.getenv("RUN_QWEN3_OMNI_TEST") != "1":
        pytest.skip("Set RUN_QWEN3_OMNI_TEST=1 to run Qwen3-Omni integration test")

    audio_path = os.getenv("QWEN3_OMNI_TEST_AUDIO")
    if not audio_path:
        pytest.skip("Set QWEN3_OMNI_TEST_AUDIO for integration test input")

    frontend = Qwen3OmniFrontend(
        model_path=os.getenv("QWEN3_OMNI_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
    )
    output = frontend.run(
        question="Provide a concise question-guided caption for this audio.",
        audio_paths=[audio_path],
    )
    assert output.question_guided_caption.strip()
