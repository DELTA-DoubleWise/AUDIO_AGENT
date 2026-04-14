"""Tests for Qwen2-Audio frontend adapter."""

import json
import os
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

from audio_agent.core.errors import FrontendError
from audio_agent.core.schemas import FrontendOutput
from audio_agent.frontend.model_frontend import FrontendInputFormat, UnifiedFrontendInput
from audio_agent.frontend.qwen2_audio_frontend import Qwen2AudioFrontend


class MockQwen2AudioFrontend(Qwen2AudioFrontend):
    """Lightweight test double that avoids loading large model weights."""

    def initialize_model(self):
        return {"model": object(), "processor": object()}

    def call_model(self, model_input: UnifiedFrontendInput) -> str:
        return f"Mock caption for: {model_input.question}"


class TestQwen2AudioFrontendShape:
    """Unit tests for frontend wiring and input format behavior."""

    def test_uses_local_multimodal_input_format(self):
        frontend = MockQwen2AudioFrontend()
        assert frontend.input_format == FrontendInputFormat.LOCAL_MULTIMODAL

    def test_build_model_input_uses_audio_url_for_remote_input(self):
        frontend = MockQwen2AudioFrontend()
        model_input = frontend.build_model_input(
            question="What is in the audio?",
            audio_paths=["https://example.com/test.wav"],
        )

        assert model_input.messages[1]["role"] == "user"
        assert model_input.messages[1]["content"][0]["type"] == "audio"
        assert model_input.messages[1]["content"][0]["audio_url"] == "https://example.com/test.wav"
        assert model_input.messages[1]["content"][1]["type"] == "text"

    def test_build_model_input_uses_audio_for_local_path(self):
        frontend = MockQwen2AudioFrontend()
        model_input = frontend.build_model_input(
            question="What is in the audio?",
            audio_paths=["/tmp/example.wav"],
        )

        assert model_input.messages[1]["role"] == "user"
        assert model_input.messages[1]["content"][0]["type"] == "audio"
        assert model_input.messages[1]["content"][0]["audio"] == "/tmp/example.wav"
        assert model_input.messages[1]["content"][1]["type"] == "text"

    def test_run_returns_frontend_output(self):
        frontend = MockQwen2AudioFrontend()
        output = frontend.run("Question", ["/tmp/example.wav"])
        assert isinstance(output, FrontendOutput)
        assert output.question_guided_caption.startswith("Mock caption")

    def test_load_audio_remote_uses_urlopen_and_librosa(self, monkeypatch):
        frontend = MockQwen2AudioFrontend()
        calls: list[object] = []

        class DummyResponse:
            def read(self):
                return b"audio-bytes"

        def fake_urlopen(url: str):
            assert url == "https://example.com/test.wav"
            return DummyResponse()

        def fake_load(source, sr: int):
            calls.append(source)
            assert sr == 16000
            assert isinstance(source, BytesIO)
            return ["wave"], sr

        monkeypatch.setattr(
            "audio_agent.frontend.qwen2_audio_frontend.urlopen",
            fake_urlopen,
        )
        monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=fake_load))

        waveform = frontend._load_audio("https://example.com/test.wav", sampling_rate=16000)
        assert waveform == ["wave"]
        assert len(calls) == 1

    def test_load_audio_local_uses_librosa_path(self, monkeypatch):
        frontend = MockQwen2AudioFrontend()

        def fake_load(source, sr: int):
            assert source == "/tmp/example.wav"
            assert sr == 16000
            return ["wave"], sr

        monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=fake_load))

        waveform = frontend._load_audio("/tmp/example.wav", sampling_rate=16000)
        assert waveform == ["wave"]

    def test_load_audio_failure_raises_frontend_error(self, monkeypatch):
        frontend = MockQwen2AudioFrontend()

        def fake_load(source, sr: int):
            raise RuntimeError("decode failed")

        monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=fake_load))

        with pytest.raises(FrontendError, match="Failed to load audio input"):
            frontend._load_audio("/tmp/example.wav", sampling_rate=16000)

    def test_call_model_raises_on_empty_decoded_response(self):
        class EmptyDecodedFrontend(Qwen2AudioFrontend):
            def initialize_model(self):
                class DummyInputs(dict):
                    input_ids = SimpleNamespace(size=lambda dim: 4)

                    def to(self, device):
                        return self

                    def __init__(self):
                        super().__init__(input_ids=self.input_ids)

                class DummyProcessor:
                    feature_extractor = SimpleNamespace(sampling_rate=16000)

                    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
                        return "prompt"

                    def __call__(self, text, audio, return_tensors, padding):
                        return DummyInputs()

                    def batch_decode(self, generate_ids, skip_special_tokens, clean_up_tokenization_spaces):
                        return ["   "]

                class DummyGeneratedIds:
                    def __getitem__(self, item):
                        return self

                class DummyModel:
                    device = "cpu"

                    def generate(self, **kwargs):
                        return DummyGeneratedIds()

                return {"model": DummyModel(), "processor": DummyProcessor()}

            def _load_audio(self, audio_path_or_uri: str, sampling_rate: int):
                return ["wave"]

        frontend = EmptyDecodedFrontend()
        with pytest.raises(FrontendError, match="Qwen2-Audio returned empty response text"):
            frontend.call_model(frontend.build_model_input("Question", ["/tmp/example.wav"]))


def test_qwen2_audio_cluster_smoke():
    """
    Optional integration test for cluster use.

    Enable by setting:
    - RUN_QWEN2_AUDIO_TEST=1
    - QWEN2_AUDIO_TEST_AUDIO=<path_or_uri>
    - optional QWEN2_AUDIO_MODEL_PATH
    """
    if os.getenv("RUN_QWEN2_AUDIO_TEST") != "1":
        pytest.skip("Set RUN_QWEN2_AUDIO_TEST=1 to run Qwen2-Audio integration test")

    audio_path = os.getenv("QWEN2_AUDIO_TEST_AUDIO")
    if not audio_path:
        pytest.skip("Set QWEN2_AUDIO_TEST_AUDIO for integration test input")

    frontend = Qwen2AudioFrontend(
        model_path=os.getenv("QWEN2_AUDIO_MODEL_PATH", "Qwen/Qwen2-Audio-7B-Instruct"),
    )
    output = frontend.run(
        question="Provide a concise question-guided caption for this audio.",
        audio_paths=[audio_path],
    )
    assert output.question_guided_caption.strip()
