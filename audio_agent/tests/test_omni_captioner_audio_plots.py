"""Tests for omni_captioner audio plot inspection."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_agent.tools.catalog.omni_captioner.model import OmniCaptionerModel


def _write_test_wav(path: Path, sample_rate: int = 16000) -> None:
    """Write a short mono WAV with silence and impulse-like bursts."""
    samples: list[int] = []
    for i in range(sample_rate):
        t = i / sample_rate
        value = 0.0
        if 0.20 <= t <= 0.24 or 0.55 <= t <= 0.59:
            value += 0.8 * math.sin(2 * math.pi * 1200 * t)
        if 0.70 <= t <= 0.85:
            value += 0.25 * math.sin(2 * math.pi * 440 * t)
        samples.append(max(-32767, min(32767, int(value * 32767))))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))


class TestOmniCaptionerAudioPlots:
    """Tests for plot construction and bounded VLM prompt behavior."""

    def test_generate_combined_audio_plots(self, tmp_path):
        """Combined plot generation should produce one PNG for selected plot types."""
        pytest.importorskip("librosa")
        audio_path = tmp_path / "test.wav"
        _write_test_wav(audio_path)
        model = OmniCaptionerModel(api_key="test")

        plot_path, plot_types, start, end = model._generate_combined_audio_plots(
            audio_path,
            plot_types=["waveform", "rms_energy", "onset_envelope"],
        )

        assert Path(plot_path).exists()
        assert Path(plot_path).suffix == ".png"
        assert plot_types == ["waveform", "rms_energy", "onset_envelope"]
        assert start == 0.0
        assert end == pytest.approx(1.0)

    def test_generate_combined_audio_plots_validates_inputs(self, tmp_path):
        """Plot generation should fail fast on invalid input."""
        audio_path = tmp_path / "test.wav"
        _write_test_wav(audio_path)
        model = OmniCaptionerModel(api_key="test")

        with pytest.raises(FileNotFoundError):
            model._generate_combined_audio_plots(tmp_path / "missing.wav")

        with pytest.raises(ValueError, match="Unsupported plot_types"):
            model._generate_combined_audio_plots(audio_path, plot_types=["unknown"])

        pytest.importorskip("librosa")
        with pytest.raises(ValueError, match="Invalid time_range"):
            model._generate_combined_audio_plots(
                audio_path,
                time_range={"start": 0.8, "end": 0.2},
            )

    def test_fixed_prompt_contains_boundaries(self):
        """The fixed prompt should constrain the VLM to visual-acoustic evidence."""
        model = OmniCaptionerModel(api_key="test")

        prompt = model._build_audio_plot_inspection_prompt(
            question="Did the speaker use an Indian accent?",
            analysis_focus="Check if the plots contain evidence relevant to accent.",
            plot_types=["waveform", "mel_spectrogram"],
            start_time=0.0,
            end_time=2.0,
        )

        assert "not listening to the audio" in prompt
        assert "Do not directly answer" in prompt
        assert "accent" in prompt
        assert "cannot be determined" in prompt
        assert "Check if the plots contain evidence relevant to accent." in prompt

    def test_call_vlm_with_single_image_sends_one_image(self, tmp_path):
        """The VLM call should send exactly one image input."""
        image_path = tmp_path / "plot.png"
        image_path.write_bytes(b"fake-png-bytes")
        captured: dict = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                delta = SimpleNamespace(content='{"visual_acoustic_clues": []}')
                choice = SimpleNamespace(delta=delta)
                return [SimpleNamespace(choices=[choice])]

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        model = OmniCaptionerModel(api_key="test")
        model._client = fake_client

        response = model._call_vlm_with_single_image("Inspect this plot.", image_path)

        assert response == '{"visual_acoustic_clues": []}'
        content = captured["messages"][0]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]
        assert len(image_items) == 1
        assert image_items[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_inspect_audio_plots_parses_structured_response(self, tmp_path, monkeypatch):
        """inspect_audio_plots should return structured evidence and keep plot path."""
        pytest.importorskip("librosa")
        audio_path = tmp_path / "test.wav"
        _write_test_wav(audio_path)
        model = OmniCaptionerModel(api_key="test")

        monkeypatch.setattr(
            model,
            "_call_vlm_with_single_image",
            lambda prompt, image_path: (
                '{"visual_acoustic_clues":["two bursts"],'
                '"timeline":[{"start":0.2,"end":0.24,"observation":"burst"}],'
                '"focus_relevant_evidence":"bursts are visible",'
                '"uncertain_or_not_determinable":["semantic source"],'
                '"recommended_next_tools":["frontend_followup"],'
                '"reliability":"medium"}'
            ),
        )

        result = model.inspect_audio_plots(
            audio_path=audio_path,
            question="How many knocks are there?",
            analysis_focus="Look for repeated short transient bursts.",
            plot_types=["waveform", "onset_envelope"],
        )

        assert result.parsing_warning is None
        assert result.structured_result["visual_acoustic_clues"] == ["two bursts"]
        assert result.structured_result["plot_path"] == result.plot_path
        assert Path(result.plot_path).exists()

    def test_inspect_audio_plots_reports_parse_warning(self, tmp_path, monkeypatch):
        """Malformed VLM output should be surfaced as raw low-reliability evidence."""
        pytest.importorskip("librosa")
        audio_path = tmp_path / "test.wav"
        _write_test_wav(audio_path)
        model = OmniCaptionerModel(api_key="test")
        monkeypatch.setattr(model, "_call_vlm_with_single_image", lambda prompt, image_path: "not json")

        result = model.inspect_audio_plots(
            audio_path=audio_path,
            question="Is there a sudden cutoff?",
            analysis_focus="Check for abrupt end-of-audio energy drop.",
        )

        assert result.parsing_warning is not None
        assert result.structured_result["reliability"] == "low"
        assert result.structured_result["raw_response"] == "not json"
