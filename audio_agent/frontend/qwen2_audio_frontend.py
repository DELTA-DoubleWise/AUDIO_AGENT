"""
Qwen2-Audio frontend implementation.

This frontend follows the sample inference flow for Qwen2-Audio:
- load model and processor from transformers
- build Qwen2-Audio multimodal conversation messages
- load one local or remote audio input
- generate text output

Output contract remains the framework FrontendOutput schema:
- question_guided_caption

This adapter is intentionally constrained to:
- audio + text input
- text output only
"""

from __future__ import annotations

from io import BytesIO
from typing import Any
from urllib.request import urlopen

from audio_agent.core.errors import FrontendError
from audio_agent.frontend.model_frontend import (
    BaseModelFrontend,
    FrontendInputFormat,
    UnifiedFrontendInput,
)
from audio_agent.utils.model_downloader import DEFAULT_QWEN2_AUDIO_PATH
from audio_agent.utils.prompt_io import load_prompt


# Use local model path by default, fallback to HuggingFace Hub if not available
DEFAULT_QWEN2_AUDIO_MODEL_PATH = DEFAULT_QWEN2_AUDIO_PATH


class Qwen2AudioFrontend(BaseModelFrontend):
    """Qwen2-Audio frontend adapter."""

    def __init__(
        self,
        model_path: str = DEFAULT_QWEN2_AUDIO_MODEL_PATH,
        device_map: str = "auto",
        max_length: int = 1024,
        generation_kwargs: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model_path = model_path
        self.device_map = device_map
        self.max_length = max_length
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_config=model_config, max_retries=max_retries)

    @property
    def name(self) -> str:
        return "qwen2_audio_frontend"

    @property
    def input_format(self) -> FrontendInputFormat:
        return FrontendInputFormat.LOCAL_MULTIMODAL

    def initialize_model(self) -> dict[str, Any]:
        """Load Qwen2-Audio model + processor."""
        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except ImportError as e:
            raise FrontendError(
                "Missing Qwen2-Audio dependencies",
                details={
                    "required_packages": ["transformers", "torch", "librosa"],
                    "hint": "Install model dependencies in your cluster environment",
                },
            ) from e

        try:
            processor = AutoProcessor.from_pretrained(self.model_path)
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=self.device_map,
            )
        except Exception as e:
            raise FrontendError(
                f"Failed to initialize Qwen2-Audio model/processor: {type(e).__name__}: {e}",
                details={"model_path": self.model_path},
            ) from e

        return {
            "model": model,
            "processor": processor,
        }

    def build_local_multimodal_model_input(
        self,
        question: str,
        audio_paths: list[str],
        question_oriented_prompt: str | None = None,
    ) -> UnifiedFrontendInput:
        """Build Qwen2-Audio-style multimodal input with local/remote audio key selection.
        
        Note: audio_paths will always have exactly one audio when this is called,
        since the base run() method iterates through multiple audios separately.
        """
        audio_path = audio_paths[0]
        user_payload = self._build_common_user_payload(question, audio_paths)
        
        # Build content with single audio and text
        audio_content: dict[str, str] = {"type": "audio"}
        if self._is_remote_audio(audio_path):
            audio_content["audio_url"] = audio_path
        else:
            audio_content["audio"] = audio_path
        
        content = [
            audio_content,
            {"type": "text", "text": self.build_frontend_task_instruction(question, audio_paths, question_oriented_prompt)},
        ]

        system_prompt = load_prompt("frontend_system")
        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_paths=audio_paths,
            user_payload=user_payload,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.LOCAL_MULTIMODAL.value,
            },
        )

    def _is_remote_audio(self, audio_path_or_uri: str) -> bool:
        """Return True if the audio reference is an HTTP(S) URL."""
        return audio_path_or_uri.startswith("http://") or audio_path_or_uri.startswith("https://")

    def _load_audio(self, audio_path_or_uri: str, sampling_rate: int) -> Any:
        """Load one audio waveform from a local path or remote URL."""
        try:
            import librosa
        except ImportError as e:
            raise FrontendError(
                "Missing librosa dependency",
                details={"required_package": "librosa"},
            ) from e

        try:
            if self._is_remote_audio(audio_path_or_uri):
                audio_bytes = BytesIO(urlopen(audio_path_or_uri).read())
                waveform, _sr = librosa.load(audio_bytes, sr=sampling_rate)
            else:
                waveform, _sr = librosa.load(audio_path_or_uri, sr=sampling_rate)
        except Exception as e:
            raise FrontendError(
                f"Failed to load audio input: {type(e).__name__}: {e}",
                details={"audio_path_or_uri": audio_path_or_uri},
            ) from e

        return waveform

    def call_model(self, model_input: UnifiedFrontendInput) -> str:
        """Run Qwen2-Audio generation and return raw text output.
        
        Note: This processes a single audio per call. The base run() method
        handles multiple audios by making separate calls.
        """
        model = self.model_handle["model"]
        processor = self.model_handle["processor"]
        conversation = model_input.messages

        try:
            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            # Load single audio file
            audio_path = model_input.audio_paths[0]
            audio_array = self._load_audio(
                audio_path,
                sampling_rate=processor.feature_extractor.sampling_rate,
            )

            inputs = processor(
                text=text,
                audio=[audio_array],
                return_tensors="pt",
                padding=True,
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)

            generate_kwargs = {"max_length": self.max_length}
            generate_kwargs.update(self.generation_kwargs)
            generate_ids = model.generate(**inputs, **generate_kwargs)
            generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
            response_texts = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except FrontendError:
            raise
        except Exception as e:
            raise FrontendError(
                f"Qwen2-Audio inference failed: {type(e).__name__}: {e}",
                details={"model_path": self.model_path},
            ) from e

        if not response_texts:
            raise FrontendError("Qwen2-Audio returned empty response list")

        raw_response = response_texts[0].strip()
        if not raw_response:
            raise FrontendError("Qwen2-Audio returned empty response text")

        return raw_response
