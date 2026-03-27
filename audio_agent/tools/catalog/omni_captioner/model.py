"""
Omni Captioner Model Wrapper.

API client for Qwen3-Omni multi-modal model via DashScope.
Supports audio captioning and text/audio generation.
"""

from __future__ import annotations

import os
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CaptionResult:
    """Result of caption generation."""
    text: str
    audio_path: str | None = None
    audio_data: bytes | None = None


class OmniCaptionerModel:
    """Wrapper for Qwen3-Omni API for audio captioning."""
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen3-omni-flash",
        voice: str = "Cherry",
        audio_format: str = "wav",
        sample_rate: int = 24000,
    ):
        """
        Initialize Omni Captioner model.
        
        Args:
            api_key: DashScope API key (or set DASHSCOPE_API_KEY env var)
            base_url: API base URL
            model: Model ID
            voice: Voice for audio generation
            audio_format: Audio format (wav, mp3, etc.)
            sample_rate: Audio sample rate
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        
        self._client = None
    
    def _get_client(self):
        """Lazy initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "openai not installed. Run: pip install openai"
                )
            
            if not self.api_key:
                raise RuntimeError(
                    "DASHSCOPE_API_KEY not set. Please provide API key."
                )
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client
    
    def caption_audio(
        self,
        audio_path: str | Path,
        prompt: str = "Describe this audio in detail.",
        generate_audio: bool = False,
        output_audio_path: str | Path | None = None,
    ) -> CaptionResult:
        """
        Generate caption for an audio file.
        
        Args:
            audio_path: Path to the audio file to caption
            prompt: Prompt for the captioning task
            generate_audio: Whether to generate audio response
            output_audio_path: Path to save the generated audio file
            
        Returns:
            CaptionResult with text and optional audio
        """
        import numpy as np
        
        client = self._get_client()
        
        # Read and encode audio
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Determine audio format from file extension
        audio_format = audio_path.suffix.lstrip(".").lower()
        if audio_format not in ["wav", "mp3", "ogg", "m4a", "flac"]:
            audio_format = "wav"  # Default fallback
        
        # Build request with audio input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": audio_format,
                        }
                    }
                ]
            }
        ]
        
        request_params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        
        if generate_audio:
            request_params["modalities"] = ["text", "audio"]
            request_params["audio"] = {
                "voice": self.voice,
                "format": self.audio_format
            }
        
        # Call API
        completion = client.chat.completions.create(**request_params)
        
        # Process response
        text_response = ""
        audio_response_base64 = ""
        
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                text_response += chunk.choices[0].delta.content
            
            if (chunk.choices and 
                hasattr(chunk.choices[0].delta, "audio") and 
                chunk.choices[0].delta.audio):
                audio_response_base64 += chunk.choices[0].delta.audio.get("data", "")
        
        # Save audio if generated
        audio_data = None
        saved_path = None
        
        if generate_audio and audio_response_base64 and output_audio_path:
            wav_bytes = base64.b64decode(audio_response_base64)
            audio_data = wav_bytes
            
            # Save to file
            output_path = Path(output_audio_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import soundfile as sf
            audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
            sf.write(str(output_path), audio_np, samplerate=self.sample_rate)
            saved_path = str(output_path)
        
        return CaptionResult(
            text=text_response,
            audio_path=saved_path,
            audio_data=audio_data,
        )
    
    def caption_text_only(
        self,
        audio_path: str | Path,
        prompt: str = "Describe this audio in detail.",
    ) -> str:
        """
        Generate text caption for an audio file (no audio output).
        
        Args:
            audio_path: Path to the audio file to caption
            prompt: Prompt for the captioning task
            
        Returns:
            Text caption
        """
        result = self.caption_audio(audio_path, prompt, generate_audio=False)
        return result.text
