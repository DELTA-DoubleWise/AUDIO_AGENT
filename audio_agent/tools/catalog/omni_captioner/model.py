"""
Omni Captioner Model Wrapper.

API client for Qwen3-Omni multi-modal model via DashScope.
Supports audio captioning and text/audio generation.
"""

from __future__ import annotations

import os
import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CaptionResult:
    """Result of caption generation."""
    text: str
    audio_path: str | None = None
    audio_data: bytes | None = None


@dataclass
class VerificationResult:
    """Result of audio quality verification."""
    verification_passed: bool
    issues_found: list[str]
    quality_assessment: str  # "Good", "Acceptable", "Poor"
    recommendations: list[str]
    spectrogram_path: str
    analysis: str


class OmniCaptionerModel:
    """Wrapper for Qwen3-Omni API for audio captioning."""
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen3.5-omni-plus",
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
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
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
        # Note: data:;base64, prefix is required for base64 audio data
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:;base64,{audio_base64}",
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
        
        # qwen3.5-omni-plus requires modalities parameter
        if generate_audio:
            request_params["modalities"] = ["text", "audio"]
        else:
            request_params["modalities"] = ["text"]  # Text-only output
        
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

    def _generate_spectrogram(
        self,
        audio_path: str | Path,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Generate magnitude spectrogram using librosa and matplotlib.
        
        Args:
            audio_path: Path to the audio file
            output_path: Optional path to save spectrogram. If None, creates temp file.
            
        Returns:
            Path to the generated spectrogram image
        """
        try:
            import librosa
            import librosa.display
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            raise RuntimeError(f"Required packages not installed: {e}")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create temp file if output path not provided
        if output_path is None:
            output_fd, output_path = tempfile.mkstemp(suffix=".png")
            os.close(output_fd)
        else:
            output_path = Path(output_path)
        
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None)
        
        # Generate spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Plot
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {audio_path.name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    def verify_audio_quality(
        self,
        audio_path: str | Path,
        verification_prompt: str,
        reference_audio_path: str | Path | None = None,
    ) -> VerificationResult:
        """
        Verify audio quality by generating spectrogram and analyzing with VLM.
        
        Args:
            audio_path: Path to the processed/enhanced audio file to verify
            verification_prompt: Specific instructions on what to check
            reference_audio_path: Optional path to original audio for comparison
            
        Returns:
            VerificationResult with pass/fail status and analysis
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Generate spectrogram for the audio to verify
        spectrogram_path = self._generate_spectrogram(audio_path)
        
        try:
            # Build the verification prompt
            base_prompt = """You are an expert audio quality analyst. Examine the provided spectrogram(s) and assess audio quality.

Look for these common issues in the spectrogram:
- Horizontal bands indicating constant noise/hum
- Vertical spikes suggesting clicks, pops, or transient artifacts
- Irregular patterns indicating distortion or processing artifacts
- Missing frequency content suggesting over-filtering
- Unnatural gaps indicating dropped audio or severe denoising
- Excessive high-frequency noise or aliasing artifacts

{verification_prompt}

Respond in this exact format:
VERIFICATION_PASSED: [true/false]
QUALITY_ASSESSMENT: [Good/Acceptable/Poor]
ISSUES_FOUND:
- [issue 1]
- [issue 2]
...
RECOMMENDATIONS:
- [recommendation 1]
- [recommendation 2]
...
ANALYSIS: [Your detailed analysis of what you see in the spectrogram]
"""
            
            full_prompt = base_prompt.format(verification_prompt=verification_prompt)
            
            # Build message content with image(s)
            content = [{"type": "text", "text": full_prompt}]
            
            # Add spectrogram image
            with open(spectrogram_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
            
            # If reference audio provided, generate and add its spectrogram
            ref_spectrogram_path = None
            if reference_audio_path:
                ref_path = Path(reference_audio_path)
                if ref_path.exists():
                    ref_spectrogram_path = self._generate_spectrogram(ref_path)
                    content.append({
                        "type": "text",
                        "text": "\nReference (original) audio spectrogram for comparison:"
                    })
                    with open(ref_spectrogram_path, "rb") as f:
                        ref_img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{ref_img_base64}"}
                    })
            
            # Call VLM API
            client = self._get_client()
            messages = [{"role": "user", "content": content}]
            
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )
            
            # Collect response
            text_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_response += chunk.choices[0].delta.content
            
            # Parse response
            return self._parse_verification_response(
                text_response, spectrogram_path, ref_spectrogram_path
            )
            
        except Exception as e:
            # Clean up spectrogram files on error
            try:
                Path(spectrogram_path).unlink(missing_ok=True)
                if ref_spectrogram_path:
                    Path(ref_spectrogram_path).unlink(missing_ok=True)
            except:
                pass
            raise RuntimeError(f"Verification failed: {e}") from e

    def _parse_verification_response(
        self,
        response: str,
        spectrogram_path: str,
        ref_spectrogram_path: str | None,
    ) -> VerificationResult:
        """
        Parse VLM response into VerificationResult.
        
        Args:
            response: Raw text response from VLM
            spectrogram_path: Path to the generated spectrogram
            ref_spectrogram_path: Path to reference spectrogram (if any)
            
        Returns:
            Parsed VerificationResult
        """
        import re
        
        # Default values
        verification_passed = False
        quality_assessment = "Unknown"
        issues_found = []
        recommendations = []
        analysis = response  # Default to full response
        
        # Parse VERIFICATION_PASSED
        passed_match = re.search(r'VERIFICATION_PASSED:\s*(true|false)', response, re.IGNORECASE)
        if passed_match:
            verification_passed = passed_match.group(1).lower() == 'true'
        
        # Parse QUALITY_ASSESSMENT
        quality_match = re.search(r'QUALITY_ASSESSMENT:\s*(\w+)', response, re.IGNORECASE)
        if quality_match:
            quality_assessment = quality_match.group(1).capitalize()
        
        # Parse ISSUES_FOUND
        issues_section = re.search(r'ISSUES_FOUND:(.*?)(?=RECOMMENDATIONS:|ANALYSIS:|$)', 
                                   response, re.DOTALL | re.IGNORECASE)
        if issues_section:
            issues_text = issues_section.group(1)
            issues_found = [line.strip('- ').strip() 
                          for line in issues_text.split('\n') 
                          if line.strip().startswith('-')]
        
        # Parse RECOMMENDATIONS
        recs_section = re.search(r'RECOMMENDATIONS:(.*?)(?=ANALYSIS:|$)', 
                                 response, re.DOTALL | re.IGNORECASE)
        if recs_section:
            recs_text = recs_section.group(1)
            recommendations = [line.strip('- ').strip() 
                             for line in recs_text.split('\n') 
                             if line.strip().startswith('-')]
        
        # Parse ANALYSIS
        analysis_match = re.search(r'ANALYSIS:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        
        # Clean up reference spectrogram if it exists (keep main one for evidence)
        if ref_spectrogram_path:
            try:
                Path(ref_spectrogram_path).unlink(missing_ok=True)
            except:
                pass
        
        return VerificationResult(
            verification_passed=verification_passed,
            issues_found=issues_found,
            quality_assessment=quality_assessment,
            recommendations=recommendations,
            spectrogram_path=spectrogram_path,
            analysis=analysis,
        )
