#!/usr/bin/env python3
"""MCP Server for FFmpeg audio processing tool - 56+ comprehensive tools."""

from __future__ import annotations

import json
import sys
from typing import Any

try:
    from .model import FFmpegWrapper, InferenceError
except ImportError:
    from model import FFmpegWrapper, InferenceError


class FFmpegMCPServer:
    def __init__(self) -> None:
        self._initialized = False
        self._wrapper: FFmpegWrapper | None = None

    def _get_wrapper(self) -> FFmpegWrapper:
        if self._wrapper is None:
            self._wrapper = FFmpegWrapper()
        return self._wrapper

    def _handle_initialize(self, request: dict[str, Any]) -> dict[str, Any]:
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ffmpeg-mcp-server", "version": "2.0.0"},
            },
        }

    def _get_tools_list(self) -> list[dict[str, Any]]:
        """Return list of all 56+ tool schemas."""
        # Tool definitions are the same as before...
        return [
            # LEGACY
            {"name": "process_audio", "description": "Process audio (legacy)", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "start_time": {"type": "number"}, "duration": {"type": "number"}, "sample_rate": {"type": "integer"}, "channels": {"type": "integer"}}, "required": ["audio_path", "output_path"]}},
            # PHASE 1: Format (5)
            {"name": "convert_format", "description": "Convert audio format", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "format_ext": {"type": "string"}, "codec": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "resample_audio", "description": "Resample audio to a specific sample rate.", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string", "description": "Path to input audio file"}, "sample_rate": {"type": "integer", "description": "Target sample rate in Hz. Use 16000 for most ML models, 22050 or 44100 for high quality. REQUIRED."}, "output_path": {"type": "string", "description": "Optional output path. If not provided, saved to temp directory."}}, "required": ["audio_path", "sample_rate"]}},
            {"name": "change_bit_depth", "description": "Change bit depth", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "bit_depth": {"type": "integer", "default": 16}}, "required": ["audio_path"]}},
            {"name": "convert_channels", "description": "Convert channel layout", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "channels": {"type": "integer", "default": 1}}, "required": ["audio_path"]}},
            {"name": "extract_audio_from_video", "description": "Extract audio from video", "inputSchema": {"type": "object", "properties": {"video_path": {"type": "string"}, "output_path": {"type": "string"}, "codec": {"type": "string", "default": "copy"}}, "required": ["video_path"]}},
            # PHASE 2: Volume (7)
            {"name": "adjust_volume", "description": "Adjust volume", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "volume_db": {"type": "number"}, "volume_factor": {"type": "number"}}, "required": ["audio_path"]}},
            {"name": "loudnorm", "description": "Loudness normalization", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "target_lufs": {"type": "number", "default": -16}, "true_peak": {"type": "number", "default": -1.5}, "loudness_range": {"type": "number", "default": 11}}, "required": ["audio_path"]}},
            {"name": "dynaudnorm", "description": "Dynamic normalization", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "frame_len": {"type": "integer", "default": 500}, "filter_size": {"type": "integer", "default": 31}}, "required": ["audio_path"]}},
            {"name": "acompressor", "description": "Dynamic compression", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "threshold": {"type": "number", "default": 0.05}, "ratio": {"type": "number", "default": 4}, "attack": {"type": "integer", "default": 200}, "release": {"type": "integer", "default": 1000}}, "required": ["audio_path"]}},
            {"name": "agate", "description": "Noise gate", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "threshold": {"type": "number", "default": 0.1}, "attack": {"type": "integer", "default": 50}, "release": {"type": "integer", "default": 50}}, "required": ["audio_path"]}},
            {"name": "alimiter", "description": "Brick-wall limiter", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "limit": {"type": "number", "default": 0.95}, "attack": {"type": "integer", "default": 5}, "release": {"type": "integer", "default": 50}}, "required": ["audio_path"]}},
            {"name": "compand", "description": "Compand", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "attack": {"type": "number", "default": 0.3}, "decay": {"type": "number", "default": 0.8}}, "required": ["audio_path"]}},
            # PHASE 3: EQ (8)
            {"name": "highpass_filter", "description": "High-pass filter", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "frequency": {"type": "number", "default": 80}, "poles": {"type": "integer", "default": 2}}, "required": ["audio_path"]}},
            {"name": "lowpass_filter", "description": "Low-pass filter", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "frequency": {"type": "number", "default": 8000}, "poles": {"type": "integer", "default": 2}}, "required": ["audio_path"]}},
            {"name": "bandpass_filter", "description": "Band-pass filter", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "frequency": {"type": "number", "default": 1000}, "width": {"type": "number", "default": 200}}, "required": ["audio_path"]}},
            {"name": "bandreject_filter", "description": "Band-reject filter", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "frequency": {"type": "number", "default": 50}, "width": {"type": "number", "default": 20}}, "required": ["audio_path"]}},
            {"name": "equalizer", "description": "Parametric EQ", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "frequency": {"type": "number", "default": 1000}, "gain": {"type": "number", "default": 5}, "width": {"type": "number", "default": 2}, "width_type": {"type": "string", "default": "o"}}, "required": ["audio_path"]}},
            {"name": "anequalizer", "description": "Graphic EQ", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "bands": {"type": "array"}}, "required": ["audio_path"]}},
            {"name": "bass_boost", "description": "Bass boost", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "gain": {"type": "number", "default": 10}, "frequency": {"type": "number", "default": 100}, "width": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            {"name": "treble_boost", "description": "Treble boost", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "gain": {"type": "number", "default": 5}, "frequency": {"type": "number", "default": 3000}, "width": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            # PHASE 4: Noise (4)
            {"name": "afftdn_denoise", "description": "FFT denoising", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "noise_reduction": {"type": "number", "default": 12}, "noise_floor": {"type": "number", "default": -25}}, "required": ["audio_path"]}},
            {"name": "afwtdn_denoise", "description": "Wavelet denoising", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "sigma": {"type": "number", "default": 0.1}, "levels": {"type": "integer", "default": 10}}, "required": ["audio_path"]}},
            {"name": "adeclick", "description": "Remove clicks", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "frequency_filter_combo", "description": "HP+LP combo", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "highpass_freq": {"type": "number", "default": 80}, "lowpass_freq": {"type": "number", "default": 8000}}, "required": ["audio_path"]}},
            # PHASE 5: Time (7)
            {"name": "change_tempo", "description": "Change tempo", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "tempo_ratio": {"type": "number", "default": 1.5}}, "required": ["audio_path"]}},
            {"name": "trim_audio", "description": "Trim audio", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "start_time": {"type": "number", "default": 0}, "duration": {"type": "number"}, "end_time": {"type": "number"}}, "required": ["audio_path"]}},
            {"name": "silenceremove", "description": "Remove silence", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "noise_db": {"type": "number", "default": -50}, "min_silence_duration": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            {"name": "pad_silence", "description": "Pad silence", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "pad_start": {"type": "number", "default": 0}, "pad_end": {"type": "number", "default": 2}}, "required": ["audio_path"]}},
            {"name": "reverse_audio", "description": "Reverse audio", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "add_delay", "description": "Add echo/delay", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "delay_ms": {"type": "number", "default": 1000}, "decay": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            {"name": "pitch_shift_rubberband", "description": "Pitch shift", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "pitch_ratio": {"type": "number", "default": 1.2}, "tempo_ratio": {"type": "number", "default": 1}}, "required": ["audio_path"]}},
            # PHASE 6: Spatial (6)
            {"name": "pan_channels", "description": "Pan/remix", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "channel_layout": {"type": "string", "default": "mono"}, "pan_expr": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "split_channels", "description": "Split channels", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_dir": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "mix_audio", "description": "Mix audio files", "inputSchema": {"type": "object", "properties": {"input_paths": {"type": "array"}, "output_path": {"type": "string"}, "weights": {"type": "array"}}, "required": ["input_paths"]}},
            {"name": "stereotools", "description": "Stereo tools", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "mode": {"type": "string", "default": "ms"}}, "required": ["audio_path"]}},
            {"name": "stereowiden", "description": "Stereo widen", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "delay": {"type": "integer", "default": 20}, "feedback": {"type": "number", "default": 0.3}, "crossfeed": {"type": "number", "default": 0.3}}, "required": ["audio_path"]}},
            {"name": "crossfeed", "description": "Headphone crossfeed", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "strength": {"type": "number", "default": 0.2}, "range_hz": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            # PHASE 7: Effects (8)
            {"name": "add_echo", "description": "Add echo", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "delays": {"type": "array"}, "decays": {"type": "array"}}, "required": ["audio_path"]}},
            {"name": "chorus_effect", "description": "Chorus effect", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "flanger_effect", "description": "Flanger effect", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "delay": {"type": "number", "default": 0}, "depth": {"type": "number", "default": 2}, "regen": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            {"name": "phaser_effect", "description": "Phaser effect", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "speed": {"type": "number", "default": 0.5}, "decay": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            {"name": "tremolo_effect", "description": "Tremolo effect", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "freq": {"type": "number", "default": 5}, "depth": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            {"name": "vibrato_effect", "description": "Vibrato effect", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "freq": {"type": "number", "default": 5}, "depth": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            {"name": "deesser", "description": "De-esser", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "intensity": {"type": "number", "default": 1}, "frequency": {"type": "number", "default": 5500}}, "required": ["audio_path"]}},
            {"name": "crystalizer", "description": "Crystalizer", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}, "intensity": {"type": "number", "default": 2}}, "required": ["audio_path"]}},
            # PHASE 8: Analysis (7)
            {"name": "audio_stats", "description": "Audio stats", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "silencedetect", "description": "Detect silence", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "noise_db": {"type": "number", "default": -50}, "min_duration": {"type": "number", "default": 0.5}}, "required": ["audio_path"]}},
            {"name": "volumedetect", "description": "Detect volume", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "ebur128", "description": "EBU R128", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "replaygain", "description": "ReplayGain", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "astats", "description": "AStats", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]}},
            {"name": "spectral_stats", "description": "Spectral stats", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]}},
            # PHASE 9: Advanced (4)
            {"name": "sidechain_compress", "description": "Sidechain compression", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "sidechain_path": {"type": "string"}, "output_path": {"type": "string"}, "threshold": {"type": "number", "default": 0.1}, "ratio": {"type": "number", "default": 4}}, "required": ["audio_path", "sidechain_path"]}},
            {"name": "crossfade_audio", "description": "Crossfade", "inputSchema": {"type": "object", "properties": {"input1_path": {"type": "string"}, "input2_path": {"type": "string"}, "output_path": {"type": "string"}, "duration": {"type": "number", "default": 10}}, "required": ["input1_path", "input2_path"]}},
            {"name": "concat_audio", "description": "Concatenate", "inputSchema": {"type": "object", "properties": {"input_paths": {"type": "array"}, "output_path": {"type": "string"}}, "required": ["input_paths"]}},
            {"name": "dynamic_equalizer", "description": "Dynamic EQ", "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}, "output_path": {"type": "string"}}, "required": ["audio_path"]}},
            # HEALTHCHECK
            {"name": "healthcheck", "description": "Check FFmpeg availability", "inputSchema": {"type": "object", "properties": {}}},
        ]

    def _handle_tools_list(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": {"tools": self._get_tools_list()}}

    def _handle_tools_call(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")
        
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        request_id = request.get("id")
        
        try:
            wrapper = self._get_wrapper()
            result = None
            
            # Legacy
            if tool_name == "process_audio":
                result = wrapper.process_audio(arguments.get("audio_path"), arguments.get("output_path") or wrapper._generate_output_path("processed"),
                    start_time=arguments.get("start_time"), duration=arguments.get("duration"),
                    sample_rate=arguments.get("sample_rate"), channels=arguments.get("channels"))
            # Phase 1
            elif tool_name == "convert_format":
                result = wrapper.convert_format(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("format_ext"), arguments.get("codec"))
            elif tool_name == "resample_audio":
                result = wrapper.resample_audio(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("sample_rate", 22050))
            elif tool_name == "change_bit_depth":
                result = wrapper.change_bit_depth(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("bit_depth", 16))
            elif tool_name == "convert_channels":
                result = wrapper.convert_channels(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("channels", 1))
            elif tool_name == "extract_audio_from_video":
                result = wrapper.extract_audio_from_video(arguments["video_path"], arguments.get("output_path"), arguments.get("codec", "copy"))
            # Phase 2
            elif tool_name == "adjust_volume":
                result = wrapper.adjust_volume(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("volume_db"), arguments.get("volume_factor"))
            elif tool_name == "loudnorm":
                result = wrapper.loudnorm(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("target_lufs", -16), arguments.get("true_peak", -1.5), arguments.get("loudness_range", 11))
            elif tool_name == "dynaudnorm":
                result = wrapper.dynaudnorm(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("frame_len", 500), arguments.get("filter_size", 31))
            elif tool_name == "acompressor":
                result = wrapper.acompressor(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("threshold", 0.05), arguments.get("ratio", 4), arguments.get("attack", 200), arguments.get("release", 1000))
            elif tool_name == "agate":
                result = wrapper.agate(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("threshold", 0.1), arguments.get("attack", 50), arguments.get("release", 50))
            elif tool_name == "alimiter":
                result = wrapper.alimiter(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("limit", 0.95), arguments.get("attack", 5), arguments.get("release", 50))
            elif tool_name == "compand":
                result = wrapper.compand(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("attack", 0.3), arguments.get("decay", 0.8))
            # Phase 3
            elif tool_name == "highpass_filter":
                result = wrapper.highpass_filter(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("frequency", 80), arguments.get("poles", 2))
            elif tool_name == "lowpass_filter":
                result = wrapper.lowpass_filter(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("frequency", 8000), arguments.get("poles", 2))
            elif tool_name == "bandpass_filter":
                result = wrapper.bandpass_filter(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("frequency", 1000), arguments.get("width", 200))
            elif tool_name == "bandreject_filter":
                result = wrapper.bandreject_filter(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("frequency", 50), arguments.get("width", 20))
            elif tool_name == "equalizer":
                result = wrapper.equalizer(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("frequency", 1000), arguments.get("gain", 5), arguments.get("width", 2), arguments.get("width_type", "o"))
            elif tool_name == "anequalizer":
                result = wrapper.anequalizer(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("bands"))
            elif tool_name == "bass_boost":
                result = wrapper.bass_boost(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("gain", 10), arguments.get("frequency", 100), arguments.get("width", 0.5))
            elif tool_name == "treble_boost":
                result = wrapper.treble_boost(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("gain", 5), arguments.get("frequency", 3000), arguments.get("width", 0.5))
            # Phase 4
            elif tool_name == "afftdn_denoise":
                result = wrapper.afftdn_denoise(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("noise_reduction", 12), arguments.get("noise_floor", -25))
            elif tool_name == "afwtdn_denoise":
                result = wrapper.afwtdn_denoise(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("sigma", 0.1), arguments.get("levels", 10))
            elif tool_name == "adeclick":
                result = wrapper.adeclick(arguments.get("audio_path"), arguments.get("output_path"))
            elif tool_name == "frequency_filter_combo":
                result = wrapper.frequency_filter_combo(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("highpass_freq", 80), arguments.get("lowpass_freq", 8000))
            # Phase 5
            elif tool_name == "change_tempo":
                result = wrapper.change_tempo(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("tempo_ratio", 1.5))
            elif tool_name == "trim_audio":
                result = wrapper.trim_audio(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("start_time", 0), arguments.get("duration"), arguments.get("end_time"))
            elif tool_name == "silenceremove":
                result = wrapper.silenceremove(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("noise_db", -50), arguments.get("min_silence_duration", 0.5))
            elif tool_name == "pad_silence":
                result = wrapper.pad_silence(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("pad_start", 0), arguments.get("pad_end", 2))
            elif tool_name == "reverse_audio":
                result = wrapper.reverse_audio(arguments.get("audio_path"), arguments.get("output_path"))
            elif tool_name == "add_delay":
                result = wrapper.add_delay(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("delay_ms", 1000), arguments.get("decay", 0.5))
            elif tool_name == "pitch_shift_rubberband":
                result = wrapper.pitch_shift_rubberband(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("pitch_ratio", 1.2), arguments.get("tempo_ratio", 1.0))
            # Phase 6
            elif tool_name == "pan_channels":
                result = wrapper.pan_channels(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("channel_layout", "mono"), arguments.get("pan_expr"))
            elif tool_name == "split_channels":
                result = wrapper.split_channels(arguments.get("audio_path"), arguments.get("output_dir"))
            elif tool_name == "mix_audio":
                result = wrapper.mix_audio(arguments["input_paths"], arguments.get("output_path"), arguments.get("weights"))
            elif tool_name == "stereotools":
                result = wrapper.stereotools(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("mode", "ms"))
            elif tool_name == "stereowiden":
                result = wrapper.stereowiden(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("delay", 20), arguments.get("feedback", 0.3), arguments.get("crossfeed", 0.3))
            elif tool_name == "crossfeed":
                result = wrapper.crossfeed(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("strength", 0.2), arguments.get("range_hz", 0.5))
            # Phase 7
            elif tool_name == "add_echo":
                result = wrapper.add_echo(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("delays"), arguments.get("decays"))
            elif tool_name == "chorus_effect":
                result = wrapper.chorus_effect(arguments.get("audio_path"), arguments.get("output_path"))
            elif tool_name == "flanger_effect":
                result = wrapper.flanger_effect(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("delay", 0), arguments.get("depth", 2), arguments.get("regen", 0.5))
            elif tool_name == "phaser_effect":
                result = wrapper.phaser_effect(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("speed", 0.5), arguments.get("decay", 0.5))
            elif tool_name == "tremolo_effect":
                result = wrapper.tremolo_effect(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("freq", 5), arguments.get("depth", 0.5))
            elif tool_name == "vibrato_effect":
                result = wrapper.vibrato_effect(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("freq", 5), arguments.get("depth", 0.5))
            elif tool_name == "deesser":
                result = wrapper.deesser(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("intensity", 1), arguments.get("frequency", 5500))
            elif tool_name == "crystalizer":
                result = wrapper.crystalizer(arguments.get("audio_path"), arguments.get("output_path"), arguments.get("intensity", 2))
            # Phase 8
            elif tool_name == "audio_stats":
                result = wrapper.audio_stats(arguments.get("audio_path"))
            elif tool_name == "silencedetect":
                result = wrapper.silencedetect(arguments.get("audio_path"), arguments.get("noise_db", -50), arguments.get("min_duration", 0.5))
            elif tool_name == "volumedetect":
                result = wrapper.volumedetect(arguments.get("audio_path"))
            elif tool_name == "ebur128":
                result = wrapper.ebur128(arguments.get("audio_path"))
            elif tool_name == "replaygain":
                result = wrapper.replaygain(arguments.get("audio_path"))
            elif tool_name == "astats":
                result = wrapper.astats(arguments.get("audio_path"))
            elif tool_name == "spectral_stats":
                result = wrapper.spectral_stats(arguments.get("audio_path"))
            # Phase 9
            elif tool_name == "sidechain_compress":
                result = wrapper.sidechain_compress(arguments.get("audio_path"), arguments["sidechain_path"], arguments.get("output_path"), arguments.get("threshold", 0.1), arguments.get("ratio", 4))
            elif tool_name == "crossfade_audio":
                result = wrapper.crossfade_audio(arguments["input1_path"], arguments["input2_path"], arguments.get("output_path"), arguments.get("duration", 10))
            elif tool_name == "concat_audio":
                result = wrapper.concat_audio(arguments["input_paths"], arguments.get("output_path"))
            elif tool_name == "dynamic_equalizer":
                result = wrapper.dynamic_equalizer(arguments.get("audio_path"), arguments.get("output_path"))
            # Healthcheck
            elif tool_name == "healthcheck":
                result = wrapper.healthcheck()
            else:
                return self._error_response(request_id, -32601, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False},
            }
            
        except InferenceError as e:
            return {"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": str(e)}], "isError": True, "error": str(e)}}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": str(exc)}], "isError": True, "error": str(exc)}}

    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        if method == "initialize":
            return self._handle_initialize(request)
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self._handle_tools_list(request)
        if method == "tools/call":
            return self._handle_tools_call(request)
        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": {}}
        return self._error_response(request.get("id"), -32601, f"Method not found: {method}")

    def run(self) -> None:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError as e:
                error_resp = self._error_response(None, -32700, f"Parse error: {e}")
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                request_id = request.get("id") if isinstance(request, dict) else None
                error_resp = self._error_response(request_id, -32603, f"Internal error: {e}")
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()


def main() -> None:
    server = FFmpegMCPServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
