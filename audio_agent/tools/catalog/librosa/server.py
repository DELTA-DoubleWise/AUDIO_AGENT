#!/usr/bin/env python3
"""MCP Server for librosa audio analysis tool - Extended with 15+ tools."""

from __future__ import annotations

import json
import sys
from typing import Any

try:
    from .model import ModelWrapper
except ImportError:
    from model import ModelWrapper


class LibrosaMCPServer:
    def __init__(self) -> None:
        self._initialized = False
        self._wrapper: ModelWrapper | None = None

    def _get_wrapper(self) -> ModelWrapper:
        if self._wrapper is None:
            self._wrapper = ModelWrapper()
        return self._wrapper

    def _handle_initialize(self, request: dict[str, Any]) -> dict[str, Any]:
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "librosa-mcp-server",
                    "version": "2.0.0",
                },
            },
        }

    def _handle_tools_list(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")
        
        tools = [
            # Original tool (kept for compatibility)
            {
                "name": "analyze_rhythm",
                "description": "Musical rhythm analysis - detects tempo (BPM), beat positions, and onset events in music/audio. Use for music analysis tasks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            # =========================================================================
            # RHYTHM ANALYSIS TOOLS
            # =========================================================================
            {
                "name": "analyze_beats",
                "description": "Detect tempo (BPM) and beat positions in audio. Returns tempo and beat timestamps.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        },
                        "units": {
                            "type": "string",
                            "enum": ["time", "frames"],
                            "default": "time",
                            "description": "Units for beat timestamps: 'time' (seconds) or 'frames'."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "analyze_onsets",
                "description": "Detect note onset events and onset strength in audio.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            # =========================================================================
            # TIMBRE & SPECTRAL FEATURES
            # =========================================================================
            {
                "name": "extract_mfcc",
                "description": "Extract MFCC (Mel-frequency cepstral coefficients) features for timbre analysis.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        },
                        "n_mfcc": {
                            "type": "integer",
                            "default": 13,
                            "description": "Number of MFCC coefficients to extract."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "analyze_spectral_features",
                "description": "Extract spectral features: centroid, bandwidth, rolloff, contrast, and flatness.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "extract_rms_energy",
                "description": "Extract RMS energy (loudness) features including dynamic range.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "extract_zero_crossing_rate",
                "description": "Extract zero-crossing rate (ZCR) for texture/noise analysis.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            # =========================================================================
            # HARMONIC & PITCH ANALYSIS
            # =========================================================================
            {
                "name": "extract_chroma",
                "description": "Extract chroma features (12-dimensional pitch class profile) for harmonic analysis.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "detect_key",
                "description": "Estimate the musical key (e.g., 'C major', 'A minor') from audio.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "estimate_tuning",
                "description": "Estimate tuning deviation from A440 in cents.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "analyze_pitch",
                "description": "Extract pitch/F0 contour using pYIN algorithm. Returns mean pitch and voiced ratio.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            # =========================================================================
            # SEGMENTATION & STRUCTURE
            # =========================================================================
            {
                "name": "segment_audio",
                "description": "Segment audio into non-silent regions. Returns list of segments with start/end times.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        },
                        "top_db": {
                            "type": "integer",
                            "default": 40,
                            "description": "Threshold (in dB) for considering a region as silent."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            # =========================================================================
            # UTILITY
            # =========================================================================
            {
                "name": "get_audio_info",
                "description": "Get basic audio file information: duration, sample rate, channels, format.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            # =========================================================================
            # AUDIO TRANSFORMATION TOOLS
            # =========================================================================
            {
                "name": "apply_pitch_shift",
                "description": "Shift the pitch of audio by specified semitones. Returns path to transformed audio.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the input audio file."
                        },
                        "n_steps": {
                            "type": "number",
                            "description": "Number of semitones to shift (positive=up, negative=down)."
                        },
                        "semitones": {
                            "type": "number",
                            "description": "Alias for n_steps. Number of semitones to shift (positive=up, negative=down)."
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional output path. If not provided, saved to temp directory."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "apply_time_stretch",
                "description": "Time-stretch audio by a rate (1.0=original, 2.0=2x faster). Returns path to transformed audio.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the input audio file."
                        },
                        "rate": {
                            "type": "number",
                            "description": "Stretch rate: 1.0=original, 0.5=half speed, 2.0=double speed."
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional output path. If not provided, saved to temp directory."
                        }
                    },
                    "required": ["audio_path", "rate"]
                }
            },
            {
                "name": "remove_silence",
                "description": "Remove leading and trailing silence from audio. Returns path to trimmed audio.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the input audio file."
                        },
                        "top_db": {
                            "type": "integer",
                            "default": 60,
                            "description": "Threshold (in dB) for considering audio as silence."
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional output path. If not provided, saved to temp directory."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "separate_harmonic_percussive",
                "description": "Separate audio into harmonic (melodic) and percussive components using HPSS.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the input audio file."
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Optional output directory. If not provided, saved to temp directory."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            # Healthcheck (environment verification)
            {
                "name": "healthcheck",
                "description": "Check whether the librosa runtime is available.",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"tools": tools}
        }

    def _handle_tools_call(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")
        
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        request_id = request.get("id")
        
        try:
            audio_path = arguments.get("audio_path")
            
            # Original tool
            if tool_name == "analyze_rhythm":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().predict(audio_path).to_dict()
            
            # RHYTHM ANALYSIS
            elif tool_name == "analyze_beats":
                if not audio_path:
                    raise ValueError("audio_path is required")
                units = arguments.get("units", "time")
                result = self._get_wrapper().analyze_beats(audio_path, units).to_dict()
            
            elif tool_name == "analyze_onsets":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().analyze_onsets(audio_path).to_dict()
            
            # TIMBRE & SPECTRAL
            elif tool_name == "extract_mfcc":
                if not audio_path:
                    raise ValueError("audio_path is required")
                n_mfcc = arguments.get("n_mfcc", 13)
                result = self._get_wrapper().extract_mfcc(audio_path, n_mfcc).to_dict()
            
            elif tool_name == "analyze_spectral_features":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().analyze_spectral_features(audio_path).to_dict()
            
            elif tool_name == "extract_rms_energy":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().extract_rms_energy(audio_path).to_dict()
            
            elif tool_name == "extract_zero_crossing_rate":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().extract_zcr(audio_path).to_dict()
            
            # HARMONIC & PITCH
            elif tool_name == "extract_chroma":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().extract_chroma(audio_path).to_dict()
            
            elif tool_name == "detect_key":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().detect_key(audio_path).to_dict()
            
            elif tool_name == "estimate_tuning":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().estimate_tuning(audio_path).to_dict()
            
            elif tool_name == "analyze_pitch":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().analyze_pitch(audio_path).to_dict()
            
            # SEGMENTATION & STRUCTURE
            elif tool_name == "segment_audio":
                if not audio_path:
                    raise ValueError("audio_path is required")
                top_db = arguments.get("top_db", 40)
                result = self._get_wrapper().segment_audio(audio_path, top_db).to_dict()
            
            # UTILITY
            elif tool_name == "get_audio_info":
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().get_audio_info(audio_path).to_dict()
            
            # AUDIO TRANSFORMATIONS
            elif tool_name == "apply_pitch_shift":
                if not audio_path:
                    raise ValueError("audio_path is required")
                # Accept both n_steps and semitones (LLM may use either)
                n_steps = arguments.get("n_steps") or arguments.get("semitones")
                if n_steps is None:
                    raise ValueError("n_steps or semitones is required")
                output_path = arguments.get("output_path")
                result = self._get_wrapper().apply_pitch_shift(audio_path, float(n_steps), output_path).to_dict()
            
            elif tool_name == "apply_time_stretch":
                if not audio_path:
                    raise ValueError("audio_path is required")
                rate = arguments.get("rate")
                if rate is None:
                    raise ValueError("rate is required")
                output_path = arguments.get("output_path")
                result = self._get_wrapper().apply_time_stretch(audio_path, rate, output_path).to_dict()
            
            elif tool_name == "remove_silence":
                if not audio_path:
                    raise ValueError("audio_path is required")
                top_db = arguments.get("top_db", 60)
                output_path = arguments.get("output_path")
                result = self._get_wrapper().remove_silence(audio_path, top_db, output_path).to_dict()
            
            elif tool_name == "separate_harmonic_percussive":
                if not audio_path:
                    raise ValueError("audio_path is required")
                output_dir = arguments.get("output_dir")
                result = self._get_wrapper().separate_harmonic_percussive(audio_path, output_dir).to_dict()
            
            # HEALTHCHECK
            elif tool_name == "healthcheck":
                result = self._get_wrapper().healthcheck()
            
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False,
                },
            }
            
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                    "error": str(exc),
                },
            }

    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

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
    server = LibrosaMCPServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
