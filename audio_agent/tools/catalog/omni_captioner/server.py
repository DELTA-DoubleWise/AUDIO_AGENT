#!/usr/bin/env python3
"""
Omni-Captioner MCP Server

MCP server implementation for audio captioning using Qwen3-Omni-30B-A3B-Captioner.
Generates detailed descriptions of audio content including speech, environmental sounds,
music, and cinematic sound effects.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


class OmniCaptionerServer:
    """MCP Server for Qwen3-Omni Audio Captioning."""
    
    def __init__(self):
        self._initialized = False
        self._model = None
        self._processor = None
        self._model_path = os.environ.get(
            "MODEL_PATH", 
            "Qwen/Qwen3-Omni-30B-A3B-Captioner"
        )
        self._device = os.environ.get("DEVICE", "auto")
        self._max_length = int(os.environ.get("MAX_LENGTH", "8192"))
        
        # Tool definitions
        self._tools = [
            {
                "name": "caption_audio",
                "description": "Generate a detailed caption/description of an audio file. Analyzes speech, environmental sounds, music, and other audio content to produce a comprehensive textual description. Best for audio clips up to 30 seconds.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to caption"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum length of the generated caption",
                            "default": 8192
                        }
                    },
                    "required": ["audio_path"]
                }
            }
        ]
    
    def _load_model(self) -> None:
        """Lazy load the Qwen3-Omni model and processor."""
        if self._model is not None:
            return
        
        try:
            from transformers import (
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor
            )
            import torch
        except ImportError as e:
            raise RuntimeError(
                f"Missing required packages. Ensure environment is set up correctly: {e}"
            ) from e
        
        print(f"Loading Qwen3-Omni Captioner model: {self._model_path}", file=sys.stderr)
        
        try:
            # Determine device and dtype
            if self._device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device
            
            # Load processor
            self._processor = Qwen3OmniMoeProcessor.from_pretrained(self._model_path)
            
            # Load model with appropriate settings
            load_kwargs = {
                "device_map": device if device != "cpu" else None,
            }
            
            # Use auto dtype if on CUDA
            if device == "cuda":
                load_kwargs["torch_dtype"] = "auto"
            
            self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self._model_path,
                **load_kwargs
            )
            
            print(f"Model loaded successfully on {device}", file=sys.stderr)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def run(self) -> None:
        """Run the server."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response:
                    self._send_response(response)
            except json.JSONDecodeError as e:
                self._send_error(None, -32700, f"Parse error: {e}")
            except Exception as e:
                request_id = request.get("id") if isinstance(request, dict) else None
                self._send_error(request_id, -32603, f"Internal error: {e}")
    
    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle JSON-RPC request."""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return self._handle_tools_call(request_id, params)
        elif method == "shutdown":
            return self._handle_shutdown(request_id)
        else:
            return self._error_response(request_id, -32601, f"Method not found: {method}")
    
    def _handle_initialize(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "omni-captioner-server",
                    "version": "1.0.0"
                }
            }
        }
    
    def _handle_tools_list(self, request_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": self._tools}
        }
    
    def _handle_tools_call(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle tools/call request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            result = self._execute_tool(tool_name, arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as e:
            error_msg = str(e)
            print(f"Tool execution error: {error_msg}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {error_msg}"}],
                    "isError": True,
                    "error": error_msg
                }
            }
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Execute a tool."""
        self._load_model()
        
        if tool_name == "caption_audio":
            return self._caption_audio(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _caption_audio(self, arguments: dict) -> dict[str, Any]:
        """Generate a detailed caption for an audio file."""
        audio_path = arguments.get("audio_path")
        max_length = arguments.get("max_length", self._max_length)
        
        if not audio_path:
            raise ValueError("audio_path is required")
        
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        print(f"Captioning audio: {audio_path}", file=sys.stderr)
        
        try:
            from qwen_omni_utils import process_mm_info
            import torch
        except ImportError as e:
            raise RuntimeError(
                f"Missing qwen-omni-utils package. Ensure environment is set up correctly: {e}"
            ) from e
        
        try:
            # Prepare messages (audio only, no text prompt)
            messages = [{
                "role": "user",
                "content": [{"type": "audio", "audio": audio_path}]
            }]
            
            # Process multimedia info
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            # Apply chat template
            text = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Process inputs
            inputs = self._processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True
            )
            
            # Move to device and dtype
            inputs = inputs.to(self._model.device)
            if hasattr(self._model, 'dtype'):
                inputs = inputs.to(self._model.dtype)
            
            # Generate caption
            print(f"Generating caption (max_length={max_length})...", file=sys.stderr)
            
            with torch.no_grad():
                text_ids, _ = self._model.generate(
                    **inputs,
                    thinker_return_dict_in_generate=True,
                    thinker_max_new_tokens=max_length,
                    thinker_do_sample=True,
                    thinker_top_p=0.95,
                    thinker_top_k=20,
                    thinker_temperature=0.6,
                    speaker="Chelsie",
                    use_audio_in_video=True,
                    return_audio=False
                )
            
            # Decode response
            response = self._processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"Caption generated: {response[:100]}...", file=sys.stderr)
            
            return {
                "content": [{"type": "text", "text": response}],
                "isError": False
            }
            
        except Exception as e:
            print(f"Caption generation failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Caption generation failed: {e}") from e
    
    def _handle_shutdown(self, request_id: Any) -> dict[str, Any]:
        """Handle shutdown request."""
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    
    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message}
        }
    
    def _send_response(self, response: dict[str, Any]) -> None:
        """Send response to stdout."""
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    
    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        """Send error response."""
        self._send_response(self._error_response(request_id, code, message))


def main():
    """Main entry point."""
    # Ensure unbuffered output for proper JSON-RPC communication
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    server = OmniCaptionerServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log to stderr only - never to stdout (breaks JSON-RPC)
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
