# ASR Qwen3 Tool

Automatic Speech Recognition (ASR) using Alibaba's Qwen3-ASR-1.7B model.

## ⚠️ IMPORTANT: Setup Required

**You MUST complete all setup steps BEFORE running this tool!**

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv
```

### 2. Download Models

```bash
# Download the ASR model and forced aligner (~4.5GB total)
audio-agent-download-models --models qwen3-asr qwen3-aligner

# Or download all models
audio-agent-download-models --all
```

### 3. Setup Tool Environment

```bash
# Setup the tool environment (recommended)
./setup.sh

# Verify it's ready
./test_env.sh
```

## Overview

This tool provides speech-to-text transcription using Alibaba's Qwen3-ASR-1.7B model. It supports:

- **52 Languages**: 30 languages + 22 Chinese dialects
- **Automatic Language Detection**: Set `language: "auto"`
- **Multilingual Input**: Handle mixed-language audio
- **High Accuracy**: Optimized for conversational speech

## Model Information

- **Model**: Qwen/Qwen3-ASR-1.7B
- **Size**: 1.7B parameters (~3.2GB weights)
- **License**: Apache 2.0
- **Source**: [HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- **GitHub**: https://github.com/QwenLM/Qwen3-ASR

## Quick Start

### Setup

```bash
# Setup environment
./setup.sh

# Verify setup
./test_env.sh
```

### Test Server

```bash
cd audio_agent/tools/catalog/asr_qwen3
.venv/bin/python server.py
```

## Usage

### Tool: `asr_transcribe`

Transcribe speech in an audio file.

**Input:**
```json
{
  "audio_path": "/path/to/audio.wav",
  "language": "auto",
  "prompt": ""
}
```

**Output:**
```json
{
  "content": [
    {"type": "text", "text": "Hello, this is the transcribed text."}
  ],
  "isError": false
}
```

**Parameters:**
- `audio_path` (required): Path to the audio file
- `language` (optional): Language code or `"auto"` for automatic detection
  - Supported: `"auto"`, `"zh"` (Chinese), `"en"` (English), `"ja"` (Japanese), `"ko"` (Korean), and 48 others
- `prompt` (optional): Optional prompt to guide transcription

### Tool: `asr_transcribe_with_timestamps`

Transcribe with word-level timestamps (uses forced alignment).

**Input:**
```json
{
  "audio_path": "/path/to/audio.wav",
  "language": "auto"
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Model path (local or HuggingFace) | `/lihaoyu/workspace/AUDIO_AGENT/models/Qwen3-ASR-1.7B` |
| `DEVICE` | Device (`auto`, `cuda`, `cpu`) | `auto` |
| `LANGUAGE` | Default language | `auto` |
| `ALIGNER_PATH` | Forced aligner model path (for timestamps) | `/lihaoyu/workspace/AUDIO_AGENT/models/Qwen3-ForcedAligner-0.6B` |
| `USE_ALIGNER` | Enable forced aligner for timestamps | `true` |

### Resource Requirements

- **Memory**: ~8GB RAM
- **GPU**: Recommended but not required
- **Storage**: ~3.2GB for model weights
- **CPU**: 4 cores recommended

## Testing with MCP Client

```python
import asyncio
from audio_agent.tools.mcp import MCPClient

async def test():
    client = MCPClient(
        command=[".venv/bin/python", "server.py"],
        working_dir="/path/to/audio_agent/tools/catalog/asr_qwen3",
        env={"DEVICE": "cuda"},
        startup_timeout_sec=300
    )
    
    await client.start()
    
    tools = await client.list_tools()
    print(f"Available: {[t.name for t in tools]}")
    
    result = await client.call_tool(
        "asr_transcribe",
        {
            "audio_path": "/path/to/your/audio.wav",
            "language": "auto"
        }
    )
    print(f"Result: {result}")
    
    await client.stop()

asyncio.run(test())
```

## Troubleshooting

### Environment not found

You forgot to setup:
```bash
./setup.sh
```

### Model download fails

The model will be downloaded automatically on first use. To pre-download:

```bash
# Login to HuggingFace (if needed)
huggingface-cli login

# Pre-download model
python -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
model_path = 'Qwen/Qwen3-ASR-1.7B'
AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
AutoModelForSpeechSeq2Seq.from_pretrained(model_path, trust_remote_code=True)
"
```

### Out of Memory

If you get OOM errors:
- Use CPU instead: `DEVICE=cpu`
- Reduce batch size (modify server.py)
- Close other GPU applications

### Slow inference

- Ensure GPU is being used: Check `nvidia-smi`
- First call is slow (model loading)
- Subsequent calls are faster

## Dependencies

See `pyproject.toml` for full list:
- `qwen-asr` - Official Qwen3-ASR package
- `torch>=2.0.0` - PyTorch
- `transformers>=4.40.0` - HuggingFace transformers
- `librosa>=0.10.0` - Audio processing
- `soundfile>=0.12.0` - Audio I/O
- `accelerate>=0.25.0` - Model loading utilities

## See Also

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-1.7B HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Tool Catalog README](../README.md)
