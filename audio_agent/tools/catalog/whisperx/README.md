# WhisperX Tool

Automatic Speech Recognition (ASR) with word-level timestamps using WhisperX.

## ⚠️ IMPORTANT: Setup Required

**You MUST complete all setup steps BEFORE running this tool!**

### Prerequisites

1. **System dependencies** (ffmpeg):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

2. **Setup Tool Environment**:
   ```bash
   # Setup the tool environment (recommended)
   ./setup.sh
   
   # Verify it's ready
   ./test_env.sh
   ```

## Overview

This tool provides speech-to-text transcription using [WhisperX](https://github.com/m-bain/whisperX). It features:

- **99 Language Support**: Multilingual transcription
- **Word-Level Timestamps**: Precise alignment via forced alignment
- **VAD Segmentation**: Voice activity detection for better accuracy
- **Speaker Diarization**: Identify "who spoke when" via pyannote-audio

## Model Information

- **Model**: OpenAI Whisper (various sizes: tiny, base, small, medium, large)
- **Alignment**: wav2vec 2.0 based forced alignment
- **VAD**: pyannote.audio or silero VAD
- **License**: MIT (WhisperX), Apache 2.0 (Whisper)
- **Source**: [WhisperX GitHub](https://github.com/m-bain/whisperX)

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
cd audio_agent/tools/catalog/whisperx
.venv/bin/python server.py
```

Then send JSON-RPC requests via stdin:

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize"}
{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "transcribe_whisperx", "arguments": {"audio_path": "/path/to/audio.wav"}}}
```

## Usage

### Tool: `transcribe_audio`

Transcribe speech in an audio file with word-level timestamps.

**Input:**
```json
{
  "audio_path": "/path/to/audio.wav",
  "language": "en"
}
```

**Output:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\n  \"segments\": [\n    {\n      \"start\": 0.0,\n      \"end\": 2.5,\n      \"text\": \"Hello world\",\n      \"words\": [\n        {\"word\": \"Hello\", \"start\": 0.0, \"end\": 0.5, \"score\": 0.95},\n        {\"word\": \"world\", \"start\": 0.6, \"end\": 1.2, \"score\": 0.98}\n      ]\n    }\n  ],\n  \"language\": \"en\"\n}"
    }
  ],
  "isError": false
}
```

**Parameters:**
- `audio_path` (required): Path to the audio file
- `language` (optional): Language code (e.g., "en", "fr", "de"). If not provided, auto-detected.

### Tool: `transcribe_with_diarization`

Transcribe speech with **speaker diarization** - identifies "who spoke when". Returns speaker labels (SPEAKER_01, SPEAKER_02, etc.) for each segment and word.

**Input:**
```json
{
  "audio_path": "/path/to/meeting.wav",
  "language": "en",
  "min_speakers": 2,
  "max_speakers": 4
}
```

**Output:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\n  \"segments\": [\n    {\n      \"start\": 0.0,\n      \"end\": 2.5,\n      \"text\": \"Hello everyone\",\n      \"speaker\": \"SPEAKER_01\",\n      \"words\": [\n        {\"word\": \"Hello\", \"start\": 0.0, \"end\": 0.5, \"speaker\": \"SPEAKER_01\"},\n        {\"word\": \"everyone\", \"start\": 0.6, \"end\": 1.2, \"speaker\": \"SPEAKER_01\"}\n      ]\n    },\n    {\n      \"start\": 2.8,\n      \"end\": 5.0,\n      \"text\": \"Thanks for joining\",\n      \"speaker\": \"SPEAKER_02\",\n      \"words\": [\n        {\"word\": \"Thanks\", \"start\": 2.8, \"end\": 3.2, \"speaker\": \"SPEAKER_02\"},\n        {\"word\": \"for\", \"start\": 3.3, \"end\": 3.5, \"speaker\": \"SPEAKER_02\"},\n        {\"word\": \"joining\", \"start\": 3.6, \"end\": 4.0, \"speaker\": \"SPEAKER_02\"}\n      ]\n    }\n  ],\n  \"language\": \"en\",\n  \"speakers\": [\"SPEAKER_01\", \"SPEAKER_02\"],\n  \"num_speakers\": 2\n}"
    }
  ],
  "isError": false
}
```

**Parameters:**
- `audio_path` (required): Path to the audio file
- `language` (optional): Language code (e.g., "en", "fr", "de"). If not provided, auto-detected.
- `min_speakers` (optional): Minimum number of speakers to detect
- `max_speakers` (optional): Maximum number of speakers to detect

**Limitations:**
- Overlapping speech is not handled well
- Accuracy depends on audio quality and speaker distinctiveness
- First run downloads pyannote models (~few hundred MB)

### Tool: `healthcheck`

Check if WhisperX is properly configured.

**Input:**
```json
{}
```

**Output:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\n  \"status\": \"ready\",\n  \"message\": \"Model loaded\",\n  \"model_loaded\": true,\n  \"model_arch\": \"small\",\n  \"device\": \"cpu\",\n  \"vad_method\": \"pyannote\"\n}"
    }
  ],
  "isError": false
}
```

## Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `MODEL_ARCH` | No | Whisper model size (tiny, base, small, medium, large) | `small` |
| `DEVICE` | No | Compute device (`cpu`, `cuda`) | `cuda` |
| `VAD_METHOD` | No | VAD method (`pyannote`, `silero`, `None`) | `pyannote` |
| `DIARIZATION_MODEL_PATH` | No | Path to pyannote diarization model | `/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/models/pyannote-speaker-diarization-community-1` |
| `HF_HOME` | No | Where Whisper models are downloaded on first use | `/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/models` |

### Pre-downloaded Models

Pyannote speaker diarization models are pre-downloaded to:
```
/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/models/pyannote-speaker-diarization-community-1/
├── config.yaml
├── embedding/
├── plda/
└── segmentation/
```

**Note**: `DIARIZATION_MODEL_PATH` can be changed to use a different model location.

### Model Sizes

| Model | Parameters | English-only | Multilingual | VRAM |
|-------|------------|--------------|--------------|------|
| tiny | 39 M | tiny.en | tiny | ~1 GB |
| base | 74 M | base.en | base | ~1 GB |
| small | 244 M | small.en | small | ~2 GB |
| medium | 769 M | medium.en | medium | ~5 GB |
| large | 1550 M | N/A | large | ~10 GB |
| large-v2 | 1550 M | N/A | large-v2 | ~10 GB |
| large-v3 | 1550 M | N/A | large-v3 | ~10 GB |

### Changing Model Size

Edit `config.yaml` or set environment variable:

```bash
export MODEL_ARCH=large-v3
```

## Testing with MCP Client

```python
import asyncio
from audio_agent.tools.catalog import load_mcp_server_config
from audio_agent.tools.mcp import MCPClient

async def test():
    config = load_mcp_server_config("whisperx")
    client = MCPClient(
        command=config.command,
        working_dir=config.working_dir,
        startup_timeout_sec=300
    )
    
    await client.start()
    
    tools = await client.list_tools()
    print(f"Available: {[t.name for t in tools]}")
    
    result = await client.call_tool(
        "transcribe_audio",
        {
            "audio_path": "/path/to/your/audio.wav",
            "language": "en"
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

Models are downloaded automatically on first use. To pre-download:

```bash
# The wrapper will download models to the cache directory
# You can monitor progress in the logs
```

### Out of Memory

If you get OOM errors:
- Use a smaller model: `MODEL_ARCH=base`
- Force CPU if GPU has issues: `DEVICE=cpu`
- Reduce batch size (modify model.py)

### Slow inference

- Ensure GPU is being used: Check `nvidia-smi`
- First call is slow (model loading)
- Subsequent calls are faster

### pyannote.audio VAD issues

If you encounter issues with pyannote VAD:
- Try silero VAD: `VAD_METHOD=silero`
- Or disable VAD: `VAD_METHOD=None`

### Overlapping Speech

WhisperX diarization does not handle overlapping speech well. This is a known limitation.

## Dependencies

See `pyproject.toml` for full list:
- `whisperx==3.8.4` - WhisperX ASR
- `torch>=2.0.0` - PyTorch
- `torchaudio>=2.0.0` - PyTorch audio

## See Also

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Tool Catalog README](../README.md)
