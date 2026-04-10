# FireRedASR2S Tool

FireRedASR2S AED (Attention-based Encoder-Decoder) ASR tool for the Audio Agent Framework.

## Features

- **SOTA Performance**: 2.89% average CER on 4 public Mandarin benchmarks
- **Multi-language Support**: Chinese (Mandarin + 20+ dialects), English, code-switching
- **Word-level Timestamps**: AED model supports precise word-level timestamps
- **Industrial Grade**: Outperforms Doubao-ASR, Qwen3-ASR-1.7B, Fun-ASR

## Model Information

- **Model**: FireRedASR2-AED
- **Source**: [HuggingFace](https://huggingface.co/FireRedTeam/FireRedASR2-AED)
- **Repo**: [GitHub](https://github.com/FireRedTeam/FireRedASR2S)
- **Local Path**: `/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/models/fireredasr2s/`

## Setup

```bash
# Ensure persistent uv is activated
source /lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh

# Run setup
./setup.sh
```

## Testing

```bash
./test_env.sh
```

## Usage

### Python API

```python
from model import ModelWrapper

model = ModelWrapper(device="cuda")
result = model.predict("audio.wav")
print(result["text"])
print(result["segments"])  # Word-level timestamps
```

### MCP Server

The tool exposes three MCP tools:

1. **transcribe**: Basic transcription
2. **transcribe_with_timestamps**: Transcription with word-level timestamps
3. **healthcheck**: Check model status

## Configuration

Environment variables:
- `MODEL_PATH`: Path to model weights (default: `/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/models/fireredasr2s/`)
- `DEVICE`: Device to use - "auto", "cuda", "cpu" (default: "auto")

## Audio Format

Recommended: 16kHz 16-bit mono PCM WAV

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

## Dependencies

- torch>=2.0.0
- transformers>=4.40.0
- fireredasr2s (from GitHub)
- See `pyproject.toml` for complete list
