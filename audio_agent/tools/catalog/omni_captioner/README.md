# Omni Captioner Tool

Audio captioning using Qwen3-Omni API via DashScope. This is an **API-based tool** - no local model download required.

## Features

- Generate text captions for audio files
- Optional audio response generation
- Supports multiple audio formats (wav, mp3, ogg, m4a, flac)
- Uses Qwen3-Omni-Flash model via DashScope API

## Setup

### 1. Set DashScope API Key

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

Get your API key from: https://dashscope.console.aliyun.com/

### 2. Setup Environment

```bash
./setup.sh
```

### 3. Verify Installation

```bash
./test_env.sh
```

## Usage

### Tool: omni_caption

Generate text caption for an audio file:

```json
{
  "audio_path": "/path/to/audio.wav",
  "prompt": "Describe this audio in detail."
}
```

### Tool: omni_caption_with_audio

Generate caption + audio response:

```json
{
  "audio_path": "/path/to/audio.wav",
  "prompt": "Describe this audio in detail.",
  "voice": "Cherry",
  "output_audio_path": "/path/to/output.wav"
}
```

## Configuration

Environment variables (set in `config.yaml` or export before use):

| Variable | Description | Default |
|----------|-------------|---------|
| `DASHSCOPE_API_KEY` | Your DashScope API key | (required) |
| `DASHSCOPE_BASE_URL` | API base URL | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `DEFAULT_MODEL` | Model ID | `qwen3-omni-flash` |
| `DEFAULT_VOICE` | Voice for audio | `Cherry` |

## Troubleshooting

### "DASHSCOPE_API_KEY not set"
Set your API key:
```bash
export DASHSCOPE_API_KEY="your-key"
```

### "Audio file not found"
Ensure the audio path is absolute or relative to the server working directory.

### API errors
Check your API key and quota at https://dashscope.console.aliyun.com/

## API Reference

- **Model**: `qwen3-omni-flash`
- **Provider**: DashScope (Alibaba Cloud)
- **Capabilities**: Text + Audio input/output
