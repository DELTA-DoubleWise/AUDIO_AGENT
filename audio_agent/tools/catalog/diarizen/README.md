# DiariZen Speaker Diarization Tool

Speaker diarization tool using DiariZen with the BUT-FIT/diarizen-wavlm-large-s80-md model.

## ⚠️ Setup Required

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Download Model
```bash
audio-agent-download-models --models diarizen
```

Or download all models:
```bash
audio-agent-download-models --all
```

### 3. Setup Environment

```bash
# Setup the tool environment (recommended)
./setup.sh

# Verify it's ready
./test_env.sh
```

## Usage

### Tool: diarize

Perform speaker diarization on an audio file. Identifies who speaks when, returning speaker segments with start/end timestamps.

**Input:**
```json
{
  "audio_path": "/path/to/audio.wav",
  "num_speakers": null,
  "min_speakers": null,
  "max_speakers": null
}
```

**Output:**
```
Speaker diarization results for: audio.wav
Total segments: 5

Segments:
  [  0.00s -   2.70s] speaker_0
  [  0.80s -  13.60s] speaker_3
  [  5.80s -   6.40s] speaker_0
  ...

Detected speakers: speaker_0, speaker_3
```

**Parameters:**
- `audio_path` (required): Path to the audio file
- `num_speakers` (optional): Expected number of speakers (auto-detected if not provided)
- `min_speakers` (optional): Minimum number of speakers
- `max_speakers` (optional): Maximum number of speakers

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to DiariZen model | `/lihaoyu/workspace/AUDIO_AGENT/models/diarizen-wavlm-large-s80-md` |
| `DEVICE` | Device to use (auto/cpu/cuda) | `auto` |

## Model Information

- **Model**: BUT-FIT/diarizen-wavlm-large-s80-md
- **Description**: Speaker diarization model based on WavLM Large with structured pruning
- **License**: CC BY-NC 4.0 (Non-Commercial)
- **Paper**: [Leveraging self-supervised learning for speaker diarization](https://arxiv.org/abs/2505.24111)

## Troubleshooting

### Environment not found
Run setup: `./setup.sh`

### Model download fails
Check HuggingFace access: `huggingface-cli login`

### Out of memory
The model requires ~8GB RAM. Reduce batch size or use CPU with:
```bash
export DEVICE=cpu
```

### Import errors in server
Ensure dependencies are installed:
```bash
rm -rf .venv
./setup.sh
```

## References

- [DiariZen GitHub](https://github.com/BUTSpeechFIT/DiariZen)
- [Model on HuggingFace](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md)
