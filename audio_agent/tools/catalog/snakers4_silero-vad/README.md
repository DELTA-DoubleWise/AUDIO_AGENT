# Silero VAD Tool

Voice Activity Detection tool using [Silero VAD](https://github.com/snakers4/silero-vad) for speech segmentation.

## Setup

```bash
./setup.sh
./test_env.sh
```

## Tools

### `vad_predict`

Run Voice Activity Detection on an audio file.

**Input:**
```json
{
  "audio_path": "/path/to/audio.wav",
  "sampling_rate": 16000
}
```

**Output:**
```json
{
  "speech_segments": [
    {"start": 0.5, "end": 2.3},
    {"start": 3.1, "end": 5.8}
  ],
  "audio_duration": 10.0
}
```

### `healthcheck`

Check if the VAD model is ready.

## Configuration

- `MODEL_DEVICE`: Device to run inference on (default: `cpu`)
- `TORCH_NUM_THREADS`: Number of threads for CPU inference (default: `1`)
