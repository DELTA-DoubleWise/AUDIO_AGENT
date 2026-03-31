# Librosa Audio Analysis Tool

Audio analysis tool using [librosa](https://librosa.org/) for onset, beat, and tempo extraction.

## Setup

```bash
./setup.sh
./test_env.sh
```

## Tools

### `analyze_rhythm`

Run onset, beat, and tempo analysis on an audio file.

**Input:**
```json
{"audio_path": "/path/to/audio.wav"}
```

**Output:**
```json
{
  "tempo": 120.5,
  "beats": [0, 4, 8, 12],
  "beat_times_sec": [0.0, 0.5, 1.0, 1.5]
}
```

### `healthcheck`

Check whether the librosa runtime is available.

## Configuration

No special environment variables required. The tool runs in its isolated `.venv`.
