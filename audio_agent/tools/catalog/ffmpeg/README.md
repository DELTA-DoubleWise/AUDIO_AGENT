# FFmpeg Audio Processing Tool

Audio processing tool using [FFmpeg](https://ffmpeg.org/) for format conversion, clipping, resampling, and channel mixing.

## Setup

```bash
./setup.sh
./test_env.sh
```

## Tools

### `process_audio`

Process an audio file using FFmpeg (clip, resample, convert format).

**Input:**
```json
{
  "input_path": "/path/to/input.wav",
  "output_path": "/path/to/output.wav",
  "start_time": 0,
  "duration": 3,
  "sample_rate": 16000,
  "channels": 1
}
```

**Output:**
```json
{
  "output_path": "/path/to/output.wav",
  "duration": 3.0,
  "sample_rate": 16000,
  "channels": 1
}
```

### `healthcheck`

Check if FFmpeg tools are available.

## Configuration

- `FFMPEG_PATH`: Path to ffmpeg binary (default: `ffmpeg`)
- `FFPROBE_PATH`: Path to ffprobe binary (default: `ffprobe`)
