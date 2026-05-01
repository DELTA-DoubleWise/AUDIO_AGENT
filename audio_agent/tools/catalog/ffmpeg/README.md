# FFmpeg Audio Processing Tool

Audio processing tool using [FFmpeg](https://ffmpeg.org/) for format conversion, clipping, resampling, and channel mixing.

## Setup

```bash
./setup.sh
./test_env.sh
```

## Tools

This server exposes specific FFmpeg operations as separate tools, such as
`trim_audio`, `resample_audio`, `convert_channels`, `loudnorm`, denoising,
filtering, and analysis utilities.

### `healthcheck`

Check if FFmpeg tools are available.

## Configuration

- `FFMPEG_PATH`: Path to ffmpeg binary (default: `ffmpeg`)
- `FFPROBE_PATH`: Path to ffprobe binary (default: `ffprobe`)
