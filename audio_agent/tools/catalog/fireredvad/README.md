# FireRedVAD Tool

Voice Activity Detection (VAD) and Audio Event Detection (AED) tool using FireRedVAD model for the Audio Agent Framework.

## Overview

This tool provides both VAD and AED capabilities using the FireRedVAD model from FireRedTeam:

- **VAD** (Voice Activity Detection): Detects speech segments in audio files
- **AED** (Audio Event Detection): Detects speech, singing, and music events in audio files

## Models

- **Model ID**: FireRedTeam/FireRedVAD
- **VAD Weights**: `/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/models/fireredvad/VAD/`
- **AED Weights**: `/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/models/fireredvad/AED/`

## Setup

### Prerequisites

- Python 3.11+
- uv (persistent installation at `/lihaoyu/workspace/AUDIO_AGENT/.uv/`)

### Installation

```bash
# Source the persistent uv activation
source /lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh

# Run setup
cd /lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/audio_agent/tools/catalog/fireredvad
./setup.sh
```

### Download Model Weights

The model weights are automatically downloaded during setup. If you need to manually download:

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='FireRedTeam/FireRedVAD',
    local_dir='/lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/models/fireredvad',
    local_dir_use_symlinks=False
)
"
```

## Usage

### As MCP Tool

The tool exposes three MCP tools:

1. **fireredvad_predict**: Run VAD on an audio file
   ```json
   {
     "name": "fireredvad_predict",
     "arguments": {
       "audio_path": "/path/to/audio.wav"
     }
   }
   ```

2. **fireredvad_aed**: Run AED on an audio file
   ```json
   {
     "name": "fireredvad_aed",
     "arguments": {
       "audio_path": "/path/to/audio.wav"
     }
   }
   ```

3. **healthcheck**: Check if the models are ready
   ```json
   {
     "name": "healthcheck",
     "arguments": {}
   }
   ```

### Direct Usage

```python
from model import ModelWrapper

wrapper = ModelWrapper()

# VAD - Voice Activity Detection
vad_result = wrapper.predict("/path/to/audio.wav")
print(vad_result.timestamps)  # [[start, end], ...]

# AED - Audio Event Detection
aed_result = wrapper.predict_aed("/path/to/audio.wav")
print(aed_result.event2timestamps)  # {'speech': [...], 'singing': [...], 'music': [...]}
print(aed_result.event2ratio)       # {'speech': 0.5, 'singing': 0.0, 'music': 0.0}
```

## Output Formats

### VAD Output

```json
{
  "timestamps": [[0.5, 2.25]],
  "dur": 2.35,
  "wav_path": "/path/to/audio.wav"
}
```

- `timestamps`: List of [start, end] pairs in seconds for speech segments
- `dur`: Total audio duration in seconds
- `wav_path`: Path to the input audio file

### AED Output

```json
{
  "event2timestamps": {
    "speech": [[0.49, 2.28]],
    "singing": [],
    "music": []
  },
  "event2ratio": {
    "speech": 0.661,
    "singing": 0.0,
    "music": 0.0
  },
  "dur": 2.35,
  "wav_path": "/path/to/audio.wav"
}
```

- `event2timestamps`: Dict mapping event types to lists of [start, end] timestamps
  - `speech`: Speech segments
  - `singing`: Singing segments
  - `music`: Music segments
- `event2ratio`: Dict mapping event types to their duration ratios (0.0 to 1.0)
- `dur`: Total audio duration in seconds
- `wav_path`: Path to the input audio file

## Testing

Run the environment tests:

```bash
./test_env.sh
```

## Files

- `model.py`: ModelWrapper class with VAD and AED support
- `server.py`: MCP server implementation
- `config.yaml`: MCP tool configuration
- `model.spec.yaml`: Tool specification
- `setup.sh`: Environment setup script
- `test_env.sh`: Environment verification script
- `artifacts/`: Build and validation artifacts

## Artifacts

Per the tool preparation workflow, the following artifacts are maintained:

- `backend_choice.json`: UV backend selection
- `build_plan.json`: Build steps
- `build.log`: Build output
- `validation.log`: Test results
- `verdict.json`: Final assessment
- `artifact_manifest.json`: Artifact inventory
