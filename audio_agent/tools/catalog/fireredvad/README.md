# FireRedVAD Tool

Voice Activity Detection (VAD) tool using FireRedVAD model for the Audio Agent Framework.

## Overview

This tool provides voice activity detection capabilities using the FireRedVAD model from FireRedTeam. It can detect speech segments in audio files and return timestamps.

## Model

- **Model ID**: FireRedTeam/FireRedVAD
- **Task**: Voice Activity Detection (VAD)
- **Weights Location**: `pretrained_models/FireRedVAD/`

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
    local_dir='pretrained_models/FireRedVAD',
    local_dir_use_symlinks=False
)
"
```

## Usage

### As MCP Tool

The tool exposes two MCP tools:

1. **fireredvad_predict**: Run VAD on an audio file
   ```json
   {
     "name": "fireredvad_predict",
     "arguments": {
       "audio_path": "/path/to/audio.wav"
     }
   }
   ```

2. **healthcheck**: Check if the model is ready
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
result = wrapper.predict("/path/to/audio.wav")
print(result.timestamps)  # [[start, end], ...]
```

## Output Format

The VAD output is JSON with the following structure:

```json
{
  "timestamps": [[0.5, 2.25], [3.0, 4.5]],
  "dur": 5.0,
  "wav_path": "/path/to/audio.wav"
}
```

- `timestamps`: List of [start, end] pairs in seconds
- `dur`: Total audio duration in seconds
- `wav_path`: Path to the input audio file

## Testing

Run the environment tests:

```bash
./test_env.sh
```

## Files

- `model.py`: ModelWrapper class for FireRedVAD
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
