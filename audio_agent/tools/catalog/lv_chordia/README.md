# lv-chordia Tool

Large-vocabulary chord transcription tool for the Audio Agent Framework.

## Overview

This tool wraps [lv-chordia](https://pypi.org/project/lv-chordia/), an implementation of the ISMIR 2019 paper "Large-Vocabulary Chord Transcription via Chord Structure Decomposition" by Jiang et al.

## Features

- **Large Vocabulary**: Supports hundreds of chord types including complex jazz chords
- **Ensemble Model**: 5 pre-trained networks for robust predictions
- **Multiple Dictionaries**:
  - `submission` (~170 chords, default): Best balance for general use
  - `ismir2017` (~25 chords): MIREX/ISMIR2017 standard
  - `full` (~600+ chords): Complete vocabulary for jazz analysis
- **JAMS Format Output**: Standard chord labels like `C:maj`, `F:maj7`, `G:min7`, `A:dim`

## Setup

```bash
# Optional: source <repo>/.uv/activate.sh for repo-local uv
./setup.sh
```

## Test

```bash
./test_env.sh
```

## Usage

### Python API

```python
from model import ModelWrapper

model = ModelWrapper()
result = model.predict("audio.wav", chord_dict_name="submission")

for seg in result.segments:
    print(f"{seg.start_time:.3f}-{seg.end_time:.3f}: {seg.chord}")
```

### MCP Server

The tool runs as an MCP server via `server.py`.

## Output Format

```json
{
  "segments": [
    {"start_time": 0.0, "end_time": 2.5, "chord": "C:maj"},
    {"start_time": 2.5, "end_time": 5.0, "chord": "F:maj7"},
    {"start_time": 5.0, "end_time": 7.5, "chord": "G:maj"}
  ],
  "duration": 10.0
}
```

## Model Storage

Pre-trained models are bundled with the `lv-chordia` pip package (~27MB). No separate model download is required.
