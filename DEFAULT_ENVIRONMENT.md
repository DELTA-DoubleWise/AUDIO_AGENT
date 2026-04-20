# Default Environment Setup

This document explains how to set up the **default environment** for the Audio Agent Framework. This environment includes the core dependencies and everything needed to run the framework with **API-based models** (e.g., via DashScope, OpenAI, or any OpenAI-compatible endpoint). It does **not** include local model-specific dependencies for running on-premise LLMs like Qwen2-Audio or Qwen2.5.

## What This Environment Includes

The default environment provides:

- **Core framework dependencies**: LangGraph workflow engine, LangChain core, Pydantic v2
- **API model support**: `openai` client for OpenAI-compatible frontends and planners
- **Development tools**: pytest, black, ruff, mypy (optional dev dependencies)
- **Dummy components**: Fully functional agent with mock frontend, planner, and tools for testing

## What This Environment Does NOT Include

Local model-specific packages required for on-premise inference (not needed for API-only usage):

- PyTorch (`torch`, `torchvision`, `torchaudio`)
- Hugging Face Transformers
- Audio processing libraries (`librosa`, `soundfile`)
- Model acceleration libraries (`accelerate`)
- Tokenizer libraries (`sentencepiece`)

> **Note**: In this environment, only **API models** are supported for real inference. No local Qwen or other on-premise model environment needs to be installed. If you need local GPU inference, see [DEMO_ENVIRONMENT.md](./DEMO_ENVIRONMENT.md).

## Prerequisites

- Python 3.11 or higher
- pip 21.0+ or conda 4.10+
- An API key for an OpenAI-compatible service (e.g., DashScope, OpenAI)

## Installation Options

### Option 1: Using pip (Recommended for simplicity)

```bash
# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install the package with core + API dependencies
pip install -e ".[api]"

# Or with development dependencies
pip install -e ".[api,dev]"
```

### Option 2: Using conda

> **Important**: On this system, conda requires initialization before use:
> ```bash
> source /lihaoyu/.conda.path.sh
> ```

```bash
# Initialize conda (if not already done)
source /lihaoyu/.conda.path.sh

# Create conda environment with Python 3.11
conda create -n audio_agent python=3.11
conda activate audio_agent

# Install the package with API support
pip install -e ".[api]"

# Or with development dependencies
pip install -e ".[api,dev]"
```

## API Configuration

To use real models in this environment, set your API key:

```bash
# For DashScope (default for API frontends/planners)
export DASHSCOPE_API_KEY="sk-xxx"

# Or for OpenAI / other providers
export OPENAI_API_KEY="sk-xxx"
```

### Supported API Components

- **Frontend**: `OpenAICompatibleFrontend` — audio understanding via API (tested with `qwen3-omni-flash` on DashScope)
- **Planner**: `OpenAICompatiblePlanner` — reasoning and decision making via API (tested with `qwen3.5-plus`, `kimi-k2.5`)

No local GPU or model downloads are required.

## Verifying the Installation

### 1. Dummy smoke test (no API key needed)

```python
from audio_agent.main import create_dummy_agent

agent = create_dummy_agent()
result = agent.run(
    question="What is being discussed in this audio?",
    audio_path_or_uri="/fake/audio.wav",
)

if agent.is_successful(result):
    print("Success! Agent is working correctly.")
    print(f"Answer: {agent.get_answer(result).answer}")
else:
    print("Agent did not produce a successful answer.")
```

### 2. API-based demo (requires API key)

```bash
export DASHSCOPE_API_KEY="sk-xxx"

python -m audio_agent.examples.demo_run_api_full \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?"
```

This uses:
- `OpenAICompatibleFrontend` with `qwen3-omni-flash`
- `OpenAICompatiblePlanner` with `qwen3.5-plus`

### 3. Run unit tests

```bash
pytest audio_agent/tests/ -v
```

## Core Dependencies (from pyproject.toml)

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | >=0.2.0 | Workflow orchestration and state management |
| `langchain-core` | >=0.3.0 | Core LangChain abstractions |
| `pydantic` | >=2.0.0 | Data validation and settings management |

### API Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | >=1.0.0 | OpenAI-compatible API client for frontend and planner |

### Development Dependencies (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=8.0.0 | Testing framework |
| `pytest-cov` | >=4.0.0 | Test coverage reporting |
| `black` | >=24.0.0 | Code formatting |
| `ruff` | >=0.1.0 | Linting and import sorting |
| `mypy` | >=1.0.0 | Static type checking |

## Next Steps

- To implement custom API frontends/planners, refer to [README.md](./README.md)
- To run with automatic MCP tool discovery (API planner + local/auto tools), try `demo_run_auto_tools.py`
- For local GPU inference with Qwen models, see [DEMO_ENVIRONMENT.md](./DEMO_ENVIRONMENT.md)

## Troubleshooting

### `ModuleNotFoundError: No module named 'audio_agent'`

Ensure you installed the package in editable mode:

```bash
pip install -e ".[api]"
```

### `Missing openai package. Install with: pip install openai`

Install the API extras:

```bash
pip install -e ".[api]"
```

### API call fails with authentication error

Make sure your API key is exported:

```bash
export DASHSCOPE_API_KEY="sk-xxx"
# or
export OPENAI_API_KEY="sk-xxx"
```

### Tests fail with import errors

Install development dependencies:

```bash
pip install -e ".[api,dev]"
```

### Python version errors

Verify Python version:

```bash
python --version  # Should be 3.11 or higher
```
