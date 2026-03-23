# Default Environment Setup

This document explains how to set up the **minimal/default environment** for the Audio Agent Framework. This environment includes only the core dependencies required to use the framework's architecture and dummy components. It does **not** include model-specific dependencies for running real LLMs like Qwen2-Audio or Qwen2.5.

## What This Environment Includes

The default environment provides:

- **Core framework dependencies**: LangGraph workflow engine, LangChain core, Pydantic v2
- **Development tools**: pytest, black, ruff, mypy (optional dev dependencies)
- **Dummy components**: Fully functional agent with mock frontend, planner, and tools

## What This Environment Does NOT Include

Model-specific packages required for real inference:

- PyTorch (`torch`, `torchvision`, `torchaudio`)
- Hugging Face Transformers
- Audio processing libraries (`librosa`, `soundfile`)
- Model acceleration libraries (`accelerate`)
- Tokenizer libraries (`sentencepiece`)

> **Note**: To run the demo with real models (Qwen2-Audio-7B-Instruct and Qwen2.5-7B-Instruct), see [DEMO_ENVIRONMENT.md](./DEMO_ENVIRONMENT.md).

## Prerequisites

- Python 3.11 or higher
- pip 21.0+ or conda 4.10+

## Installation Options

### Option 1: Using pip (Recommended for simplicity)

```bash
# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install the package with core dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
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

# Install the package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Verifying the Installation

Run the basic smoke test with dummy components:

```python
from audio_agent.main import create_dummy_agent

# Create agent with dummy components (no GPU required)
agent = create_dummy_agent()

# Run a test query
result = agent.run(
    question="What is being discussed in this audio?",
    audio_path_or_uri="/fake/audio.wav",  # Fake path is fine for dummy mode
)

# Check result
if agent.is_successful(result):
    print("Success! Agent is working correctly.")
    print(f"Answer: {agent.get_answer(result).answer}")
else:
    print("Agent did not produce a successful answer.")
```

Or run the unit tests:

```bash
pytest audio_agent/tests/ -v
```

## Core Dependencies (from pyproject.toml)

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | >=0.2.0 | Workflow orchestration and state management |
| `langchain-core` | >=0.3.0 | Core LangChain abstractions |
| `pydantic` | >=2.0.0 | Data validation and settings management |

### Development Dependencies (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=8.0.0 | Testing framework |
| `pytest-cov` | >=4.0.0 | Test coverage reporting |
| `black` | >=24.0.0 | Code formatting |
| `ruff` | >=0.1.0 | Linting and import sorting |
| `mypy` | >=1.0.0 | Static type checking |

## Next Steps

- To run with real audio models, set up the [Demo Environment](./DEMO_ENVIRONMENT.md)
- To implement custom frontends/planners, refer to [README.md](./README.md)

## Troubleshooting

### `ModuleNotFoundError: No module named 'audio_agent'`

Ensure you installed the package in editable mode:

```bash
pip install -e .
```

### Tests fail with import errors

Install development dependencies:

```bash
pip install -e ".[dev]"
```

### Python version errors

Verify Python version:

```bash
python --version  # Should be 3.11 or higher
```
