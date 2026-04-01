# Demo Environment Setup

This document explains how to build an environment for running the **full demo** with real models:

- `audio_agent/examples/demo_run.py`
- `Qwen2AudioFrontend` with `Qwen/Qwen2-Audio-7B-Instruct`
- `Qwen25Planner` with `Qwen/Qwen2.5-7B-Instruct`

## What the Demo Needs

The current demo uses:

- Local Hugging Face model loading
- GPU-oriented `device_map="auto"` execution
- `Qwen2-Audio-7B-Instruct` for frontend audio captioning
- `Qwen2.5-7B-Instruct` for text planning
- The repo's built-in dummy tools for tool execution

**Important implications:**

- The demo is **not** a lightweight CPU example
- `Qwen2-Audio-7B-Instruct` support requires a recent `transformers` build from source
- You should plan for a Linux GPU environment

## System Requirements

### Recommended Baseline

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| OS | Linux | Linux |
| Python | 3.11 | 3.11 |
| GPU | NVIDIA CUDA-capable | NVIDIA A100 or similar |
| CUDA | 12.1+ | 12.1+ |
| RAM | 32 GB | 64 GB |
| Disk | 40 GB | 60 GB+ |
| VRAM | 24 GB | 40 GB+ |

**Practical notes:**

- Both models are 7B-scale and loaded in one process
- Running them together is GPU-memory intensive
- If VRAM is limited, consider using quantized models or adjusting the code for sequential loading

## Why `transformers` from Source is Required

The `Qwen2-Audio-7B-Instruct` model card explicitly recommends building `transformers` from source. Without it, you will hit:

```
KeyError: 'qwen2-audio'
```

**Sources:**
- [Qwen2-Audio-7B-Instruct Model Card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- [Qwen2.5-7B-Instruct Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## Installation Steps

### Step 1: Initialize Conda (if using conda)

> **Important**: On this system, conda requires initialization before use:
> ```bash
> source /lihaoyu/.conda.path.sh
> ```

### Step 2: Create Environment

**Using conda (recommended for GPU environments):**

```bash
# Initialize conda
source /lihaoyu/.conda.path.sh

# Create environment from provided environment.yml
conda env create -f environment.yml
conda activate audio_agent_demo
```

**Using pip with virtual environment:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch with CUDA

Install PyTorch using the official selector for your machine:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

**Example for Linux + pip + CUDA 12.1:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Example for conda + CUDA 12.1:**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

If your cluster uses a different CUDA version, use the matching command from PyTorch docs.

### Step 4: Install Hugging Face and Audio Dependencies

Install `transformers` from source (required for Qwen2-Audio):

```bash
pip install git+https://github.com/huggingface/transformers
```

Then install the remaining runtime packages:

```bash
pip install accelerate librosa soundfile sentencepiece
```

**Package purposes:**

| Package | Purpose |
|---------|---------|
| `accelerate` | Required for `device_map="auto"` and model distribution |
| `librosa` | Audio loading and preprocessing for Qwen2AudioFrontend |
| `soundfile` | Audio decoding backend (required by librosa and transformers) |
| `sentencepiece` | Tokenizer compatibility for Qwen models |

### Step 5: Install This Package

From the repo root:

```bash
# Install the base package
pip install -e .

# With development dependencies (for testing)
pip install -e ".[dev]"
```

### Step 6: Optional Hugging Face Auth/Cache Setup

If your environment requires authenticated model pulls or you want predictable cache paths:

```bash
# Set custom cache directory
export HF_HOME=$PWD/.hf_cache

# Login if needed (for gated models or rate limits)
huggingface-cli login
```

## Verification

### Import Check

```bash
python - <<'PY'
from audio_agent.frontend.qwen2_audio_frontend import Qwen2AudioFrontend
from audio_agent.planner.qwen25_planner import Qwen25Planner
print("All imports successful!")
PY
```

### Syntax Check

```bash
python -m py_compile audio_agent/examples/demo_run.py
```

### GPU Check

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

## Running the Demo

```bash
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?" \
  --frontend-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --planner-model-path Qwen/Qwen2.5-7B-Instruct \
  --max-steps 5
```

**Notes:**
- You can pass an HTTP(S) URL to `--audio` (Qwen2AudioFrontend supports both local paths and remote URLs)
- First run will download models (several GB) to the Hugging Face cache

## Common Failure Modes

### `ModuleNotFoundError: No module named 'transformers'`

**Fix:**
```bash
pip install git+https://github.com/huggingface/transformers
```

### `KeyError: 'qwen2-audio'`

Your `transformers` build is too old for `Qwen2-Audio`.

**Fix:**
```bash
pip install --upgrade git+https://github.com/huggingface/transformers
```

### `ImportError` or runtime errors around `device_map="auto"`

**Fix:**
```bash
pip install accelerate
```

### Audio loading fails

**Likely causes:**
- Invalid local path
- Remote URL inaccessible
- Unsupported or corrupted audio file
- Missing audio decoding backend

**Fix:**
```bash
pip install librosa soundfile
```

Verify the audio file independently with:
```python
import librosa
waveform, sr = librosa.load("/path/to/audio.wav", sr=16000)
print(f"Loaded: {len(waveform)} samples at {sr} Hz")
```

### Out-of-Memory (OOM) during model load or generation

This is the most likely operational issue.

**Current demo limitations:**
- Both models are loaded in one process
- No quantization or manual offload settings
- Default `device_map="auto"` behavior

**Solutions:**
1. Use a GPU with more VRAM (40GB+ recommended)
2. Modify the code to load models sequentially (not simultaneously)
3. Use 4-bit or 8-bit quantization (requires code changes)
4. Use `accelerate`'s offloading features

## Minimal Install Command Summary

If you already have the right CUDA/PyTorch installed:

```bash
# Using pip virtualenv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/transformers
pip install accelerate librosa soundfile sentencepiece
pip install -e .
```

```bash
# Using conda
source /lihaoyu/.conda.path.sh
conda env create -f environment.yml
conda activate audio_agent_demo
```

## Qwen3-Omni Support (Optional)

The framework also includes a frontend for `Qwen3-Omni-30B-A3B-Instruct`. This requires additional dependencies:

```bash
# Install qwen_omni_utils from source
pip install git+https://github.com/QwenLM/Qwen3-Omni.git
```

**Note:** Qwen3-Omni is a 30B parameter model and requires significantly more resources than the 7B models used in the default demo.

## MCP Tool Models (Optional)

The framework includes MCP-based tools that require additional models:

```bash
# Download ASR tool model (Qwen3-ASR-1.7B, ~4GB)
audio-agent-download-models --models qwen3-asr

# Download diarization model (DiariZen, ~1GB)
audio-agent-download-models --models diarizen

# Download all tool models
audio-agent-download-models --models qwen3-asr qwen3-aligner diarizen omni-captioner
```

Before using MCP tools, you must set up their environments:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup tool environments
cd audio_agent/tools/catalog/asr_qwen3 && ./setup.sh && cd -
cd audio_agent/tools/catalog/diarizen && ./setup.sh && cd -

# Verify all tools
./verify_all_tools.sh
```

## References

- [PyTorch Local Install Guide](https://pytorch.org/get-started/locally/)
- [Qwen2-Audio-7B-Instruct Model Card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- [Qwen2.5-7B-Instruct Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)
