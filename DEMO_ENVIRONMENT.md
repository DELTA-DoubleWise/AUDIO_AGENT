# Demo Environment Setup

This document explains how to build an environment for running:

- `audio_agent/examples/demo_run.py`
- `Qwen2AudioFrontend` with `Qwen/Qwen2-Audio-7B-Instruct`
- `Qwen25Planner` with `Qwen/Qwen2.5-7B-Instruct`

## What the demo needs

The current demo uses:

- local Hugging Face model loading
- GPU-oriented `device_map="auto"` execution
- `Qwen2-Audio-7B-Instruct` for frontend audio captioning
- `Qwen2.5-7B-Instruct` for text planning
- the repo's built-in dummy tools for tool execution

Important implications:

- the demo is not a lightweight CPU example
- `Qwen2-Audio-7B-Instruct` support requires a recent `transformers` build
- you should plan for a Linux GPU environment

## Recommended environment

Recommended baseline:

- OS: Linux
- Python: `3.11`
- GPU: NVIDIA CUDA-capable GPU
- CUDA: match your PyTorch install
- RAM: at least `32 GB`
- Disk: at least `40-60 GB` free for models, cache, and environment

Practical note:

- both models are 7B-scale
- running them together in one process is GPU-memory heavy
- if VRAM is limited, start with a larger GPU machine or be prepared to adjust model loading manually later

This repo does not currently provide quantized demo adapters, CPU-friendly settings, or model offload tuning in `demo_run.py`.

## Why `transformers` from source is recommended

The `Qwen2-Audio-7B-Instruct` model card advises building `transformers` from source, otherwise you may hit:

- `KeyError: 'qwen2-audio'`

Source:

- Hugging Face model card: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct

For `Qwen2.5-7B-Instruct`, the model card advises using the latest `transformers`.

Source:

- Hugging Face model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## Step 1: Create a virtual environment

From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Step 2: Install PyTorch first

Install PyTorch using the official selector for your machine:

- https://pytorch.org/get-started/locally/

Example for Linux + pip + CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If your cluster uses a different CUDA version, use the matching command from PyTorch docs instead.

## Step 3: Install Hugging Face and audio dependencies

Install `transformers` from source because of Qwen2-Audio:

```bash
pip install git+https://github.com/huggingface/transformers
```

Then install the remaining runtime packages used by the demo:

```bash
pip install accelerate librosa soundfile sentencepiece
```

Why these packages:

- `accelerate`: commonly required when using `device_map="auto"`
- `librosa`: used by `Qwen2AudioFrontend` to load local and remote audio
- `soundfile`: backend commonly used during audio decoding
- `sentencepiece`: useful for tokenizer/model compatibility in HF environments

## Step 4: Install this repo

The repo's base dependencies in `pyproject.toml` do not include the Qwen demo stack, so install both the project and the extra runtime packages above.

From the repo root:

```bash
pip install -e .
```

If you also want test tooling:

```bash
pip install -e ".[dev]"
```

## Step 5: Optional Hugging Face auth/cache setup

If your environment requires authenticated model pulls or you want predictable cache paths:

```bash
export HF_HOME=$PWD/.hf_cache
```

If needed:

```bash
huggingface-cli login
```

## Step 6: Sanity checks

Check imports first:

```bash
python - <<'PY'
from audio_agent.frontend.qwen2_audio_frontend import Qwen2AudioFrontend
from audio_agent.planner.qwen25_planner import Qwen25Planner
print("imports_ok")
PY
```

Optional syntax check for the demo:

```bash
python -m py_compile audio_agent/examples/demo_run.py
```

## Step 7: Run the demo

Example:

```bash
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?" \
  --frontend-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --planner-model-path Qwen/Qwen2.5-7B-Instruct \
  --max-steps 5
```

You can also pass an HTTP(S) audio URL to `--audio`, because `Qwen2AudioFrontend` supports both:

- local file paths
- remote URLs

## Common failure modes

### `ModuleNotFoundError: No module named 'transformers'`

Install the HF stack:

```bash
pip install git+https://github.com/huggingface/transformers
```

### `KeyError: 'qwen2-audio'`

Your `transformers` build is too old for `Qwen2-Audio`.

Fix:

```bash
pip install --upgrade git+https://github.com/huggingface/transformers
```

### `ImportError` or runtime errors around `device_map="auto"`

Install `accelerate`:

```bash
pip install accelerate
```

### Audio loading fails

Likely causes:

- invalid local path
- remote URL inaccessible from the machine
- unsupported or corrupted audio file
- missing audio decoding backend

Try:

```bash
pip install librosa soundfile
```

and verify the audio file independently.

### Out-of-memory during model load or generation

This is the most likely operational issue.

Current demo limitations:

- both models are loaded in one process
- the demo does not implement quantization or manual offload settings
- the demo keeps default `device_map="auto"` behavior

If you hit OOM, use a larger GPU environment first. If that is not possible, the code will need a separate change to support lower-memory loading strategies.

## Minimal install command summary

If you already have the right CUDA/PyTorch installed, the shortest path is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install git+https://github.com/huggingface/transformers
pip install accelerate librosa soundfile sentencepiece
pip install -e .
```

## References

- PyTorch local install guide: https://pytorch.org/get-started/locally/
- Qwen2-Audio model card: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
- Qwen2.5-7B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
