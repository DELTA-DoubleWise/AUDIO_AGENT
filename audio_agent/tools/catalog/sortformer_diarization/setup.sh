#!/bin/bash
# Setup script for SortFormer speaker diarization tool
# Follows SERVER_SPECIFIC_UV_SETUP.md for persistent uv usage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

# Optionally use a repo-local persistent uv install at $REPO_ROOT/.uv.
if [ -f "$REPO_ROOT/.uv/activate.sh" ]; then
    source "$REPO_ROOT/.uv/activate.sh"
fi

# Find uv: prefer a repo-local persistent install, else system uv on PATH.
if [ -f "$REPO_ROOT/.uv/bin/uv" ]; then
    UV="$REPO_ROOT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Install via 'curl -LsSf https://astral.sh/uv/install.sh | sh' or place a uv binary at $REPO_ROOT/.uv/bin/uv." >&2
    exit 1
fi

echo "Using uv: $UV"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install PyTorch with CUDA 12.1 (must be before NeMo)
# NOTE: NeMo 2.7+ requires torch>=2.5 for nn.Buffer; we pin 2.5.1+cu121
echo "Installing PyTorch with CUDA support..."
$UV pip install --python .venv/bin/python torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install ModelScope SDK for checkpoint download
echo "Installing ModelScope SDK..."
$UV pip install --python .venv/bin/python modelscope>=1.15.0

# Install NeMo ASR toolkit
echo "Installing NVIDIA NeMo ASR toolkit..."
$UV pip install --python .venv/bin/python "nemo_toolkit[asr]>=2.0.0"

# Install remaining dependencies from pyproject.toml
echo "Installing remaining dependencies..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
echo ""
echo "Run ./test_env.sh to verify the installation."
