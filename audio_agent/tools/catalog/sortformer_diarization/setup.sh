#!/bin/bash
# Setup script for SortFormer speaker diarization tool
# Follows SERVER_SPECIFIC_UV_SETUP.md for persistent uv usage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ⭐ CRITICAL: Activate persistent uv first
if [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh" ]; then
    source /lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh
    echo "Activated persistent uv"
else
    echo "Warning: Persistent uv activation script not found at /lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh"
fi

# Check uv availability
if [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv" ]; then
    UV="/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Please ensure uv is installed."
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
