#!/bin/bash
# Setup script for ASR Qwen3 model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find uv - check persistent location first, then PATH
if [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv" ]; then
    UV="/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Please install uv first."
    exit 1
fi

echo "Using uv: $UV"
echo "uv version: $($UV --version)"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install PyTorch with CUDA (explicitly using .venv Python)
# echo "Installing PyTorch with CUDA support..."
# $UV pip install --python .venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install package (explicitly using .venv Python)
echo "Installing ASR Qwen3 package..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
