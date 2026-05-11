#!/bin/bash
# Setup script for tempo-cnn tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ⭐ CRITICAL: Use persistent uv to survive server restarts
if [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh" ]; then
    source /lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh
fi

# Find uv - persistent location first, then PATH
if [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv" ]; then
    UV="/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Please ensure uv is installed at /lihaoyu/workspace/AUDIO_AGENT/.uv/"
    exit 1
fi

echo "Using uv: $UV"
echo "Using UV_PYTHON_INSTALL_DIR: $UV_PYTHON_INSTALL_DIR"
echo "Using UV_CACHE_DIR: $UV_CACHE_DIR"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install package
echo "Installing tempo-cnn tool package..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
