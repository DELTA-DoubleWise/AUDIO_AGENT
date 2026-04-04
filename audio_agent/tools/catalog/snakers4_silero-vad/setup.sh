#!/bin/bash
# Setup script for snakers4_silero-vad tool

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

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install package
echo "Installing snakers4_silero-vad tool package..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
