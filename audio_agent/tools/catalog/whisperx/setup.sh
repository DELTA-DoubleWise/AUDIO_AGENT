#!/bin/bash
# Setup script for WhisperX tool

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
    echo "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Using uv: $UV"
echo "uv version: $($UV --version)"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install package (explicitly using .venv Python)
echo "Installing WhisperX package..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
echo ""
echo "To verify the installation, run: ./test_env.sh"
