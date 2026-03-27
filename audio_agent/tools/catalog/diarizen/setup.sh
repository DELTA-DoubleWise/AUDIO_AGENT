#!/bin/bash
# Setup script for DiariZen speaker diarization tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda
source /lihaoyu/.conda.path.sh

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.10 (NOT 3.11!)
echo "Creating virtual environment with Python 3.10..."
conda create --prefix ./.venv python=3.10 -y

# Install PyTorch with CUDA (Step 2 - MUST be before DiariZen!)
echo "Installing PyTorch with CUDA support..."
.venv/bin/pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install DiariZen from submodule (Step 3)
echo "Installing DiariZen from submodule..."
git clone https://github.com/BUTSpeechFIT/DiariZen.git
mv DiariZen diarizen_src
cd diarizen_src
../.venv/bin/pip install -r requirements.txt && ../.venv/bin/pip install -e .

# Install pyannote-audio from submodule (Step 4 - CRITICAL!)
echo "Installing pyannote-audio from submodule..."
cd pyannote-audio
../../.venv/bin/pip install -e .
cd ../..

# Lock NumPy version (Step 5 - CRITICAL!)
echo "Locking NumPy to 1.26.4..."
.venv/bin/pip install numpy==1.26.4

# Install missing dependencies (Step 6)
echo "Installing missing dependencies..."
.venv/bin/pip install psutil accelerate

# Install remaining dependencies from pyproject.toml
echo "Installing remaining dependencies..."
.venv/bin/pip install huggingface-hub>=0.20.0

# Install the local package (server code)
echo "Installing server package..."
.venv/bin/pip install -e . --no-deps

echo ""
echo "Setup complete!"
echo ""
echo "Python version: $(.venv/bin/python --version)"
echo ""
echo "Run ./test_env.sh to verify the installation."
