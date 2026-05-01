#!/bin/bash
# Setup script for DiariZen speaker diarization tool.
#
# DiariZen is the one catalog tool that needs a different environment shape:
# it is pinned to Python 3.10 and the local pyannote-audio submodule. This
# host does not provide conda, so we keep the catalog's `.venv/bin/python`
# contract but create it with uv + Python 3.10.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

# Find uv - check persistent location first, then PATH
if [ -f "$REPO_ROOT/.uv/bin/uv" ]; then
    UV="$REPO_ROOT/.uv/bin/uv"
elif [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv" ]; then
    UV="/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Please install uv first."
    exit 1
fi

echo "Using uv: $UV"
echo "Using UV_CACHE_DIR: $UV_CACHE_DIR"
echo "uv version: $($UV --version)"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    chmod -R u+w .venv 2>/dev/null || true
    if ! rm -rf .venv 2>/dev/null; then
        backup_dir=".venv.stale.$(date +%s)"
        echo "Falling back to renaming busy environment to $backup_dir"
        mv .venv "$backup_dir"
        rm -rf "$backup_dir" >/dev/null 2>&1 &
    fi
fi

# Create virtual environment with Python 3.10 (NOT 3.11!)
echo "Creating virtual environment with Python 3.10..."
$UV venv --python 3.10

# Install PyTorch first. This host currently has no usable NVIDIA driver, so
# CPU wheels are the stable default for validation.
echo "Installing CPU PyTorch..."
$UV pip install --python .venv/bin/python torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Prepare DiariZen source. Prefer an already checked-out local copy from the
# sibling sure-eval workspace to avoid a fragile network clone.
if [ ! -d "diarizen_src" ]; then
    DEFAULT_SRC="/cpfs/user/jingpeng/workspace/sure-eval/src/sure_eval/models/diarizen/diarizen_src"
    DIARIZEN_SRC_PATH="${DIARIZEN_SRC_PATH:-$DEFAULT_SRC}"
    if [ -d "$DIARIZEN_SRC_PATH" ]; then
        echo "Copying DiariZen source from: $DIARIZEN_SRC_PATH"
        cp -a "$DIARIZEN_SRC_PATH" diarizen_src
    else
        echo "Cloning DiariZen source..."
        git clone --recursive https://github.com/BUTSpeechFIT/DiariZen.git diarizen_src
    fi
fi

# Install DiariZen and its required patched pyannote-audio submodule.
echo "Installing DiariZen from local source..."
cd diarizen_src
$UV pip install --python ../.venv/bin/python -r requirements.txt
$UV pip install --python ../.venv/bin/python -e .

echo "Installing patched pyannote-audio submodule..."
if [ ! -d "pyannote-audio" ]; then
    echo "Error: diarizen_src/pyannote-audio is missing. DiariZen requires the patched submodule."
    exit 1
fi
cd pyannote-audio
$UV pip install --python ../../.venv/bin/python -e .
cd ../..

# Lock NumPy version (critical for the patched pyannote stack).
echo "Locking NumPy to 1.26.4..."
$UV pip install --python .venv/bin/python numpy==1.26.4

# Install missing dependencies.
echo "Installing missing dependencies..."
$UV pip install --python .venv/bin/python psutil accelerate

# Install remaining dependencies from pyproject.toml
echo "Installing remaining dependencies..."
$UV pip install --python .venv/bin/python "huggingface-hub>=0.20.0"

# Install the local package (server code)
echo "Installing server package..."
$UV pip install --python .venv/bin/python -e . --no-deps

echo ""
echo "Setup complete!"
echo ""
echo "Python version: $(.venv/bin/python --version)"
echo ""
echo "Run ./test_env.sh to verify the installation."
