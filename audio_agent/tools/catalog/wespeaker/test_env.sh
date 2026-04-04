#!/bin/bash
# Test script for WeSpeaker tool environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== WeSpeaker Tool Environment Test ==="
echo ""

# Check Python
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

echo "✓ Virtual environment exists"
echo "  Python: $(.venv/bin/python --version)"
echo ""

# Test imports
echo "Testing imports..."
if .venv/bin/python -c "import wespeaker" 2>/dev/null; then
    echo "✓ wespeaker imported successfully"
else
    echo "❌ Failed to import wespeaker"
    exit 1
fi

if .venv/bin/python -c "import torch" 2>/dev/null; then
    echo "✓ torch imported successfully"
else
    echo "❌ Failed to import torch"
    exit 1
fi

if .venv/bin/python -c "import soundfile" 2>/dev/null; then
    echo "✓ soundfile imported successfully"
else
    echo "❌ Failed to import soundfile"
    exit 1
fi

echo ""

# Test model wrapper import
echo "Testing model wrapper..."
if .venv/bin/python -c "from model import ModelWrapper, SpeakerVerificationResult" 2>/dev/null; then
    echo "✓ Model wrapper imported successfully"
else
    echo "❌ Failed to import model wrapper"
    exit 1
fi

echo ""

# Test server import
echo "Testing server..."
if .venv/bin/python -c "from server import WeSpeakerServer" 2>/dev/null; then
    echo "✓ Server imported successfully"
else
    echo "❌ Failed to import server"
    exit 1
fi

echo ""
echo "=== All tests passed! ==="
