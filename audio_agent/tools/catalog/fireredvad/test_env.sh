#!/bin/bash
# Quick environment test wrapper for FireRedVAD

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python interpreter not found at $PYTHON_EXE"
    echo "Please run setup.sh first."
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
echo "Python version: $($PYTHON_EXE --version)"
echo ""

# Test key imports
echo "Testing imports..."
$PYTHON_EXE -c "import fireredvad; print('✓ fireredvad imported successfully')" || exit 1
$PYTHON_EXE -c "import torch; print(f'✓ torch {torch.__version__} imported successfully')" || exit 1
$PYTHON_EXE -c "import numpy; print(f'✓ numpy {numpy.__version__} imported successfully')" || exit 1
$PYTHON_EXE -c "import soundfile; print('✓ soundfile imported successfully')" || exit 1
echo ""

# Test wrapper imports
echo "Testing wrapper imports..."
$PYTHON_EXE -c "from model import ModelWrapper, VADResult, AEDResult; print('✓ ModelWrapper, VADResult, AEDResult imported successfully')" || exit 1
echo ""

# Generate a tiny synthetic WAV so this test is portable across machines.
# Override by exporting AUDIO_AGENT_TEST_WAV to an existing file.
TEST_WAV="${AUDIO_AGENT_TEST_WAV:-}"
if [ -z "$TEST_WAV" ] || [ ! -f "$TEST_WAV" ]; then
    TEST_WAV="$(mktemp -t fireredvad_test_XXXXXX.wav)"
    $PYTHON_EXE -c "
import numpy as np, soundfile as sf
sr = 16000
# Two seconds: silence + 500ms 440Hz tone + silence, gives VAD/AED something to look at.
t = np.linspace(0, 0.5, int(sr*0.5), endpoint=False)
tone = 0.3 * np.sin(2*np.pi*440*t).astype('float32')
audio = np.concatenate([np.zeros(int(sr*0.5), 'float32'), tone, np.zeros(int(sr*1.0), 'float32')])
sf.write('$TEST_WAV', audio, sr)
print(f'Wrote synthetic WAV: $TEST_WAV')
" || exit 1
fi
export TEST_WAV

# Test VAD
echo "Testing VAD..."
$PYTHON_EXE -c "
import os
from model import ModelWrapper
wrapper = ModelWrapper()
result = wrapper.predict(os.environ['TEST_WAV'])
print(f'✓ VAD prediction successful: {len(result.timestamps)} speech segments')
" || exit 1
echo ""

# Test AED
echo "Testing AED..."
$PYTHON_EXE -c "
import os
from model import ModelWrapper
wrapper = ModelWrapper()
result = wrapper.predict_aed(os.environ['TEST_WAV'])
events = [k for k, v in result.event2timestamps.items() if v]
print(f'✓ AED prediction successful: detected events - {events}')
" || exit 1
echo ""

# Test MCP server initialization
echo "Testing MCP server..."
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize"}' | $PYTHON_EXE server.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ MCP server initialization works"
else
    echo "✗ MCP server initialization failed"
    exit 1
fi
echo ""

echo "========================================"
echo "All environment tests passed!"
echo "========================================"
