#!/bin/bash
# Master setup script for all MCP tools
# Runs individual tool setup scripts with persistent uv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR/audio_agent/tools/catalog"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

echo "============================================================"
echo "Setting up all MCP tools with persistent uv"
echo "============================================================"
echo ""

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv command not found"
    exit 1
fi

echo "  uv location: $(which uv)"
echo "  uv version: $(uv --version)"
echo "  UV_CACHE_DIR: $UV_CACHE_DIR"
echo ""

# List of tools
TOOLS=("asr_qwen3" "diarizen" "ffmpeg" "fireredasr2s" "fireredvad" "librosa" "omni_captioner" "snakers4_silero-vad" "wespeaker" "whisperx")
TOTAL=${#TOOLS[@]}
CURRENT=0
FAILED=()

for tool in "${TOOLS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "============================================================"
    echo "[$CURRENT/$TOTAL] Setting up $tool..."
    echo "============================================================"
    
    TOOL_DIR="$TOOLS_DIR/$tool"
    
    if [ ! -d "$TOOL_DIR" ]; then
        echo "⚠ Tool directory not found: $TOOL_DIR"
        FAILED+=("$tool (directory not found)")
        continue
    fi
    
    if [ ! -f "$TOOL_DIR/setup.sh" ]; then
        echo "⚠ Setup script not found: $TOOL_DIR/setup.sh"
        FAILED+=("$tool (setup.sh not found)")
        continue
    fi
    
    cd "$TOOL_DIR"
    
    if ./setup.sh; then
        echo ""
        echo "✓ $tool setup completed successfully"
    else
        echo ""
        echo "✗ $tool setup failed"
        FAILED+=("$tool")
    fi
done

# Summary
echo ""
echo "============================================================"
echo "Setup Summary"
echo "============================================================"
echo ""

SUCCESS_COUNT=$((TOTAL - ${#FAILED[@]}))
echo "  Successful: $SUCCESS_COUNT/$TOTAL"
echo "  Failed: ${#FAILED[@]}/$TOTAL"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed tools:"
    for tool in "${FAILED[@]}"; do
        echo "  - $tool"
    done
    echo ""
    echo "To retry a specific tool:"
    echo "  cd AUDIO_AGENT/audio_agent/tools/catalog/<tool>"
    echo "  ./setup.sh"
    exit 1
else
    echo ""
    echo "✓ All tools set up successfully!"
    echo ""
    echo "To verify after server restart:"
    echo "  ./verify_all_tools.sh"
fi

echo ""
