#!/bin/bash
# Master setup script for all MCP tools
# Runs individual tool setup scripts with persistent uv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR/AUDIO_AGENT/audio_agent/tools/catalog"

echo "============================================================"
echo "Setting up all MCP tools with persistent uv"
echo "============================================================"
echo ""

# Activate persistent uv first
echo "Activating persistent uv..."
if [ ! -f "$SCRIPT_DIR/.uv/activate.sh" ]; then
    echo "Error: Persistent uv activation script not found at $SCRIPT_DIR/.uv/activate.sh"
    exit 1
fi

source "$SCRIPT_DIR/.uv/activate.sh"

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv command not found after activation"
    exit 1
fi

echo "  uv location: $(which uv)"
echo "  uv version: $(uv --version)"
echo ""

# List of tools
TOOLS=("asr_qwen3" "diarizen" "ffmpeg" "librosa" "omni_captioner" "snakers4_silero-vad")
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
    echo "  source /lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh"
    echo "  ./verify_all_tools.sh"
fi

echo ""
