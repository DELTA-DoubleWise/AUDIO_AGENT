#!/bin/bash
# MMAU Agent: Gemini 2.5 Pro Frontend + qwen3.5-plus Planner
# Usage: nohup bash run_mmau_agent.sh &

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load API keys
source .env 2>/dev/null || true

OUTPUT_DIR="$SCRIPT_DIR/test_result/mmau_gemini_2.5pro_qwen3.5plus"
LOG_FILE="$OUTPUT_DIR/agent.log"
mkdir -p "$OUTPUT_DIR"

exec > "$LOG_FILE" 2>&1

echo "========================================"
echo "MMAU Agent started at $(date)"
echo "========================================"

.venv/bin/python -m benchmarking.mmau.run_gemini_parallel_multigpu \
  --num-samples 1000 \
  --workers 4 \
  --gpus 2 \
  --planner-model qwen3.5-plus \
  --planner-backend openai \
  --output-dir "$OUTPUT_DIR" \
  --dataset-dir /cpfs/user/jingpeng/workspace/nfs/data/test/MMAU \
  --checkpoint-interval 5 \
  --disable-run-logging

echo "========================================"
echo "MMAU Agent finished at $(date)"
echo "========================================"
