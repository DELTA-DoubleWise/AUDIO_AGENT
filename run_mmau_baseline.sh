#!/bin/bash
# MMAU Baseline: Gemini 2.5 Pro Direct (No Agent Flow)
# Usage: nohup bash run_mmau_baseline.sh &

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load API keys
source .env 2>/dev/null || true

OUTPUT_DIR="$SCRIPT_DIR/test_result/mmau_gemini_baseline"
LOG_FILE="$OUTPUT_DIR/baseline.log"
mkdir -p "$OUTPUT_DIR"

exec > "$LOG_FILE" 2>&1

echo "========================================"
echo "MMAU Baseline started at $(date)"
echo "========================================"

.venv/bin/python run_gemini_baseline.py \
  --dataset-module mmau \
  --dataset-dir /cpfs/user/jingpeng/workspace/nfs/data/test/MMAU \
  --output-dir "$OUTPUT_DIR" \
  --num-samples 1000 \
  --workers 4 \
  --gemini-model gemini-2.5-pro \
  --checkpoint-interval 10

echo "========================================"
echo "MMAU Baseline finished at $(date)"
echo "========================================"
