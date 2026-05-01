#!/bin/bash
# MMAU Full Benchmark - Dual Frontend ENABLED
# 用法: nohup bash run_mmau_dual.sh &

cd "$(dirname "$0")"

export GEMINI_API_KEY=1561cda555764bed97c8acba1ac46f92
export DASHSCOPE_API_KEY=sk-f8ae3fc37bdd4953977e813f77b7324f

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="test_result/mmau_dual_${TIMESTAMP}"

echo "======================================"
echo "  MMAU Benchmark - DUAL FRONTEND"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: /tmp/mmau_dual_${TIMESTAMP}.log"
echo "======================================"

.venv/bin/python -m benchmarking.mmau.run_gemini_parallel_multigpu \
  --num-samples 1000 \
  --workers 4 \
  --gpus 2 \
  --gpu-strategy round_robin \
  --use-dual-frontend \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "/tmp/mmau_dual_${TIMESTAMP}.log"

echo "======================================"
echo "  Benchmark Finished"
echo "  Results: ${OUTPUT_DIR}"
echo "======================================"
