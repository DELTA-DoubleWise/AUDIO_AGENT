python -m benchmarking.mmar.run \
 --frontend-model qwen3.5-omni-plus \
 --planner-model qwen3.5-plus \
 --output-dir /lihaoyu/workspace/AUDIO_AGENT/benchmarking/results/mmar \
 --log-dir /lihaoyu/workspace/AUDIO_AGENT/benchmarking/results/mmar/logs \

 python -m benchmarking.mmar.run_multi \
    --frontend-model qwen3.5-omni-plus \
    --planner-model qwen3.5-plus \
    --output-dir /lihaoyu/workspace/AUDIO_AGENT/benchmarking/results/mmar_v3


 python -m benchmarking.mmar.run \
  --indices-file /lihaoyu/workspace/AUDIO_AGENT/benchmarking/results/mmar_v3/failed_indices.txt \
  --output-dir /lihaoyu/workspace/AUDIO_AGENT/benchmarking/results/mmar_v3_rerun \
  --frontend-model qwen3.5-omni-plus \
  --planner-model qwen3.5-plus

export DASHSCOPE_API_KEY="sk-f8ae3fc37bdd4953977e813f77b7324f"

 python -m benchmarking.mmar.run_multi \
  --output-dir /lihaoyu/workspace/AUDIO_AGENT/benchmarking/results/mmar_v4 \
  --frontend-model qwen3.5-omni-plus \
  --planner-model qwen3.5-plus

python -m benchmarking.mmar_direct_qwen.run \
        --output-dir ./benchmarking/results/mmar_qwen_direct_answer