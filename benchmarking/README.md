# Benchmarking Framework for Audio Agent

This directory contains a reusable benchmarking framework for evaluating the audio agent on various audio understanding benchmarks.

## Structure

```
benchmarking/
├── __init__.py           # Package exports
├── base.py               # BaseBenchmark abstract class
├── metrics.py            # Evaluation metrics
├── runners.py            # BenchmarkRunner for agent execution
├── utils.py              # Utilities (progress, saving, reporting)
├── README.md             # This file
├── mmau_pro/             # MMAU-Pro benchmark implementation
│   ├── __init__.py
│   ├── dataset.py        # MMAUProBenchmark class
│   ├── download.py       # Dataset download script
│   └── run.py            # Main benchmark runner
└── mmar/                 # MMAR benchmark implementation
    ├── __init__.py
    ├── dataset.py        # MMARBenchmark class
    ├── download.py       # Dataset download script
    └── run.py            # Main benchmark runner
```

## Quick Start

### 1. Download MMAU-Pro Dataset

```bash
python -m benchmarking.mmau_pro.download \
    --output-dir /lihaoyu/datasets/MMAU-Pro
```

### 2. Run Benchmark

```bash
# Set API key
export DASHSCOPE_API_KEY="sk-xxx"

# Run full benchmark
python -m benchmarking.mmau_pro.run \
    --frontend-model qwen3-omni-flash \
    --planner-model qwen3.5-plus \
    --output-dir ./results/mmau_pro

# Run subset for testing
python -m benchmarking.mmau_pro.run \
    --num-samples 100 \
    --output-dir ./results/mmau_pro_test

# Resume from checkpoint
python -m benchmarking.mmau_pro.run \
    --resume-from ./results/mmau_pro/checkpoint.json
```

### 3. View Results

```bash
# View summary report
cat ./results/mmau_pro/report.txt

# View full results
cat ./results/mmau_pro/results.json
```

---

## MMAR Benchmark

MMAR (Multi-modal Audio Reasoning) is a challenging benchmark for deep reasoning
in speech, audio, music, and their mix.

**Dataset:** https://huggingface.co/datasets/BoJack/MMAR  
**Paper:** arXiv:2505.13032

### Download MMAR Dataset

```bash
python -m benchmarking.mmar.download \
    --output-dir /lihaoyu/datasets/MMAR
```

### Run MMAR Benchmark

```bash
# Set API key
export DASHSCOPE_API_KEY="sk-xxx"

# Run full benchmark (1,000 samples)
python -m benchmarking.mmar.run \
    --frontend-model qwen3-omni-flash \
    --planner-model qwen3.5-plus \
    --output-dir ./results/mmar

# Run subset for testing
python -m benchmarking.mmar.run \
    --num-samples 100 \
    --output-dir ./results/mmar_test

# Resume from checkpoint
python -m benchmarking.mmar.run \
    --resume-from ./results/mmar/checkpoint.json
```

### MMAR Dataset Info

- **Total samples:** 1,000
- **Question type:** Multiple choice (all MCQ)
- **Modalities:** sound, music, speech, and their mixes
- **Categories:** Perception Layer, Semantic Layer, Cultural Layer, Signal Layer

## Adding a New Benchmark

To add a new benchmark (e.g., Clotho, AudioCaps):

1. **Create a new directory** under `benchmarking/`:
   ```bash
   mkdir benchmarking/my_benchmark
   touch benchmarking/my_benchmark/__init__.py
   ```

2. **Implement the `BaseBenchmark` interface** in `dataset.py`:
   ```python
   from benchmarking.base import BaseBenchmark
   from datasets import load_dataset
   
   class MyBenchmark(BaseBenchmark):
       def load_dataset(self):
           return load_dataset("my-dataset", split="test")
       
       def format_question(self, sample):
           question = sample["question"]
           audio_paths = [sample["audio_path"]]
           return question, audio_paths
       
       def evaluate_answer(self, prediction, ground_truth, **kwargs):
           correct = prediction.lower().strip() == ground_truth.lower().strip()
           return {"correct": correct}
       
       def compute_metrics(self, results):
           accuracy = sum(r["correct"] for r in results) / len(results)
           return {"accuracy": accuracy}
   ```

3. **Create a `run.py` script** using `BenchmarkRunner`:
   ```python
   from benchmarking.runners import BenchmarkRunner
   from benchmarking.my_benchmark.dataset import MyBenchmark
   
   async def main():
       benchmark = MyBenchmark()
       runner = BenchmarkRunner(
           frontend_model="qwen3-omni-flash",
           planner_model="qwen3.5-plus",
       )
       
       async with runner:
           for sample in benchmark.get_dataset():
               question, audio_paths = benchmark.format_question(sample)
               result = await runner.run_single(question, audio_paths)
               # Evaluate and store results...
   ```

## Components

### BaseBenchmark

Abstract base class that defines the benchmark interface:

- `load_dataset()`: Load the dataset
- `format_question(sample)`: Convert sample to (question, audio_paths)
- `evaluate_answer(prediction, ground_truth, **kwargs)`: Evaluate single prediction
- `compute_metrics(results)`: Compute aggregate metrics

### BenchmarkRunner

Wrapper around AudioAgent for benchmark execution:

- Initializes API-based frontend and planner
- Optionally registers MCP tools
- Handles async execution and error handling
- Provides progress tracking

### Metrics

Available evaluation metrics:

- `exact_match(pred, target)`: Case-insensitive exact match
- `contains_match(pred, target)`: Check if target is in prediction
- `extract_choice(pred, choices)`: Extract selected choice from MCQ
- `choice_match(pred, target, choices)`: Full MCQ evaluation
- `f1_score(pred, target)`: Token-level F1 score

### Utilities

- `save_results()`: Save results to JSON
- `load_results()`: Load results from JSON
- `save_checkpoint()`: Save progress checkpoint
- `load_checkpoint()`: Load progress checkpoint
- `format_metrics_report()`: Format metrics as readable report

## Configuration Options

### Model Configuration

- `--frontend-model`: Audio understanding model (e.g., qwen3-omni-flash)
- `--planner-model`: Planning model (e.g., qwen3.5-plus)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max-tokens`: Max tokens to generate (default: 4096)
- `--max-steps`: Max agent steps (default: 10)
- `--enable-thinking`: Enable thinking mode for planner

### Dataset Configuration

- `--dataset-dir`: Dataset directory
- `--num-samples`: Number of samples to run (default: all)
- `--start-idx`: Start index (default: 0)

### Execution Configuration

- `--output-dir`: Output directory for results
- `--checkpoint-interval`: Save checkpoint every N samples
- `--resume-from`: Resume from checkpoint file
- `--no-mcp-tools`: Disable MCP tools
- `--debug`: Enable debug logging

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "timestamp": "2025-01-01T00:00:00",
  "num_samples": 5305,
  "metrics": {
    "accuracy": 0.52,
    "correct": 2759,
    "total": 5305,
    "failed": 23,
    "by_category": {
      "sound": 0.55,
      "music": 0.48,
      "speech": 0.53
    },
    "by_task_type": {
      "sound": 0.56,
      "open": 0.45
    }
  },
  "results": [
    {
      "id": "c93e3644-5227-4710-b27b-5c46750afbff",
      "category": "sound",
      "task_type": "sound",
      "question": "What is being prepared in the audio?",
      "ground_truth": "Boba tea",
      "prediction": "Boba tea",
      "correct": true,
      "match_method": "exact",
      "agent_status": "answered",
      "step_count": 3,
      "tool_calls": 1
    }
  ]
}
```

## Resume Capability

The framework supports resuming interrupted benchmark runs:

1. Progress is saved periodically to `checkpoint.json`
2. Use `--resume-from checkpoint.json` to continue
3. Completed samples are skipped automatically
4. Partial results are also saved for analysis

## Cost Estimation

For 5,305 samples on MMAU-Pro:
- Each sample makes 1+ API calls (frontend + planner iterations)
- With MCP tools, additional tool calls may be made
- Estimated cost depends on model pricing and average steps

Monitor your API usage and set appropriate budgets.
