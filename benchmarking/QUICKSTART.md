# Quick Start Guide: MMAU-Pro Benchmark

## Overview

This benchmarking framework provides a reusable system for evaluating the audio agent on MMAU-Pro and future benchmarks.

## ⚠️ Important: HuggingFace Authentication Required

**MMAU-Pro is a gated dataset on HuggingFace.** You must:

1. **Accept the dataset terms**: https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro
2. **Get your token**: https://huggingface.co/settings/tokens  
3. **Login**: `huggingface-cli login` or `export HF_TOKEN="your-token"`

## What Was Built

```
benchmarking/
├── base.py           # Abstract BaseBenchmark class - reusable for any benchmark
├── metrics.py        # Evaluation metrics (exact match, contains match, MCQ extraction)
├── runners.py        # BenchmarkRunner - wraps AudioAgent with API components
├── utils.py          # Progress tracking, checkpointing, reporting utilities
├── mmau_pro/         # MMAU-Pro specific implementation
│   ├── dataset.py    # MMAUProBenchmark class
│   ├── download.py   # Dataset download script
│   ├── run.py        # Main benchmark runner
│   ├── test_benchmark.py  # Test script
│   └── README.md     # MMAU-Pro specific docs
└── README.md         # Full documentation
```

## Quick Commands

### 0. Setup Authentication (Required)

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN="your-huggingface-token"

# Set API key for agent
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 1. Download Dataset (Done)
```bash
cd /lihaoyu/workspace/AUDIO_AGENT
python -m benchmarking.mmau_pro.download --output-dir /lihaoyu/datasets/MMAU-Pro
```

### 2. Test the Implementation
```bash
cd /lihaoyu/workspace/AUDIO_AGENT
python -m benchmarking.mmau_pro.test_benchmark
```

### 3. Run Benchmark
```bash
# Set API key
export DASHSCOPE_API_KEY="your-api-key"

cd /lihaoyu/workspace/AUDIO_AGENT

# Full benchmark (5,305 samples)
python -m benchmarking.mmau_pro.run \
    --frontend-model qwen3-omni-flash \
    --planner-model qwen3.5-plus \
    --output-dir ./results/mmau_pro

# Test run with 10 samples
python -m benchmarking.mmau_pro.run \
    --num-samples 10 \
    --output-dir ./results/mmau_pro_test

# Resume from checkpoint
python -m benchmarking.mmau_pro.run \
    --resume-from ./results/mmau_pro/checkpoint.json
```

### 4. View Results
```bash
cat ./results/mmau_pro/report.txt       # Summary report
cat ./results/mmau_pro/results.json     # Full results
```

## Key Features

### Reusable Framework
The `BaseBenchmark` class provides a standard interface:
```python
class MyNewBenchmark(BaseBenchmark):
    def load_dataset(self): ...
    def format_question(self, sample): ...
    def evaluate_answer(self, prediction, ground_truth, **kwargs): ...
    def compute_metrics(self, results): ...
```

### Resume Capability
Checkpoints are saved every N samples (default: 10). Resume with `--resume-from checkpoint.json`.

### Progress Tracking
Progress bar shows:
- Current accuracy
- Agent status
- Estimated time remaining

### Evaluation Metrics
- **MCQ**: Choice extraction and matching
- **Open-ended**: Normalized string matching
- **Category breakdown**: Accuracy by category and task type

## Dataset Info

**MMAU-Pro Test Split:**
- Total samples: 5,305
- Categories: sound (1,048), music (1,418), speech (891), and more
- Task types: MCQ ("sound") and open-ended ("open")
- Length types: short, medium, long, ultra-long

## Cost Estimation

For 5,305 samples with API models:
- Each sample: 1+ API calls (frontend + planner steps)
- With MCP tools: Additional tool execution time
- Estimated: Several hours for full dataset

## Troubleshooting

**Issue**: Audio files not found
- Audio files are streamed from HuggingFace on first access
- The dataset handles this automatically

**Issue**: API rate limiting
- The runner has built-in retry logic
- Reduce `--num-samples` for testing
- Use checkpointing to resume if interrupted

**Issue**: Out of memory
- Reduce `--max-steps` to limit agent iterations
- Use `--no-mcp-tools` to disable tools

## Next Steps

To add a new benchmark (e.g., Clotho, AudioCaps):
1. Create `benchmarking/my_benchmark/dataset.py`
2. Implement `BaseBenchmark` interface
3. Create `run.py` using `BenchmarkRunner`
4. See `README.md` for full guide
