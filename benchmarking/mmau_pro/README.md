# MMAU-Pro Benchmark

This directory contains the MMAU-Pro benchmark implementation for the audio agent framework.

## Prerequisites

### HuggingFace Authentication

**MMAU-Pro is a gated dataset on HuggingFace.** You need to:

1. **Create a HuggingFace account** (if you don't have one): https://huggingface.co/join

2. **Accept the dataset terms**: Visit https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro and click "Access repository" to accept the terms.

3. **Get your access token**: Go to https://huggingface.co/settings/tokens and create a token.

4. **Login via CLI**:
   ```bash
   huggingface-cli login
   # Enter your token when prompted
   ```
   
   Or set environment variable:
   ```bash
   export HF_TOKEN="your-huggingface-token"
   ```

## Dataset Information

- **Name**: MMAU-Pro (gamma-lab-umd/MMAU-Pro)
- **Size**: 5,305 test samples
- **Audio**: ~47GB total (stored in data.zip)
- **Categories**: sound, music, speech, multi, spatial_audio, voice_chat, etc.
- **Task Types**: Multiple choice (MCQ) and open-ended questions

## Quick Start

### 1. Download Dataset Metadata

```bash
cd /lihaoyu/workspace/AUDIO_AGENT
python -m benchmarking.mmau_pro.download \
    --output-dir /lihaoyu/datasets/MMAU-Pro
```

### 2. Download Audio Files (Optional but Recommended)

Audio files are stored in a 47GB zip file. You can either:

**Option A: Download all audio files for offline use** (~47GB disk space)
```bash
python -m benchmarking.mmau_pro.download \
    --output-dir /lihaoyu/datasets/MMAU-Pro \
    --download-audio
```

**Option B: Download subset for testing**
```bash
python -m benchmarking.mmau_pro.download \
    --output-dir /lihaoyu/datasets/MMAU-Pro \
    --download-audio \
    --max-audio-files 100
```

**Option C: On-demand downloading (slower, requires internet)**
- Skip `--download-audio`
- Audio files will be downloaded automatically during benchmarking
- Each file is downloaded from the zip on first access

### 3. Run Benchmark

```bash
# Set API keys
export DASHSCOPE_API_KEY="your-dashscope-api-key"
export HF_TOKEN="your-huggingface-token"  # If not using huggingface-cli login

# Run full benchmark
cd /lihaoyu/workspace/AUDIO_AGENT
python -m benchmarking.mmau_pro.run \
    --frontend-model qwen3-omni-flash \
    --planner-model qwen3.5-plus \
    --output-dir ./results/mmau_pro

# Test with 10 samples first
python -m benchmarking.mmau_pro.run \
    --num-samples 10 \
    --output-dir ./results/mmau_pro_test

# Resume from checkpoint
python -m benchmarking.mmau_pro.run \
    --resume-from ./results/mmau_pro/checkpoint.json
```

### 4. View Results

```bash
cat ./results/mmau_pro/report.txt
cat ./results/mmau_pro/results.json
```

## File Structure

```
benchmarking/mmau_pro/
├── __init__.py           # Package exports
├── dataset.py            # MMAUProBenchmark class
├── download.py           # Dataset download script
├── run.py                # Main benchmark runner
├── test_benchmark.py     # Test script
└── README.md             # This file
```

## Implementation Details

### MMAUProBenchmark Class

The `MMAUProBenchmark` class (`dataset.py`) implements the `BaseBenchmark` interface:

- **load_dataset()**: Loads dataset from HuggingFace
- **format_question()**: Formats MCQ with choices or open-ended questions
- **evaluate_answer()**: Evaluates predictions using exact/contains match
- **compute_metrics()**: Computes accuracy by category and task type

### Question Formatting

MCQ questions are formatted as:
```
What is being prepared in the audio?

Options:
A. Boba tea
B. Milk
C. Coffee
...

Please answer with exactly one of the options above (just the option text, not the letter).
```

### Evaluation

- **MCQ**: Extracts choice from prediction and compares to ground truth
- **Open-ended**: Uses normalized string matching (exact or contains)
- **Metrics**: Overall accuracy + breakdown by category and task type

### Audio File Handling

Audio files are handled in three ways:

1. **Pre-downloaded**: If audio files exist locally, use them directly
2. **On-demand**: Download individual files from data.zip on first access
3. **Streaming**: The HuggingFace datasets library can stream audio (if configured)

## Troubleshooting

### "401 Client Error" or "Repository Not Found"

The dataset is gated. You need to:
1. Accept terms at https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro
2. Login with `huggingface-cli login` or set `HF_TOKEN`

### "No audio files found"

Run the download script first:
```bash
python -m benchmarking.mmau_pro.download --download-audio --max-audio-files 10
```

Or let the benchmark download on-demand (slower).

### Out of Disk Space

The full dataset requires ~47GB. Options:
- Use `--max-audio-files N` to download only N files for testing
- Use on-demand downloading (no `--download-audio` flag)
- Download to a different disk with more space

### API Rate Limiting

The benchmark includes retry logic. If you hit rate limits:
- Reduce `--num-samples` for testing
- Use `--checkpoint-interval` to save progress more frequently
- Resume with `--resume-from` if interrupted

## Citation

If you use MMAU-Pro, please cite:

```bibtex
@article{kumar2025mmau,
  title={MMAU-Pro: A Challenging and Comprehensive Benchmark for Holistic Evaluation of Audio General Intelligence},
  author={Kumar, Sonal and Sedl{\'a}{\v{c}}ek, {\v{S}}imon and others},
  journal={arXiv preprint arXiv:2508.13992},
  year={2025}
}
```
