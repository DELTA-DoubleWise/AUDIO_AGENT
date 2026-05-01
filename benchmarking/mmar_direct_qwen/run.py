#!/usr/bin/env python3
"""
Run MMAR Benchmark with direct Qwen3.5-omni-plus API calls.

This script bypasses the Audio Agent framework and sends audio + question
directly to qwen3.5-omni-plus via the DashScope OpenAI-compatible API.

Usage:
    export DASHSCOPE_API_KEY="sk-xxx"
    python -m benchmarking.mmar_direct_qwen.run \
        --output-dir ./benchmarking/results/mmar_direct_qwen_lowtemp

    # Run subset for testing
    python -m benchmarking.mmar_direct_qwen.run \
        --num-samples 100 \
        --output-dir ./benchmarking/results/mmar_direct_qwen_test

    # Resume from checkpoint
    python -m benchmarking.mmar_direct_qwen.run \
        --resume-from ./benchmarking/results/mmar_direct_qwen_lowtemp/checkpoint.json
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm

from benchmarking.mmar.dataset import MMARBenchmark
from benchmarking.utils import (
    format_duration,
    load_checkpoint,
    save_checkpoint,
    save_metrics_report,
    save_results,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run MMAR Benchmark with direct Qwen3.5-omni-plus API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run
    python -m benchmarking.mmar_direct_qwen.run --output-dir ./results/mmar_direct_qwen

    # Run subset for testing
    python -m benchmarking.mmar_direct_qwen.run --num-samples 100 --output-dir ./results/test

    # Resume from checkpoint
    python -m benchmarking.mmar_direct_qwen.run --resume-from ./results/mmar_direct_qwen/checkpoint.json
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5-omni-plus",
        help='Model name (default: qwen3.5-omni-plus)',
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set DASHSCOPE_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Sampling temperature (default: 0.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/lihaoyu/datasets/MMAR",
        help="Dataset directory (default: /lihaoyu/datasets/MMAR)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to run (default: all)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/mmar_direct_qwen",
        help="Output directory (default: ./results/mmar_direct_qwen)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N samples (default: 10)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )

    return parser


def encode_audio(audio_path: str) -> tuple[str, str]:
    """Encode audio file to base64 data URL and determine format."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_format = path.suffix.lstrip(".").lower()
    if audio_format not in {"wav", "mp3", "ogg", "m4a", "flac"}:
        audio_format = "wav"

    with open(path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:;base64,{audio_base64}", audio_format


def call_qwen(
    client,
    model: str,
    audio_path: str,
    question: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str | None, str | None]:
    """
    Call Qwen3.5-omni-plus directly with audio and question.

    Returns:
        Tuple of (prediction_text, error_message). If error occurs,
        prediction_text is None and error_message is set.
    """
    try:
        audio_data_url, audio_format = encode_audio(audio_path)
    except Exception as e:
        return None, f"Audio encoding error: {e}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert audio reasoning assistant. "
                "Listen to the provided audio carefully. The audio may contain sound events, music, speech, or any mixture of them. "
                "Analyze what you hear deeply and reason step by step. "
                "After your reasoning, conclude your response with a line starting with 'Final Answer:' followed by your chosen option. "
                # "Direct answer with 'Final Answer:' followed by your chosen option. "
                "You may write either the full option text or the letter label (A, B, C, D)."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data_url,
                        "format": audio_format,
                    },
                },
            ],
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            modalities=["text"],
        )
    except Exception as e:
        return None, f"API call error: {e}"

    text_response = ""
    try:
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text_response += chunk.choices[0].delta.content
    except Exception as e:
        return None, f"Stream processing error: {e}"

    return text_response.strip(), None


def resolve_choice(segment: str, choices: list[str]) -> str | None:
    """
    Resolve the final answer segment to an actual choice.

    Tries:
    1. Exact / contains match with choice text
    2. Letter label match (A, B, C, ...) mapped to the corresponding choice
    """
    if not segment or not choices:
        return None

    # Strategy 1: direct choice text match (exact / contains / quoted)
    from benchmarking.metrics import extract_choice
    matched = extract_choice(segment, choices)
    if matched:
        return matched

    # Strategy 2: letter label match
    # Look for standalone letter at the start or after common prefixes
    import re
    # Patterns like "A", "A.", "(A)", "option A", "answer: A"
    letter_match = re.search(
        r'(?:^|[\s:,-])\(?([A-' + chr(64 + len(choices)) + r'])\)?\.?(?:\s|$)',
        segment.strip(),
        re.IGNORECASE
    )
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord('A')
        if 0 <= idx < len(choices):
            return choices[idx]

    return None


def extract_final_answer_segment(text: str) -> str:
    """
    Extract the final answer portion after the last 'Final Answer:' marker.

    If the marker is not found, returns the original text.
    This isolates the final choice from any intermediate reasoning that might
    mention other options.
    """
    if not text:
        return text
    marker = "Final Answer:"
    lower_text = text.lower()
    lower_marker = marker.lower()
    idx = lower_text.rfind(lower_marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    return text


def _format_question_for_direct(benchmark, sample: dict) -> tuple[str, list[str]]:
    """
    Format a dataset sample for direct model evaluation.

    Similar to benchmark.format_question() but without the restrictive
    'just the option text, not the letter' suffix, allowing the model to
    reason freely while still indicating it should select one option.
    """
    import os

    question = sample["question"]
    choices = sample.get("choices", [])

    if choices and len(choices) > 0:
        lines = [question, "", "Options:"]
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, ...
            lines.append(f"{letter}. {choice}")
        lines.extend([
            "",
            "Please select exactly one of the above options. "
            "You may explain your reasoning, but make sure to state your final choice clearly at the end of your response.",
        ])
        question = "\n".join(lines)

    # Resolve audio path (mirroring benchmark.format_question logic)
    audio_path = sample.get("audio_path", "")
    if not audio_path:
        raise ValueError(f"No audio path found for sample {sample.get('id')}")

    resolved_paths = []
    if not os.path.isabs(audio_path) and benchmark._audio_base_path:
        if audio_path.startswith("./"):
            audio_path = audio_path[2:]
        full_path = benchmark._audio_base_path / audio_path
        if not full_path.exists():
            benchmark._download_audio_file(audio_path, full_path)
        audio_path = str(full_path)

    resolved_paths.append(audio_path)
    return question, resolved_paths


def run_benchmark(args: argparse.Namespace):
    """Run the MMAR benchmark with direct API calls."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MMAR Direct Benchmark (Qwen3.5-omni-plus)")
    print("=" * 60)
    print()

    # Initialize OpenAI client
    try:
        from openai import OpenAI
    except ImportError as e:
        print("Error: openai package is required. Install with: pip install openai")
        raise SystemExit(1) from e

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Initialize benchmark
    print("Initializing benchmark...")
    benchmark = MMARBenchmark(
        dataset_dir=args.dataset_dir,
        split="test",
    )
    dataset = benchmark.load_dataset()

    total_samples = len(dataset)
    start_idx = args.start_idx
    end_idx = args.num_samples + start_idx if args.num_samples else total_samples
    end_idx = min(end_idx, total_samples)

    print(f"\nRunning on samples {start_idx} to {end_idx - 1} (total: {end_idx - start_idx})")

    # Load checkpoint if resuming
    completed_indices: set[int] = set()
    results: list[dict] = []

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResuming from checkpoint: {args.resume_from}")
        completed_indices, results = load_checkpoint(args.resume_from)
        print(f"Loaded {len(results)} completed samples")
        if completed_indices:
            start_idx = max(completed_indices) + 1
            print(f"Resuming from index {start_idx}")

    print(f"\nModel: {args.model}")
    print(f"Base URL: {args.base_url}")

    print("\nRunning benchmark...")
    start_time = time.time()

    pbar = tqdm(
        range(start_idx, end_idx),
        initial=len(completed_indices),
        total=end_idx - start_idx,
        desc="Processing",
    )

    for idx in pbar:
        if idx in completed_indices:
            continue

        sample = dataset[idx]
        metadata = benchmark.get_sample_metadata(sample)

        # Format question and resolve audio path
        try:
            question, audio_paths = _format_question_for_direct(benchmark, sample)
        except Exception as e:
            print(f"\nError formatting sample {idx}: {e}")
            results.append({
                **metadata,
                "index": idx,
                "prediction": None,
                "correct": False,
                "match_method": "format_error",
                "error": str(e),
            })
            completed_indices.add(idx)
            continue

        audio_path = audio_paths[0]

        # Call model directly
        prediction, error = call_qwen(
            client=client,
            model=args.model,
            audio_path=audio_path,
            question=question,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        # Isolate the final answer for evaluation to avoid incorrect choices
        # mentioned during reasoning from affecting the match.
        final_segment = extract_final_answer_segment(prediction) if prediction else None
        prediction_for_eval = resolve_choice(final_segment, metadata.get("choices", [])) if final_segment else None

        # Evaluate
        eval_result = benchmark.evaluate_answer(
            prediction=prediction_for_eval,
            ground_truth=metadata["ground_truth"],
            choices=metadata.get("choices", []),
        )

        results.append({
            **metadata,
            "index": idx,
            "prediction": prediction,
            **eval_result,
            "agent_status": "answered" if prediction is not None else "failed",
            "step_count": 1 if prediction is not None else 0,
            "tool_calls": 0,
            "confidence": 1.0 if eval_result.get("correct", False) else 0.0,
            "agent_error": error,
        })

        # Update progress bar postfix
        correct_so_far = sum(1 for r in results if r.get("correct", False))
        accuracy_so_far = correct_so_far / len(results) if results else 0.0
        pbar.set_postfix({"acc": f"{accuracy_so_far:.2%}"})

        completed_indices.add(idx)

        # Save checkpoint periodically
        if len(completed_indices) % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / "checkpoint.json"
            save_checkpoint(completed_indices, results, str(checkpoint_path))

            results_path = output_dir / "results_partial.json"
            metrics = benchmark.compute_metrics(results)
            save_results(results, str(results_path), metrics)

    elapsed = time.time() - start_time
    print(f"\n\nBenchmark completed in {format_duration(elapsed)}")

    # Compute final metrics
    print("\nComputing metrics...")
    metrics = benchmark.compute_metrics(results)

    # Save final results
    results_path = output_dir / "results.json"
    save_results(results, str(results_path), metrics)

    # Save metrics report
    report_path = output_dir / "report.txt"
    save_metrics_report(
        metrics,
        str(report_path),
        title=f"MMAR Direct Benchmark Results\n  Model: {args.model}",
    )

    # Save checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    save_checkpoint(completed_indices, results, str(checkpoint_path))

    # Print report
    print("\n" + "=" * 60)
    with open(report_path, "r") as f:
        print(f.read())

    print(f"\nResults saved to: {output_dir}")
    print("  - results.json: Full results")
    print("  - report.txt: Summary report")
    print("  - checkpoint.json: Resume checkpoint")


def main():
    parser = build_parser()
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key required.")
        print("Provide via --api-key or set DASHSCOPE_API_KEY / OPENAI_API_KEY environment variable.")
        sys.exit(1)

    args.api_key = api_key
    run_benchmark(args)


if __name__ == "__main__":
    main()
