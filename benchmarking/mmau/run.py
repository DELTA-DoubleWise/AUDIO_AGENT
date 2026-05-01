#!/usr/bin/env python3
"""
Run MMAU Benchmark.

This script runs the audio agent on the MMAU test-mini dataset and evaluates
performance.

Usage:
    # Set API key
    export DASHSCOPE_API_KEY="sk-xxx"

    # Run full benchmark (1,000 samples)
    python -m benchmarking.mmau.run \
        --frontend-model qwen3-omni-flash \
        --planner-model qwen3.5-plus \
        --output-dir ./results/mmau

    # Run subset for testing
    python -m benchmarking.mmau.run \
        --num-samples 100 \
        --output-dir ./results/mmau_test

    # Resume from checkpoint
    python -m benchmarking.mmau.run \
        --resume-from ./results/mmau/checkpoint.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm

from benchmarking.mmau.audio_utils import preprocess_audio_paths
from benchmarking.mmau.dataset import MMAUBenchmark
from benchmarking.runners import BenchmarkRunner
from benchmarking.utils import (
    save_results,
    save_checkpoint,
    load_checkpoint,
    save_metrics_report,
    format_duration,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run MMAU Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run
    python -m benchmarking.mmau.run --output-dir ./results/mmau

    # Run subset for testing
    python -m benchmarking.mmau.run --num-samples 100 --output-dir ./results/test

    # Resume from checkpoint
    python -m benchmarking.mmau.run --resume-from ./results/mmau/checkpoint.json
        """,
    )

    # Model configuration
    parser.add_argument(
        "--frontend-model",
        type=str,
        default="qwen3-omni-flash",
        help="Frontend model name (default: qwen3-omni-flash)",
    )
    parser.add_argument(
        "--planner-model",
        type=str,
        default="qwen3.5-plus",
        help="Planner model name (default: qwen3.5-plus)",
    )
    parser.add_argument(
        "--planner-backend",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="Planner backend: openai (default) or gemini",
    )
    parser.add_argument(
        "--gemini-planner-api-key",
        type=str,
        default=None,
        help="Gemini planner API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini-planner-base-url",
        type=str,
        default=None,
        help="Gemini planner base URL",
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
        help="Sampling temperature (default: 0.05)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Max agent steps (default: 10)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for planner",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/lihaoyu/datasets/MMAU",
        help="Dataset directory (default: /lihaoyu/datasets/MMAU)",
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

    # Execution configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmarking/results/mmau",
        help="Output directory (default: ./results/mmau)",
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
    parser.add_argument(
        "--no-mcp-tools",
        action="store_true",
        help="Disable MCP tools",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--disable-run-logging",
        action="store_true",
        help="Disable run logging to files",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for run logs (default: {output_dir}/logs)",
    )
    parser.add_argument(
        "--preprocess-audio",
        action="store_true",
        help="Preprocess audio to mono 16kHz before sending to agent",
    )
    parser.add_argument(
        "--preprocess-cache-dir",
        type=str,
        default=None,
        help="Directory to cache preprocessed audio (default: {dataset_dir}/audio/.preprocessed_audio)",
    )

    return parser


async def run_benchmark(args: argparse.Namespace):
    """
    Run the MMAU benchmark.

    Args:
        args: Parsed command-line arguments.
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MMAU Benchmark")
    print("=" * 60)
    print()

    # Initialize benchmark
    print("Initializing benchmark...")
    benchmark = MMAUBenchmark(
        dataset_dir=args.dataset_dir,
        split="test_mini",
    )
    dataset = benchmark.load_dataset()

    # Determine sample range
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
        # Adjust start_idx based on checkpoint
        if completed_indices:
            start_idx = max(completed_indices) + 1
            print(f"Resuming from index {start_idx}")

    # Initialize runner
    print("\nInitializing agent runner...")
    print(f"  Frontend: {args.frontend_model}")
    print(f"  Planner: {args.planner_model}")
    print(f"  Planner Backend: {args.planner_backend}")
    print(f"  MCP Tools: {'disabled' if args.no_mcp_tools else 'enabled'}")

    log_dir = args.log_dir if args.log_dir else str(output_dir / "logs")

    runner = BenchmarkRunner(
        frontend_model=args.frontend_model,
        planner_model=args.planner_model,
        planner_backend=args.planner_backend,
        gemini_planner_api_key=args.gemini_planner_api_key,
        gemini_planner_base_url=args.gemini_planner_base_url,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        enable_thinking=args.enable_thinking,
        enable_mcp_tools=not args.no_mcp_tools,
        debug=args.debug,
        enable_run_logging=not args.disable_run_logging,
        log_dir=log_dir,
    )

    async with runner:
        print("\nRunning benchmark...")
        start_time = time.time()

        # Progress bar
        pbar = tqdm(
            range(start_idx, end_idx),
            initial=len(completed_indices),
            total=end_idx - start_idx,
            desc="Processing",
        )

        for idx in pbar:
            # Skip if already completed
            if idx in completed_indices:
                continue

            # Get sample
            sample = dataset[idx]
            metadata = benchmark.get_sample_metadata(sample)

            # Format question
            try:
                question, audio_paths = benchmark.format_question(sample)
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

            # Preprocess audio if requested
            if args.preprocess_audio:
                try:
                    cache_dir = args.preprocess_cache_dir
                    audio_paths = preprocess_audio_paths(
                        audio_paths,
                        cache_dir=cache_dir,
                        target_sr=16000,
                        target_channels=1,
                    )
                except Exception as e:
                    print(f"\nError preprocessing audio for sample {idx}: {e}")
                    results.append({
                        **metadata,
                        "index": idx,
                        "prediction": None,
                        "correct": False,
                        "match_method": "preprocess_error",
                        "error": str(e),
                    })
                    completed_indices.add(idx)
                    continue

            # Run agent
            try:
                result = await runner.run_single(
                    question=question,
                    audio_paths=audio_paths,
                    run_log_name=metadata["id"],
                )

                prediction = result.get("answer")

                # Evaluate
                eval_result = benchmark.evaluate_answer(
                    prediction=prediction,
                    ground_truth=metadata["ground_truth"],
                    choices=metadata.get("choices", []),
                )

                # Store result
                results.append({
                    **metadata,
                    "index": idx,
                    "prediction": prediction,
                    **eval_result,
                    "agent_status": result.get("status"),
                    "step_count": result.get("step_count"),
                    "tool_calls": result.get("tool_calls"),
                    "confidence": result.get("confidence"),
                    "agent_error": result.get("error"),
                })

                # Update progress bar postfix
                correct_so_far = sum(1 for r in results if r.get("correct", False))
                accuracy_so_far = correct_so_far / len(results) if results else 0.0
                pbar.set_postfix({
                    "acc": f"{accuracy_so_far:.2%}",
                    "status": result.get("status", "unknown"),
                })

            except Exception as e:
                print(f"\nError running sample {idx}: {e}")
                results.append({
                    **metadata,
                    "index": idx,
                    "prediction": None,
                    "correct": False,
                    "match_method": "runtime_error",
                    "error": str(e),
                })

            # Mark as completed
            completed_indices.add(idx)

            # Save checkpoint periodically
            if len(completed_indices) % args.checkpoint_interval == 0:
                checkpoint_path = output_dir / "checkpoint.json"
                save_checkpoint(completed_indices, results, str(checkpoint_path))

                # Also save intermediate results
                results_path = output_dir / "results_partial.json"
                metrics = benchmark.compute_metrics(results)
                save_results(results, str(results_path), metrics)

        # Final timing
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
        title=f"MMAU Benchmark Results\n  Models: {args.frontend_model} / {args.planner_model} ({args.planner_backend})",
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

    # Check API key
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key required.")
        print("Provide via --api-key or set DASHSCOPE_API_KEY / OPENAI_API_KEY environment variable.")
        sys.exit(1)

    args.api_key = api_key

    # Run benchmark
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
