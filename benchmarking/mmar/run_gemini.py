#!/usr/bin/env python3
"""
Run MMAR Benchmark with Gemini 2.5 Pro Frontend.

This script is a variant of benchmarking.mmar.run that uses Gemini 2.5 Pro
as the frontend model instead of an OpenAI-compatible API model.

Usage:
    # Set API keys
    export DASHSCOPE_API_KEY="sk-xxx"
    export GEMINI_API_KEY="xxx"
    
    # Small test run (5 samples, default)
    python -m benchmarking.mmar.run_gemini
    
    # Run more samples
    python -m benchmarking.mmar.run_gemini --num-samples 50
    
    # Full benchmark
    python -m benchmarking.mmar.run_gemini --num-samples 1000
    
    # Resume from checkpoint
    python -m benchmarking.mmar.run_gemini --resume-from ./test_result/gemini_mmar/checkpoint.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm

from benchmarking.mmar.audio_utils import preprocess_audio_paths
from benchmarking.mmar.dataset import MMARBenchmark
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
        description="Run MMAR Benchmark with Gemini 2.5 Pro Frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Small test run (5 samples)
    python -m benchmarking.mmar.run_gemini
    
    # Run 50 samples
    python -m benchmarking.mmar.run_gemini --num-samples 50
    
    # Full benchmark (1000 samples)
    python -m benchmarking.mmar.run_gemini --num-samples 1000
    
    # Resume from checkpoint
    python -m benchmarking.mmar.run_gemini --resume-from ./test_result/gemini_mmar/checkpoint.json
        """,
    )
    
    # Model configuration
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
        help="DashScope API key for planner (or set DASHSCOPE_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="Planner API base URL",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini-base-url",
        type=str,
        default="https://runway.devops.rednote.life/openai/google/v1:generateContent",
        help="Gemini API base URL",
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
        default="/cpfs/user/jingpeng/workspace/nfs/data/test/MMAR",
        help="Dataset directory (default: /cpfs/user/jingpeng/workspace/nfs/data/test/MMAR)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to run (default: 5 for quick test)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index (default: 0)",
    )
    parser.add_argument(
        "--indices-file",
        type=str,
        default=None,
        help="File with indices to run",
    )
    
    # Execution configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cpfs/user/jingpeng/workspace/AUDIO_AGENT/test_result/gemini_mmar",
        help="Output directory (default: test_result/gemini_mmar)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N samples (default: 5)",
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
        "--no-preprocess-audio",
        action="store_true",
        help="Disable audio preprocessing (default: enabled)",
    )
    parser.add_argument(
        "--preprocess-cache-dir",
        type=str,
        default=None,
        help="Directory to cache preprocessed audio (default: {dataset_dir}/audio/.preprocessed_audio)",
    )
    return parser


def load_target_indices(indices_file: str) -> list[int]:
    """Load target indices from a file."""
    path = Path(indices_file)
    if not path.exists():
        raise FileNotFoundError(f"Indices file not found: {indices_file}")
    
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    indices: set[int] = set()
    
    if first_line == "category\tindex\tid":
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3 and parts[0] != "both_correct":
                    indices.add(int(parts[1]))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                indices.add(int(line))
    
    return sorted(indices)


async def run_benchmark(args: argparse.Namespace):
    """Run the MMAR benchmark with Gemini frontend."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  MMAR Benchmark - Gemini 2.5 Pro Frontend")
    print("=" * 60)
    print()
    print("  Frontend: gemini-2.5-pro")
    print(f"  Planner:  {args.planner_model}")
    print(f"  Planner Backend: {args.planner_backend}")
    print()
    
    # Initialize benchmark
    print("Initializing benchmark...")
    benchmark = MMARBenchmark(
        dataset_dir=args.dataset_dir,
        split="test",
    )
    dataset = benchmark.load_dataset()
    
    # Determine sample range
    total_samples = len(dataset)
    
    if args.indices_file:
        target_indices = load_target_indices(args.indices_file)
        print(f"\nLoaded {len(target_indices)} target indices from: {args.indices_file}")
        print(f"Index range: {target_indices[0]} to {target_indices[-1]}")
    else:
        start_idx = args.start_idx
        end_idx = args.num_samples + start_idx if args.num_samples else total_samples
        end_idx = min(end_idx, total_samples)
        target_indices = list(range(start_idx, end_idx))
        print(f"\nRunning on samples {target_indices[0]} to {target_indices[-1]} (total: {len(target_indices)})")
    
    # Load checkpoint if resuming
    completed_indices: set[int] = set()
    results: list[dict] = []
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResuming from checkpoint: {args.resume_from}")
        completed_indices, results = load_checkpoint(args.resume_from)
        print(f"Loaded {len(results)} completed samples")
        if completed_indices:
            original_count = len(target_indices)
            target_indices = [idx for idx in target_indices if idx not in completed_indices]
            print(f"Resuming with {len(target_indices)} remaining samples (skipped {original_count - len(target_indices)} already completed)")
            if not target_indices:
                print("All target samples already completed!")
                return
    
    # Initialize runner
    print("\nInitializing agent runner...")
    print(f"  Frontend: gemini-2.5-pro")
    print(f"  Planner:  {args.planner_model}")
    print(f"  Planner Backend: {args.planner_backend}")
    print(f"  MCP Tools: {'disabled' if args.no_mcp_tools else 'enabled'}")
    
    log_dir = str(Path(args.log_dir)) if args.log_dir else str(output_dir / "logs")
    
    runner = BenchmarkRunner(
        planner_model=args.planner_model,
        planner_backend=args.planner_backend,
        gemini_planner_api_key=args.gemini_planner_api_key,
        gemini_planner_base_url=args.gemini_planner_base_url,
        api_key=args.api_key,
        base_url=args.base_url,
        max_steps=args.max_steps,
        enable_thinking=args.enable_thinking,
        enable_mcp_tools=not args.no_mcp_tools,
        debug=args.debug,
        enable_run_logging=not args.disable_run_logging,
        log_dir=log_dir,
        use_gemini=True,
        gemini_api_key=args.gemini_api_key,
        gemini_base_url=args.gemini_base_url,
    )
    
    async with runner:
        print("\nRunning benchmark...")
        start_time = time.time()
        
        # Deferred samples queue (for token limit / rate limit retries)
        deferred_samples: list[tuple[int, dict, str, list[str]]] = []
        
        # Progress bar
        total_target = len(target_indices) + len(completed_indices)
        pbar = tqdm(
            target_indices,
            initial=len(completed_indices),
            total=total_target,
            desc="Processing",
        )
        
        for idx in pbar:
            
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
            
            # Preprocess audio if not disabled
            if not args.no_preprocess_audio:
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
                
                # Check for deferrable error (token limit / rate limit)
                if result.get("status") == "error" and "10001" in str(result.get("error", "")):
                    deferred_samples.append((idx, metadata, question, audio_paths))
                    print(f"\n[Deferred] Sample {idx} queued for retry (token limit)")
                    continue
                
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
                if "10001" in str(e):
                    deferred_samples.append((idx, metadata, question, audio_paths))
                    print(f"\n[Deferred] Sample {idx} queued for retry (token limit)")
                    continue
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
        
        # ── Multi-round Deferred Retry Phase ──
        MAX_DEFERRED_ROUNDS = 3
        round_num = 1
        while deferred_samples and round_num <= MAX_DEFERRED_ROUNDS:
            print(f"\n{'='*60}")
            print(f"  Deferred Retry Round {round_num}/{MAX_DEFERRED_ROUNDS}")
            print(f"  {len(deferred_samples)} samples to retry")
            print(f"  Cooling down for 10s...")
            print(f"{'='*60}")
            await asyncio.sleep(10)
            
            still_failed: list[tuple[int, dict, str, list[str]]] = []
            
            for idx, metadata, question, audio_paths in deferred_samples:
                print(f"\n[Deferred R{round_num}] Retrying sample {idx}...")
                try:
                    result = await runner.run_single(
                        question=question,
                        audio_paths=audio_paths,
                        run_log_name=metadata["id"],
                    )
                    # Check if still deferrable after retry
                    if result.get("status") == "error" and "10001" in str(result.get("error", "")):
                        print(f"[Deferred R{round_num}] Sample {idx} still token-limited")
                        still_failed.append((idx, metadata, question, audio_paths))
                    else:
                        prediction = result.get("answer")
                        eval_result = benchmark.evaluate_answer(
                            prediction=prediction,
                            ground_truth=metadata["ground_truth"],
                            choices=metadata.get("choices", []),
                        )
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
                        status_str = "successfully" if result.get("status") != "error" else "with error"
                        print(f"[Deferred R{round_num}] Sample {idx} retried {status_str}")
                except Exception as e:
                    print(f"\n[Deferred R{round_num}] Error running sample {idx}: {e}")
                    results.append({
                        **metadata,
                        "index": idx,
                        "prediction": None,
                        "correct": False,
                        "match_method": "runtime_error",
                        "error": str(e),
                    })
                completed_indices.add(idx)
                checkpoint_path = output_dir / "checkpoint.json"
                save_checkpoint(completed_indices, results, str(checkpoint_path))
            
            deferred_samples = still_failed
            if deferred_samples:
                print(f"\n[Deferred R{round_num}] {len(deferred_samples)} samples still failed")
            round_num += 1
        
        if deferred_samples:
            print(f"\n[Deferred] {len(deferred_samples)} samples exhausted all {MAX_DEFERRED_ROUNDS} rounds")
        else:
            print(f"\nDeferred retry completed. All samples resolved.")
        
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
        title=f"MMAR Benchmark Results\n  Frontend: gemini-2.5-pro  Planner: {args.planner_model} ({args.planner_backend})",
    )
    
    # Save checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    save_checkpoint(completed_indices, results, str(checkpoint_path))
    
    # Print report
    print("\n" + "=" * 60)
    with open(report_path, "r") as f:
        print(f.read())
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.json: Full results")
    print(f"  - report.txt: Summary report")
    print(f"  - checkpoint.json: Resume checkpoint")


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Check API keys
    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    
    if not gemini_key:
        print("Error: Gemini API key required for frontend.")
        print("Provide via --gemini-api-key or set GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    # DashScope key only needed when planner_backend is openai (default)
    if args.planner_backend == "openai":
        dashscope_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not dashscope_key:
            print("Error: DashScope API key required for OpenAI-compatible planner.")
            print("Provide via --api-key or set DASHSCOPE_API_KEY / OPENAI_API_KEY environment variable.")
            sys.exit(1)
        args.api_key = dashscope_key
    
    args.gemini_api_key = gemini_key
    
    # Gemini planner key check
    if args.planner_backend == "gemini":
        gemini_planner_key = args.gemini_planner_api_key or os.environ.get("GEMINI_API_KEY")
        if not gemini_planner_key:
            print("Error: Gemini planner API key required.")
            print("Provide via --gemini-planner-api-key or set GEMINI_API_KEY environment variable.")
            sys.exit(1)
        args.gemini_planner_api_key = gemini_planner_key
    
    # Run benchmark
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
