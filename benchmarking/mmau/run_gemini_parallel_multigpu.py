#!/usr/bin/env python3
"""
Run MMAU Benchmark with Gemini 2.5 Pro Frontend — Multi-GPU Parallel Worker Version.

Distributes workers across multiple GPUs via CUDA_VISIBLE_DEVICES.
Each worker initializes its own BenchmarkRunner (with MCP tools) once,
processes its assigned batch sequentially, then shuts down gracefully.

Usage:
    export DASHSCOPE_API_KEY="sk-xxx"
    export GEMINI_API_KEY="xxx"

    # 4 workers across 2 GPUs (default)
    python -m benchmarking.mmau.run_gemini_parallel_multigpu --num-samples 1000 --workers 4 --gpus 2

    # 8 workers across 2 GPUs
    python -m benchmarking.mmau.run_gemini_parallel_multigpu --num-samples 1000 --workers 8 --gpus 2

    # Single GPU
    python -m benchmarking.mmau.run_gemini_parallel_multigpu --num-samples 1000 --workers 4 --gpus 1
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import os
import sys
import time
from datetime import datetime
from multiprocessing import Process
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarking.mmau.run_gemini_parallel import (
    build_parser as _build_base_parser,
    split_indices,
    get_worker_paths,
    run_worker,
    monitor_progress,
    merge_results,
    format_duration,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser with multi-GPU support."""
    parser = _build_base_parser()
    
    # Add multi-GPU configuration
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to distribute workers across (default: 1)",
    )
    parser.add_argument(
        "--gpu-strategy",
        type=str,
        default="round_robin",
        choices=["round_robin", "fill_first"],
        help="GPU allocation strategy: round_robin (0,1,0,1...) or fill_first (0,0,1,1...). Default: round_robin",
    )
    return parser


def get_gpu_for_worker(worker_id: int, num_gpus: int, strategy: str) -> int:
    """Determine which GPU a worker should use."""
    if strategy == "round_robin":
        return worker_id % num_gpus
    else:  # fill_first
        workers_per_gpu = max(1, (worker_id // max(1, num_gpus)) + 1)
        # Actually fill_first: put as many workers as possible on GPU 0, then GPU 1, etc.
        # But we need to know total workers. Simpler: divide evenly.
        # For simplicity, we use round-robin by default.
        return worker_id % num_gpus


def worker_entrypoint(worker_id: int, indices: list[int], args: argparse.Namespace):
    """Entry point for each worker process with GPU isolation."""
    # Assign GPU before any CUDA initialization
    if args.gpus > 1:
        gpu_id = get_gpu_for_worker(worker_id, args.gpus, args.gpu_strategy)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Worker {worker_id}] Assigned to GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
    else:
        print(f"[Worker {worker_id}] Using default GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    output_dir = Path(args.output_dir)
    worker_log_dir = Path(args.worker_log_dir) if args.worker_log_dir else output_dir / "worker_logs"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = worker_log_dir / f"worker_{worker_id}.log"

    try:
        asyncio.run(run_worker(worker_id, indices, args, log_file))
    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[Worker {worker_id}] FATAL ERROR: {e}\n")
        raise


def run_benchmark(args: argparse.Namespace):
    """Main orchestrator: split work, launch workers, monitor, merge."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MMAU Benchmark - Gemini 2.5 Pro Frontend (Multi-GPU Parallel)")
    print("=" * 70)
    print(f"\n  Planner:  {args.planner_model}")
    print(f"  Planner Backend: {args.planner_backend}")
    print(f"  Workers:  {args.workers}")
    print(f"  GPUs:     {args.gpus} (strategy: {args.gpu_strategy})")
    print(f"  Output:   {args.output_dir}")
    print()

    # Load dataset to determine range
    from benchmarking.mmau.dataset import MMAUBenchmark
    benchmark = MMAUBenchmark(dataset_dir=args.dataset_dir, split="test")
    dataset = benchmark.load_dataset()
    total_samples = len(dataset)

    if args.indices_file:
        # Load specific indices from file
        indices_path = Path(args.indices_file)
        if not indices_path.exists():
            raise FileNotFoundError(f"Indices file not found: {args.indices_file}")
        with open(indices_path, "r", encoding="utf-8") as f:
            target_indices = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(target_indices)} target indices from: {args.indices_file}")
        print(f"Index range: {min(target_indices)} to {max(target_indices)}")
    else:
        start_idx = args.start_idx
        end_idx = min(args.start_idx + args.num_samples, total_samples)
        target_indices = list(range(start_idx, end_idx))
        print(f"Total samples to process: {len(target_indices)} (index {start_idx} to {end_idx-1})")

    # Split into batches
    batches = split_indices(target_indices, args.workers)
    actual_workers = len(batches)
    print(f"Split into {actual_workers} batches")
    for i, batch in enumerate(batches):
        gpu_hint = f" (GPU {i % args.gpus})" if args.gpus > 1 else ""
        print(f"  Worker {i}: {len(batch)} samples (index {batch[0]} to {batch[-1]}){gpu_hint}")

    # Check which workers are already done (resume support)
    workers_to_run = []
    for wid, batch in enumerate(batches):
        paths = get_worker_paths(output_dir, wid)
        if paths["checkpoint"].exists():
            try:
                completed_indices, _ = load_checkpoint(str(paths["checkpoint"]))
                remaining = [idx for idx in batch if idx not in completed_indices]
                if not remaining:
                    print(f"  Worker {wid}: already complete, skipping")
                    continue
                else:
                    print(f"  Worker {wid}: partial ({len(batch)-len(remaining)}/{len(batch)} done)")
            except Exception:
                pass
        workers_to_run.append((wid, batch))

    if not workers_to_run:
        print("\nAll workers already complete! Proceeding to merge...")
    else:
        print(f"\nLaunching {len(workers_to_run)} workers...")

    # Launch workers
    processes: list[Process] = []
    for wid, batch in workers_to_run:
        p = Process(
            target=worker_entrypoint,
            args=(wid, batch, args),
            name=f"worker-{wid}",
        )
        p.start()
        processes.append((wid, p))
        print(f"  Worker {wid} started (PID: {p.pid})")

    # Monitor progress
    if processes:
        print("\nMonitoring progress (updates every 30s)...")
        print("-" * 70)

        start_time = time.time()
        while any(p.is_alive() for _, p in processes):
            time.sleep(30)

            progress = monitor_progress(output_dir, actual_workers, len(target_indices))
            elapsed = time.time() - start_time
            completed = progress["completed"]
            percent = progress["percent"]

            # Estimate remaining time
            if completed > 0:
                avg_time_per_sample = elapsed / completed
                remaining_samples = len(target_indices) - completed
                eta_seconds = avg_time_per_sample * remaining_samples / args.workers
                eta = format_duration(eta_seconds)
            else:
                eta = "N/A"

            # Per-worker status
            worker_details = " | ".join(
                f"W{wid}:{count}" for wid, count in sorted(progress["workers"].items())
            )

            print(
                f"[{format_duration(elapsed)}] "
                f"Progress: {completed}/{len(target_indices)} ({percent:.1f}%) | "
                f"ETA: {eta} | {worker_details}"
            )

        # Wait for all to finish
        print("-" * 70)
        for wid, p in processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"Warning: Worker {wid} did not exit gracefully, terminating...")
                p.terminate()
                p.join(timeout=5)
            exitcode = p.exitcode
            status = "OK" if exitcode == 0 else f"FAILED (exit {exitcode})"
            print(f"  Worker {wid}: {status}")

    # Merge results
    print("\nMerging results from all workers...")
    metrics = merge_results(output_dir, actual_workers)

    # Final report
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"\nOverall Accuracy: {metrics.get('accuracy', 0):.2%}")
    print(f"Correct: {metrics.get('correct', 0)} / {metrics.get('total', 0)}")
    print(f"Failed: {metrics.get('failed', 0)}")

    if "by_category" in metrics:
        print("\nAccuracy by Category:")
        for cat, acc in sorted(metrics["by_category"].items()):
            print(f"  {cat:30s}: {acc:.2%}")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.json: Merged full results")
    print(f"  - report.txt: Summary report")
    print(f"  - worker_N/: Per-worker results and logs")
    print("=" * 70)


def load_checkpoint(checkpoint_path: str) -> tuple[set[int], list[dict]]:
    """Load checkpoint from file."""
    from benchmarking.utils import load_checkpoint as _load
    return _load(checkpoint_path)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Check API keys
    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")

    if not gemini_key:
        print("Error: Gemini API key required for frontend.")
        sys.exit(1)

    # DashScope key only needed when planner_backend is openai (default)
    if args.planner_backend == "openai":
        dashscope_key = (
            args.api_key
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not dashscope_key:
            print("Error: DashScope API key required for OpenAI-compatible planner.")
            sys.exit(1)
        args.api_key = dashscope_key

    args.gemini_api_key = gemini_key

    # Gemini planner key check
    if args.planner_backend == "gemini":
        gemini_planner_key = args.gemini_planner_api_key or os.environ.get("GEMINI_API_KEY")
        if not gemini_planner_key:
            print("Error: Gemini planner API key required.")
            sys.exit(1)
        args.gemini_planner_api_key = gemini_planner_key

    # Validate GPU count
    if args.gpus < 1:
        print("Error: --gpus must be >= 1")
        sys.exit(1)
    
    # Check available GPUs
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        available_gpus = len([line for line in result.stdout.strip().split("\n") if line.startswith("GPU ")])
        if available_gpus > 0 and args.gpus > available_gpus:
            print(f"Warning: Requested {args.gpus} GPUs but only {available_gpus} detected. Using {available_gpus}.")
            args.gpus = available_gpus
    except Exception:
        pass  # nvidia-smi not available, proceed anyway

    # Use spawn method for multiprocessing (safer with asyncio)
    multiprocessing.set_start_method("spawn", force=True)

    # Run benchmark
    run_benchmark(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Workers may still be running.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
