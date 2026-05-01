#!/usr/bin/env python3
"""
Run MMAR Benchmark with Gemini 2.5 Pro Frontend — Parallel Worker Version.

Splits the dataset across multiple worker processes for concurrent execution.
Each worker initializes its own BenchmarkRunner (with MCP tools) once,
processes its assigned batch sequentially, then shuts down gracefully.

Usage:
    export DASHSCOPE_API_KEY="sk-xxx"
    export GEMINI_API_KEY="xxx"

    # Full benchmark with 4 workers (default)
    python -m benchmarking.mmar.run_gemini_parallel --num-samples 1000

    # More workers for faster execution
    python -m benchmarking.mmar.run_gemini_parallel --num-samples 1000 --workers 8

    # Resume from previous run
    python -m benchmarking.mmar.run_gemini_parallel --num-samples 1000 \
        --output-dir ./test_result/gemini_mmar_full
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

from benchmarking.mmar.audio_utils import preprocess_audio_paths
from benchmarking.mmar.dataset import MMARBenchmark
from benchmarking.runners import BenchmarkRunner
from benchmarking.utils import (
    save_results,
    save_checkpoint,
    load_checkpoint,
    save_metrics_report,
    format_duration,
    load_results,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run MMAR Benchmark with Gemini 2.5 Pro Frontend (Parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full benchmark with 4 workers (default)
    python -m benchmarking.mmar.run_gemini_parallel --num-samples 1000

    # 8 workers for faster execution
    python -m benchmarking.mmar.run_gemini_parallel --num-samples 1000 --workers 8

    # Resume from previous run (auto-detects completed workers)
    python -m benchmarking.mmar.run_gemini_parallel --num-samples 1000 \
        --output-dir ./test_result/gemini_mmar_full
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
        help="Dataset directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to run (default: 1000)",
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
        help="File with specific indices to run (one per line)",
    )

    # Parallel execution configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cpfs/user/jingpeng/workspace/AUDIO_AGENT/test_result/gemini_mmar_full",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N samples per worker (default: 5)",
    )
    parser.add_argument(
        "--worker-log-dir",
        type=str,
        default=None,
        help="Directory for worker stdout/stderr logs (default: {output_dir}/worker_logs)",
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
    parser.add_argument(
        "--use-dual-frontend",
        action="store_true",
        help="Enable dual frontend calls (verifier caption + observer direct answer)",
    )
    return parser


def split_indices(indices: list[int], n_workers: int) -> list[list[int]]:
    """Split indices into N roughly equal batches."""
    if n_workers <= 0:
        n_workers = 1
    batch_size = max(1, len(indices) // n_workers)
    batches = []
    for i in range(n_workers):
        start = i * batch_size
        if i == n_workers - 1:
            end = len(indices)
        else:
            end = min(start + batch_size, len(indices))
        if start < end:
            batches.append(indices[start:end])
    return batches


def get_worker_paths(output_dir: Path, worker_id: int) -> dict[str, Path]:
    """Get all file paths for a worker."""
    worker_dir = output_dir / f"worker_{worker_id}"
    return {
        "worker_dir": worker_dir,
        "log_dir": worker_dir / "logs",
        "checkpoint": worker_dir / "checkpoint.json",
        "results": worker_dir / "results.json",
        "report": worker_dir / "report.txt",
    }


async def run_worker(
    worker_id: int,
    indices: list[int],
    args: argparse.Namespace,
    log_file_path: Path,
):
    """
    Worker main coroutine: initialize runner, process batch, save results.
    MCP tools are launched once at the start and reused for all samples.
    """
    output_dir = Path(args.output_dir)
    paths = get_worker_paths(output_dir, worker_id)
    paths["worker_dir"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)

    # Redirect stdout/stderr to worker log file
    log_fh = open(log_file_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_fh
    sys.stderr = log_fh

    try:
        print(f"\n{'='*60}")
        print(f"  Worker {worker_id} started")
        print(f"  Samples: {len(indices)} (index {indices[0]} to {indices[-1]})")
        print(f"{'='*60}\n")

        # Load checkpoint if exists
        completed_indices: set[int] = set()
        results: list[dict] = []

        if paths["checkpoint"].exists():
            print(f"[Worker {worker_id}] Resuming from checkpoint...")
            completed_indices, results = load_checkpoint(str(paths["checkpoint"]))
            original_count = len(indices)
            indices = [idx for idx in indices if idx not in completed_indices]
            print(
                f"[Worker {worker_id}] {len(results)} done, "
                f"{len(indices)} remaining (skipped {original_count - len(indices)})"
            )
            if not indices:
                print(f"[Worker {worker_id}] All samples already completed!")
                return

        # Initialize benchmark (for metadata formatting)
        benchmark = MMARBenchmark(
            dataset_dir=args.dataset_dir,
            split="test",
        )
        dataset = benchmark.load_dataset()

        # Initialize runner (starts MCP servers once)
        print(f"[Worker {worker_id}] Initializing runner (launching MCP tools)...")
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
            log_dir=str(paths["log_dir"]),
            use_gemini=True,
            gemini_api_key=args.gemini_api_key,
            gemini_base_url=args.gemini_base_url,
            use_dual_frontend=args.use_dual_frontend,
        )

        async with runner:
            print(f"[Worker {worker_id}] Runner ready. Processing samples...")
            start_time = time.time()
            last_checkpoint_time = start_time
            
            # Deferred samples queue (for token limit / rate limit retries)
            deferred_samples: list[tuple[int, dict, str, list[str]]] = []

            for i, idx in enumerate(indices):
                sample = dataset[idx]
                metadata = benchmark.get_sample_metadata(sample)
                sample_start = time.time()

                # Format question
                try:
                    question, audio_paths = benchmark.format_question(sample)
                except Exception as e:
                    print(f"\n[Worker {worker_id}] Error formatting sample {idx}: {e}")
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
                        print(f"\n[Worker {worker_id}] Error preprocessing audio for sample {idx}: {e}")
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
                        print(f"\n[Worker {worker_id}][Deferred] Sample {idx} queued for retry (token limit)")
                        continue
                    
                    prediction = result.get("answer")

                    # Evaluate
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

                    sample_elapsed = time.time() - sample_start
                    correct_flag = "✓" if eval_result.get("correct") else "✗"
                    print(
                        f"[Worker {worker_id}] {i+1}/{len(indices)} "
                        f"idx={idx} {correct_flag} "
                        f"pred={prediction!r} gt={metadata['ground_truth']!r} "
                        f"steps={result.get('step_count')} "
                        f"time={sample_elapsed:.1f}s"
                    )

                except Exception as e:
                    if "10001" in str(e):
                        deferred_samples.append((idx, metadata, question, audio_paths))
                        print(f"\n[Worker {worker_id}][Deferred] Sample {idx} queued for retry (token limit)")
                        continue
                    print(f"\n[Worker {worker_id}] Error running sample {idx}: {e}")
                    results.append({
                        **metadata,
                        "index": idx,
                        "prediction": None,
                        "correct": False,
                        "match_method": "runtime_error",
                        "error": str(e),
                    })

                completed_indices.add(idx)

                # Save checkpoint periodically
                now = time.time()
                if (
                    len(completed_indices) % args.checkpoint_interval == 0
                    or now - last_checkpoint_time > 300  # also every 5 min
                ):
                    save_checkpoint(
                        completed_indices, results, str(paths["checkpoint"])
                    )
                    last_checkpoint_time = now

            # ── Multi-round Deferred Retry Phase ──
            MAX_DEFERRED_ROUNDS = 3
            round_num = 1
            while deferred_samples and round_num <= MAX_DEFERRED_ROUNDS:
                print(f"\n[Worker {worker_id}] {'='*60}")
                print(f"[Worker {worker_id}] Deferred Retry Round {round_num}/{MAX_DEFERRED_ROUNDS}")
                print(f"[Worker {worker_id}] {len(deferred_samples)} samples to retry")
                print(f"[Worker {worker_id}] Cooling down for 10s...")
                print(f"[Worker {worker_id}] {'='*60}")
                await asyncio.sleep(10)
                
                still_failed: list[tuple[int, dict, str, list[str]]] = []
                
                for idx, metadata, question, audio_paths in deferred_samples:
                    print(f"\n[Worker {worker_id}][Deferred R{round_num}] Retrying sample {idx}...")
                    try:
                        result = await runner.run_single(
                            question=question,
                            audio_paths=audio_paths,
                            run_log_name=metadata["id"],
                        )
                        if result.get("status") == "error" and "10001" in str(result.get("error", "")):
                            print(f"[Worker {worker_id}][Deferred R{round_num}] Sample {idx} still token-limited")
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
                            print(f"[Worker {worker_id}][Deferred R{round_num}] Sample {idx} retried {status_str}")
                    except Exception as e:
                        print(f"\n[Worker {worker_id}][Deferred R{round_num}] Error running sample {idx}: {e}")
                        results.append({
                            **metadata,
                            "index": idx,
                            "prediction": None,
                            "correct": False,
                            "match_method": "runtime_error",
                            "error": str(e),
                        })
                    completed_indices.add(idx)
                    save_checkpoint(completed_indices, results, str(paths["checkpoint"]))
                
                deferred_samples = still_failed
                if deferred_samples:
                    print(f"\n[Worker {worker_id}][Deferred R{round_num}] {len(deferred_samples)} samples still failed")
                round_num += 1
            
            if deferred_samples:
                print(f"\n[Worker {worker_id}] {len(deferred_samples)} samples exhausted all {MAX_DEFERRED_ROUNDS} rounds")
            else:
                print(f"\n[Worker {worker_id}] Deferred retry completed. All samples resolved.")
            
            # Final timing
            elapsed = time.time() - start_time
            correct_count = sum(1 for r in results if r.get("correct", False))
            print(f"\n[Worker {worker_id}] Batch completed in {format_duration(elapsed)}")
            print(
                f"[Worker {worker_id}] Correct: {correct_count} / {len(results)} "
                f"({correct_count/len(results)*100:.1f}%)"
            )

        # Compute metrics
        metrics = benchmark.compute_metrics(results)

        # Save results
        save_results(results, str(paths["results"]), metrics)
        save_metrics_report(
            metrics,
            str(paths["report"]),
            title=f"MMAR Benchmark Results - Worker {worker_id}\n"
            f"  Frontend: gemini-2.5-pro  Planner: {args.planner_model} ({args.planner_backend})",
        )

        # Final checkpoint
        save_checkpoint(completed_indices, results, str(paths["checkpoint"]))

        print(f"[Worker {worker_id}] Results saved to {paths['worker_dir']}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fh.close()


def worker_entrypoint(worker_id: int, indices: list[int], args: argparse.Namespace):
    """Entry point for each worker process."""
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


def monitor_progress(output_dir: Path, n_workers: int, total: int) -> dict:
    """
    Check all worker checkpoints and return aggregate progress.
    Returns dict with completed count and per-worker status.
    """
    completed_total = 0
    worker_status = {}

    for wid in range(n_workers):
        paths = get_worker_paths(output_dir, wid)
        checkpoint_path = paths["checkpoint"]
        if checkpoint_path.exists():
            try:
                completed_indices, _ = load_checkpoint(str(checkpoint_path))
                count = len(completed_indices)
                completed_total += count
                worker_status[wid] = count
            except Exception:
                worker_status[wid] = 0
        else:
            worker_status[wid] = 0

    return {
        "completed": completed_total,
        "total": total,
        "percent": completed_total / total * 100 if total > 0 else 0,
        "workers": worker_status,
    }


def merge_results(output_dir: Path, n_workers: int) -> dict:
    """Merge results from all workers into a single report."""
    all_results = []

    for wid in range(n_workers):
        paths = get_worker_paths(output_dir, wid)
        # Try results.json first, fall back to checkpoint.json
        results_path = paths["results"]
        checkpoint_path = paths["checkpoint"]
        worker_results = []

        if results_path.exists():
            try:
                data = load_results(str(results_path))
                worker_results = data.get("results", [])
            except Exception as e:
                print(f"Warning: Could not load worker {wid} results: {e}")

        # If no results.json or it's empty, try checkpoint.json
        if not worker_results and checkpoint_path.exists():
            try:
                completed_indices, checkpoint_results = load_checkpoint(str(checkpoint_path))
                worker_results = checkpoint_results
                print(f"  Worker {wid}: using checkpoint ({len(checkpoint_results)} samples)")
            except Exception as e:
                print(f"Warning: Could not load worker {wid} checkpoint: {e}")

        all_results.extend(worker_results)

    if not all_results:
        print("Warning: No results found from any worker!")
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    # Sort by index
    all_results.sort(key=lambda r: r.get("index", 0))

    # Compute metrics
    benchmark = MMARBenchmark()
    metrics = benchmark.compute_metrics(all_results)

    # Save merged results
    save_results(all_results, str(output_dir / "results.json"), metrics)
    save_metrics_report(
        metrics,
        str(output_dir / "report.txt"),
        title="MMAR Benchmark Results\n  Frontend: gemini-2.5-pro  Planner: qwen3.5-plus",
    )

    return metrics


def run_benchmark(args: argparse.Namespace):
    """Main orchestrator: split work, launch workers, monitor, merge."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MMAR Benchmark - Gemini 2.5 Pro Frontend (Parallel)")
    print("=" * 70)
    print(f"\n  Planner:  {args.planner_model}")
    print(f"  Planner Backend: {args.planner_backend}")
    print(f"  Workers:  {args.workers}")
    print(f"  Output:   {args.output_dir}")
    print()

    # Load dataset to determine range
    benchmark = MMARBenchmark(dataset_dir=args.dataset_dir, split="test")
    dataset = benchmark.load_dataset()
    total_samples = len(dataset)

    start_idx = args.start_idx
    end_idx = min(args.start_idx + args.num_samples, total_samples)
    target_indices = list(range(start_idx, end_idx))
    print(f"Total samples to process: {len(target_indices)} (index {start_idx} to {end_idx-1})")

    # Split into batches
    batches = split_indices(target_indices, args.workers)
    actual_workers = len(batches)
    print(f"Split into {actual_workers} batches")
    for i, batch in enumerate(batches):
        print(f"  Worker {i}: {len(batch)} samples (index {batch[0]} to {batch[-1]})")

    # Check which workers are already done (resume support)
    workers_to_run = []
    for wid, batch in enumerate(batches):
        paths = get_worker_paths(output_dir, wid)
        # Check if this worker has a checkpoint covering all its indices
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

    # Use spawn method for multiprocessing (safer with asyncio)
    multiprocessing.set_start_method("spawn", force=True)

    # Run benchmark
    run_benchmark(args)


if __name__ == "__main__":
    main()
