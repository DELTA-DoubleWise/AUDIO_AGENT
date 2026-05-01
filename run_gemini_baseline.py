#!/usr/bin/env python3
"""Run MMAR Benchmark with Gemini Direct (No Agent Flow).

This script sends audio + question directly to Gemini API and compares
the response against ground truth. No planner, no tools, no agent loop.

Usage:
    # Test run (10 samples)
    python -m benchmarking.mmar.run_gemini_baseline --num-samples 10

    # Full benchmark with 4 workers
    python -m benchmarking.mmar.run_gemini_baseline --num-samples 1000 --workers 4

    # Resume from checkpoint
    python -m benchmarking.mmar.run_gemini_baseline --resume-from test_result/gemini_baseline/checkpoint.json

    # Use Flash instead of Pro
    python -m benchmarking.mmar.run_gemini_baseline --gemini-model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap: pre-load benchmarking submodules to avoid triggering
# benchmarking/__init__.py (which imports langgraph-dependent code).
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent


def _load_module(mod_name: str, rel_path: str):
    """Load a module directly from file without triggering parent __init__.py."""
    path = _PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load pure-python submodules that dataset.py depends on.
_load_module("benchmarking.base", "benchmarking/base.py")
_load_module("benchmarking.metrics", "benchmarking/metrics.py")

# Dataset class will be loaded lazily based on --dataset-module argument.
BenchmarkClass = None

# ---------------------------------------------------------------------------
# Inline utils (avoid importing benchmarking.utils which goes through __init__)
# ---------------------------------------------------------------------------


def _save_results(results: list[dict], output_path: str, metrics: dict | None = None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(results),
        "results": results,
    }
    if metrics:
        data["metrics"] = metrics
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def _save_checkpoint(completed_indices: set[int], results: list[dict], checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "completed_indices": list(completed_indices),
        "num_completed": len(completed_indices),
        "results": results,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def _load_checkpoint(checkpoint_path: str) -> tuple[set[int], list[dict]]:
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    completed_indices = set(checkpoint.get("completed_indices", []))
    results = checkpoint.get("results", [])
    return completed_indices, results


def _format_metrics_report(metrics: dict, title: str = "Benchmark Results") -> str:
    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
        "",
    ]
    if "accuracy" in metrics:
        lines.append(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    if "correct" in metrics and "total" in metrics:
        lines.append(f"Correct: {metrics['correct']} / {metrics['total']}")
    if "failed" in metrics:
        lines.append(f"Failed: {metrics['failed']}")
    lines.append("")
    if "by_category" in metrics:
        lines.append("Accuracy by Category:")
        lines.append("-" * 40)
        for category, acc in sorted(metrics["by_category"].items()):
            lines.append(f"  {category:20s}: {acc:.2%}")
        lines.append("")
    if "by_task_type" in metrics:
        lines.append("Accuracy by Task Type:")
        lines.append("-" * 40)
        for task_type, acc in sorted(metrics["by_task_type"].items()):
            lines.append(f"  {task_type:20s}: {acc:.2%}")
        lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def _save_metrics_report(metrics: dict, output_path: str, title: str = "Benchmark Results"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = _format_metrics_report(metrics, title)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {output_path}")


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Gemini direct API
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert audio understanding assistant. "
    "Listen to the audio carefully and answer the question based only on what you hear. "
    "For multiple-choice questions, respond with exactly one of the provided options. "
    "Be concise and answer directly."
)


def _encode_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_payload(
    model: str,
    system_prompt: str,
    user_text: str,
    audio_paths: list[str],
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for path in audio_paths:
        audio_b64 = _encode_audio(path)
        parts.append({
            "inline_data": {
                "mime_type": "audio/wav",
                "data": audio_b64,
            }
        })
    parts.append({"text": user_text})

    return {
        "model": model,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"includeThoughts": True},
        },
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
    }


def _call_gemini(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 300,
    max_retries: int = 30,
    retry_interval: float = 0.5,
) -> str:
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }

    import requests
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()

            if "Code" in result and result["Code"] == 10001:
                last_error = RuntimeError(f"Gemini API token limit error (Code 10001)")
                if attempt < max_retries:
                    time.sleep(retry_interval)
                    continue
                raise last_error

            if "candidates" not in result or not result["candidates"]:
                raise RuntimeError(f"Gemini API returned no candidates: {result}")

            parts = result["candidates"][0]["content"]["parts"]

            if len(parts) > 1:
                text = parts[1].get("text", "")
            elif len(parts) > 0:
                text = parts[0].get("text", "")
            else:
                text = ""

            text = text.strip()
            if not text:
                raise RuntimeError("Gemini API returned empty text")

            return text

        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_interval)
                continue
            raise RuntimeError(f"Gemini API request failed after {max_retries + 1} retries: {e}") from e

    raise last_error or RuntimeError("Gemini API call exhausted all retries")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_main(
    wid: int,
    indices: list[int],
    args_dict: dict[str, Any],
    output_dir: Path,
):
    benchmark = args_dict["benchmark_class"](
        dataset_dir=args_dict["dataset_dir"],
        split="test",
    )
    dataset = benchmark.load_dataset()

    checkpoint_path = output_dir / f"worker_{wid}_checkpoint.json"
    completed: set[int] = set()
    results: list[dict] = []

    if checkpoint_path.exists():
        completed, results = _load_checkpoint(str(checkpoint_path))
        indices = [idx for idx in indices if idx not in completed]

    if not indices:
        return

    # Deferred retry queue: samples that failed with Code 10001
    deferred_samples: list[tuple[int, dict, str, list[str]]] = []

    def _process_sample(idx: int, metadata: dict, question: str, audio_paths: list[str]) -> dict | None:
        """Process a single sample. Returns result dict, or None if deferred."""
        try:
            payload = _build_payload(
                model=args_dict["gemini_model"],
                system_prompt=args_dict["system_prompt"],
                user_text=question,
                audio_paths=audio_paths,
                temperature=args_dict["temperature"],
                max_output_tokens=args_dict["max_output_tokens"],
            )

            prediction = _call_gemini(
                base_url=args_dict["gemini_base_url"],
                api_key=args_dict["gemini_api_key"],
                payload=payload,
                timeout=args_dict["timeout"],
            )

            eval_result = benchmark.evaluate_answer(
                prediction=prediction,
                ground_truth=metadata["ground_truth"],
                choices=metadata.get("choices", []),
            )

            return {
                **metadata,
                "index": idx,
                "prediction": prediction,
                **eval_result,
                "agent_status": "answered",
                "step_count": 1,
                "tool_calls": 0,
                "confidence": None,
                "agent_error": None,
            }

        except RuntimeError as e:
            if "10001" in str(e):
                # Token limit — defer for later retry
                return None
            # Other runtime error — record as failed
            return {
                **metadata,
                "index": idx,
                "prediction": None,
                "correct": False,
                "match_method": "runtime_error",
                "error": str(e),
            }
        except Exception as e:
            return {
                **metadata,
                "index": idx,
                "prediction": None,
                "correct": False,
                "match_method": "runtime_error",
                "error": str(e),
            }

    # Phase 1: Process all samples
    for idx in indices:
        sample = dataset[idx]
        metadata = benchmark.get_sample_metadata(sample)

        try:
            question, audio_paths = benchmark.format_question(sample)
        except Exception as e:
            results.append({
                **metadata,
                "index": idx,
                "prediction": None,
                "correct": False,
                "match_method": "format_error",
                "error": str(e),
            })
            completed.add(idx)
            continue

        result = _process_sample(idx, metadata, question, audio_paths)
        if result is None:
            # Deferred for retry
            deferred_samples.append((idx, metadata, question, audio_paths))
            print(f"[Worker {wid}] Deferred sample {idx} (token limit)")
        else:
            results.append(result)
            completed.add(idx)

        if len(completed) % args_dict["checkpoint_interval"] == 0:
            _save_checkpoint(completed, results, str(checkpoint_path))

    # Phase 2: Deferred retry (3 rounds with 10s cooldown)
    MAX_DEFERRED_ROUNDS = 3
    round_num = 1
    while deferred_samples and round_num <= MAX_DEFERRED_ROUNDS:
        print(f"\n[Worker {wid}] Deferred Retry Round {round_num}/{MAX_DEFERRED_ROUNDS}")
        print(f"[Worker {wid}] {len(deferred_samples)} samples to retry")
        print(f"[Worker {wid}] Cooling down for 10s...")
        time.sleep(10)

        still_deferred: list[tuple[int, dict, str, list[str]]] = []

        for idx, metadata, question, audio_paths in deferred_samples:
            print(f"[Worker {wid}] Retrying sample {idx}...")
            result = _process_sample(idx, metadata, question, audio_paths)
            if result is None:
                still_deferred.append((idx, metadata, question, audio_paths))
                print(f"[Worker {wid}] Sample {idx} still token-limited")
            else:
                results.append(result)
                completed.add(idx)
                print(f"[Worker {wid}] Sample {idx} succeeded")

            _save_checkpoint(completed, results, str(checkpoint_path))

        deferred_samples = still_deferred
        if deferred_samples:
            print(f"[Worker {wid}] {len(deferred_samples)} samples still deferred")
        round_num += 1

    if deferred_samples:
        print(f"[Worker {wid}] {len(deferred_samples)} samples exhausted all deferred rounds")
        # Record final failures
        for idx, metadata, question, audio_paths in deferred_samples:
            results.append({
                **metadata,
                "index": idx,
                "prediction": None,
                "correct": False,
                "match_method": "runtime_error",
                "error": "Token limit (Code 10001) exhausted all retries",
            })
            completed.add(idx)
        _save_checkpoint(completed, results, str(checkpoint_path))
    else:
        print(f"[Worker {wid}] All deferred samples resolved")

    _save_checkpoint(completed, results, str(checkpoint_path))


# ---------------------------------------------------------------------------
# Merge & monitor
# ---------------------------------------------------------------------------

def merge_worker_checkpoints(output_dir: Path, num_workers: int) -> tuple[set[int], list[dict]]:
    all_completed: set[int] = set()
    all_results: list[dict] = []

    for wid in range(num_workers):
        cp_path = output_dir / f"worker_{wid}_checkpoint.json"
        if cp_path.exists():
            completed, results = _load_checkpoint(str(cp_path))
            all_completed.update(completed)
            all_results.extend(results)

    seen: set[int] = set()
    deduped: list[dict] = []
    for r in reversed(all_results):
        idx = r["index"]
        if idx not in seen:
            seen.add(idx)
            deduped.append(r)
    deduped.reverse()

    return all_completed, deduped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MMAR Benchmark with Gemini Direct (No Agent Flow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (10 samples, 1 worker)
    python3 run_gemini_baseline.py --num-samples 10

    # Full benchmark with 4 workers
    python3 run_gemini_baseline.py --num-samples 1000 --workers 4

    # Resume
    python3 run_gemini_baseline.py --resume-from test_result/gemini_baseline/checkpoint.json

    # Use Flash
    python3 run_gemini_baseline.py --gemini-model gemini-2.5-flash
        """,
    )

    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro)",
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
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for Gemini",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Max output tokens (default: 4096)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="API request timeout in seconds (default: 300)",
    )

    parser.add_argument(
        "--dataset-module",
        type=str,
        default="mmar",
        choices=["mmar", "mmau"],
        help="Dataset module to use: mmar or mmau (default: mmar)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory (default: auto-detected based on dataset-module)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to run (default: 10)",
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
        default="./test_result/gemini_baseline",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N samples per worker (default: 10)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from merged checkpoint file",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Gemini API key required. Provide via --gemini-api-key or set GEMINI_API_KEY.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset class based on --dataset-module
    if args.dataset_module == "mmau":
        _dataset_mod = _load_module("benchmarking.mmau.dataset", "benchmarking/mmau/dataset.py")
        dataset_class = _dataset_mod.MMAUBenchmark
        default_dir = None
        benchmark_name = "MMAU"
    else:
        _dataset_mod = _load_module("benchmarking.mmar.dataset", "benchmarking/mmar/dataset.py")
        dataset_class = _dataset_mod.MMARBenchmark
        default_dir = None
        benchmark_name = "MMAR"

    dataset_dir = args.dataset_dir or default_dir

    print("=" * 60)
    print(f"  {benchmark_name} Benchmark - Gemini Direct Baseline")
    print("=" * 60)
    print(f"  Model:   {args.gemini_model}")
    print(f"  Workers: {args.workers}")
    print()

    benchmark = dataset_class(dataset_dir=dataset_dir, split="test")
    dataset = benchmark.load_dataset()
    total_samples = len(dataset)

    start_idx = args.start_idx
    end_idx = min(start_idx + args.num_samples, total_samples)
    target_indices = list(range(start_idx, end_idx))

    completed_indices: set[int] = set()
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from: {args.resume_from}")
        completed_indices, _ = _load_checkpoint(args.resume_from)
        original = len(target_indices)
        target_indices = [i for i in target_indices if i not in completed_indices]
        print(f"  Skipped {original - len(target_indices)} already completed")
        print(f"  Remaining: {len(target_indices)}")

    if not target_indices:
        print("All samples already completed!")
        return

    print(f"\nDataset: {total_samples} samples total")
    print(f"Running: {len(target_indices)} samples (index {target_indices[0]} to {target_indices[-1]})")

    num_workers = min(args.workers, len(target_indices))
    chunk_size = (len(target_indices) + num_workers - 1) // num_workers
    worker_indices: list[list[int]] = []
    for i in range(num_workers):
        chunk = target_indices[i * chunk_size:(i + 1) * chunk_size]
        if chunk:
            worker_indices.append(chunk)

    args_dict = {
        "gemini_model": args.gemini_model,
        "gemini_base_url": args.gemini_base_url,
        "gemini_api_key": api_key,
        "system_prompt": args.system_prompt,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "timeout": args.timeout,
        "dataset_dir": dataset_dir,
        "checkpoint_interval": args.checkpoint_interval,
        "benchmark_class": dataset_class,
    }

    print(f"\nLaunching {len(worker_indices)} workers...")
    processes: list[Process] = []
    for wid, indices in enumerate(worker_indices):
        print(f"  Worker {wid}: {len(indices)} samples")
        p = Process(target=worker_main, args=(wid, indices, args_dict, output_dir))
        p.start()
        processes.append(p)

    print("\nMonitoring progress...")
    pbar = tqdm(
        total=len(target_indices) + len(completed_indices),
        initial=len(completed_indices),
        desc="Processing",
    )

    last_seen = len(completed_indices)
    while any(p.is_alive() for p in processes):
        time.sleep(5)
        current_completed, current_results = merge_worker_checkpoints(output_dir, len(worker_indices))
        current_count = len(current_completed)
        if current_count > last_seen:
            pbar.update(current_count - last_seen)
            last_seen = current_count

            correct_so_far = sum(1 for r in current_results if r.get("correct"))
            acc = correct_so_far / len(current_results) if current_results else 0.0
            pbar.set_postfix({"acc": f"{acc:.2%}"})

    for p in processes:
        p.join()

    pbar.close()

    print("\nMerging worker checkpoints...")
    all_completed, all_results = merge_worker_checkpoints(output_dir, len(worker_indices))
    print(f"  Total completed: {len(all_completed)}")

    checkpoint_path = output_dir / "checkpoint.json"
    _save_checkpoint(all_completed, all_results, str(checkpoint_path))

    print("\nComputing metrics...")
    metrics = benchmark.compute_metrics(all_results)

    results_path = output_dir / "results.json"
    _save_results(all_results, str(results_path), metrics)

    report_path = output_dir / "report.txt"
    _save_metrics_report(
        metrics,
        str(report_path),
        title=f"{benchmark_name} Gemini Direct Baseline\n  Model: {args.gemini_model}",
    )

    print("\n" + "=" * 60)
    with open(report_path, "r") as f:
        print(f.read())

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
