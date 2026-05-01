"""
Utility functions for benchmarking.

Provides helper functions for progress tracking, result serialization,
checkpoint management, and report generation.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def save_results(results: list[dict], output_path: str, metrics: dict | None = None):
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: List of result dictionaries.
        output_path: Path to save the results.
        metrics: Optional aggregate metrics to include.
    """
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


def load_results(results_path: str) -> dict:
    """
    Load benchmark results from a JSON file.
    
    Args:
        results_path: Path to the results file.
        
    Returns:
        Dictionary containing the loaded results.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    completed_indices: set[int],
    results: list[dict],
    checkpoint_path: str,
):
    """
    Save a checkpoint for resuming benchmark runs.
    
    Args:
        completed_indices: Set of indices that have been processed.
        results: List of results so far.
        checkpoint_path: Path to save the checkpoint.
    """
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


def load_checkpoint(checkpoint_path: str) -> tuple[set[int], list[dict]]:
    """
    Load a checkpoint for resuming benchmark runs.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        
    Returns:
        Tuple of (completed_indices, results).
    """
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    
    completed_indices = set(checkpoint.get("completed_indices", []))
    results = checkpoint.get("results", [])
    
    return completed_indices, results


def format_metrics_report(metrics: dict, title: str = "Benchmark Results") -> str:
    """
    Format metrics as a readable report string.
    
    Args:
        metrics: Dictionary of metrics.
        title: Title for the report.
        
    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
        "",
    ]
    
    # Overall metrics
    if "accuracy" in metrics:
        lines.append(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    if "correct" in metrics and "total" in metrics:
        lines.append(f"Correct: {metrics['correct']} / {metrics['total']}")
    if "failed" in metrics:
        lines.append(f"Failed: {metrics['failed']}")
    
    lines.append("")
    
    # Category breakdown
    if "by_category" in metrics:
        lines.append("Accuracy by Category:")
        lines.append("-" * 40)
        for category, acc in sorted(metrics["by_category"].items()):
            lines.append(f"  {category:20s}: {acc:.2%}")
        lines.append("")
    
    # Task type breakdown
    if "by_task_type" in metrics:
        lines.append("Accuracy by Task Type:")
        lines.append("-" * 40)
        for task_type, acc in sorted(metrics["by_task_type"].items()):
            lines.append(f"  {task_type:20s}: {acc:.2%}")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_metrics_report(metrics: dict, output_path: str, title: str = "Benchmark Results"):
    """
    Save a formatted metrics report to a text file.
    
    Args:
        metrics: Dictionary of metrics.
        output_path: Path to save the report.
        title: Title for the report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = format_metrics_report(metrics, title)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")


def compute_accuracy_by_key(
    results: list[dict],
    key: str,
    correct_key: str = "correct",
) -> dict[str, float]:
    """
    Compute accuracy grouped by a key.
    
    Args:
        results: List of result dictionaries.
        key: The key to group by (e.g., "category").
        correct_key: The key indicating correctness.
        
    Returns:
        Dictionary mapping group values to accuracy.
    """
    groups: dict[str, dict] = {}
    
    for result in results:
        group_value = result.get(key, "unknown")
        
        if group_value not in groups:
            groups[group_value] = {"correct": 0, "total": 0}
        
        groups[group_value]["total"] += 1
        if result.get(correct_key, False):
            groups[group_value]["correct"] += 1
    
    return {
        group: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for group, stats in groups.items()
    }


def format_duration(seconds: float) -> str:
    """
    Format seconds as a human-readable duration string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted string (e.g., "2h 15m 30s").
    """
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
