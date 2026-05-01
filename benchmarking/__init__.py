"""
Benchmarking framework for Audio Agent.

This package provides a reusable framework for evaluating audio agents
on various benchmarks.
"""

from benchmarking.base import BaseBenchmark
from benchmarking.runners import BenchmarkRunner
from benchmarking.metrics import (
    exact_match,
    contains_match,
    extract_choice,
    normalize_answer,
)
from benchmarking.utils import (
    save_results,
    load_results,
    save_checkpoint,
    load_checkpoint,
    format_metrics_report,
)

__all__ = [
    "BaseBenchmark",
    "BenchmarkRunner",
    "exact_match",
    "contains_match",
    "extract_choice",
    "normalize_answer",
    "save_results",
    "load_results",
    "save_checkpoint",
    "load_checkpoint",
    "format_metrics_report",
]
