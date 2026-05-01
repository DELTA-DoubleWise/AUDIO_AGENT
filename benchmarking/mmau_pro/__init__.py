"""
MMAU-Pro Benchmark Implementation.

This module provides the benchmark implementation for MMAU-Pro dataset.

Usage:
    from benchmarking.mmau_pro import MMAUProBenchmark
    
    benchmark = MMAUProBenchmark(dataset_dir="/path/to/dataset")
    dataset = benchmark.load_dataset()
"""

from benchmarking.mmau_pro.dataset import MMAUProBenchmark

__all__ = ["MMAUProBenchmark"]
