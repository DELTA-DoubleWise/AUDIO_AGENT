#!/usr/bin/env python3
"""
Test script for MMAU-Pro benchmark without API calls.

This tests the benchmark pipeline without making actual API calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "AUDIO_AGENT"))

from benchmarking.mmau_pro.dataset import MMAUProBenchmark
from benchmarking.metrics import choice_match


def test_benchmark():
    """Test the MMAU-Pro benchmark implementation."""
    print("=" * 60)
    print("  MMAU-Pro Benchmark Test")
    print("=" * 60)
    print()
    
    # Initialize benchmark
    print("1. Initializing benchmark...")
    benchmark = MMAUProBenchmark(dataset_dir="/lihaoyu/datasets/MMAU-Pro")
    dataset = benchmark.load_dataset()
    print(f"   Loaded {len(dataset)} samples")
    print()
    
    # Test formatting
    print("2. Testing question formatting...")
    sample = dataset[0]
    question, audio_paths = benchmark.format_question(sample)
    print(f"   Sample ID: {sample['id']}")
    print(f"   Question length: {len(question)} chars")
    print(f"   Audio paths: {audio_paths}")
    print()
    
    # Test evaluation
    print("3. Testing evaluation...")
    
    # Test correct answer
    prediction = sample["answer"]
    ground_truth = sample["answer"]
    choices = sample.get("choices", [])
    
    result = benchmark.evaluate_answer(prediction, ground_truth, choices=choices)
    print(f"   Correct prediction test: {result}")
    assert result["correct"] == True, "Should match exact answer"
    
    # Test wrong answer
    if len(choices) > 1:
        wrong_answer = [c for c in choices if c != ground_truth][0]
        result = benchmark.evaluate_answer(wrong_answer, ground_truth, choices=choices)
        print(f"   Wrong prediction test: {result}")
        assert result["correct"] == False, "Should not match wrong answer"
    
    # Test with extra text
    prediction_with_extra = f"The answer is {ground_truth}"
    result = benchmark.evaluate_answer(prediction_with_extra, ground_truth, choices=choices)
    print(f"   Extra text prediction test: {result}")
    assert result["correct"] == True, "Should match with contains"
    print()
    
    # Test metrics computation
    print("4. Testing metrics computation...")
    test_results = [
        {"correct": True, "category": "sound", "task_type": "sound"},
        {"correct": True, "category": "sound", "task_type": "sound"},
        {"correct": False, "category": "music", "task_type": "open"},
        {"correct": True, "category": "music", "task_type": "open"},
    ]
    metrics = benchmark.compute_metrics(test_results)
    print(f"   Overall accuracy: {metrics['accuracy']:.2%}")
    print(f"   By category: {metrics['by_category']}")
    print(f"   By task type: {metrics['by_task_type']}")
    assert metrics["accuracy"] == 0.75, "Should be 3/4 correct"
    print()
    
    # Test multiple samples
    print("5. Testing multiple samples...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        question, audio_paths = benchmark.format_question(sample)
        print(f"   Sample {i+1}: {sample['id'][:8]}... | Category: {sample.get('category', 'N/A')}")
    print()
    
    print("=" * 60)
    print("  All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmark()
