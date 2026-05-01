"""
Abstract base class for benchmarks.

All benchmarks should inherit from BaseBenchmark and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any
from datasets import Dataset


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks.
    
    Subclasses must implement all abstract methods to define:
    - How to load the dataset
    - How to format questions for the agent
    - How to evaluate agent responses
    - How to compute aggregate metrics
    
    Example:
        class MyBenchmark(BaseBenchmark):
            def load_dataset(self) -> Dataset:
                return load_dataset("my-dataset", split="test")
            
            def format_question(self, sample: dict) -> tuple[str, list[str]]:
                return sample["question"], [sample["audio_path"]]
            
            def evaluate_answer(self, prediction: str, ground_truth: str, **kwargs) -> dict:
                correct = exact_match(prediction, ground_truth)
                return {"correct": correct, "match_type": "exact"}
            
            def compute_metrics(self, results: list[dict]) -> dict:
                accuracy = sum(r["correct"] for r in results) / len(results)
                return {"accuracy": accuracy}
    """
    
    def __init__(self, dataset_dir: str | None = None):
        """
        Initialize the benchmark.
        
        Args:
            dataset_dir: Directory where the dataset is stored or should be downloaded.
        """
        self.dataset_dir = dataset_dir
        self._dataset: Dataset | None = None
    
    @abstractmethod
    def load_dataset(self) -> Dataset:
        """
        Load and return the dataset.
        
        Returns:
            HuggingFace Dataset object containing all samples.
        """
        pass
    
    @abstractmethod
    def format_question(self, sample: dict) -> tuple[str, list[str]]:
        """
        Format a dataset sample into agent inputs.
        
        Args:
            sample: A single dataset sample (row from the dataset).
            
        Returns:
            Tuple of (question_text, audio_paths).
        """
        pass
    
    @abstractmethod
    def evaluate_answer(
        self, 
        prediction: str, 
        ground_truth: str, 
        **kwargs
    ) -> dict:
        """
        Evaluate a single prediction against ground truth.
        
        Args:
            prediction: The agent's predicted answer.
            ground_truth: The correct answer from the dataset.
            **kwargs: Additional context (e.g., choices for MCQ).
            
        Returns:
            Dictionary with evaluation results (e.g., {"correct": True, "method": "exact"}).
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, results: list[dict]) -> dict:
        """
        Compute aggregate metrics from evaluation results.
        
        Args:
            results: List of evaluation results from evaluate_answer().
            
        Returns:
            Dictionary of aggregate metrics (e.g., {"accuracy": 0.75}).
        """
        pass
    
    def get_dataset(self) -> Dataset:
        """
        Get the dataset, loading it if necessary.
        
        Returns:
            The loaded dataset.
        """
        if self._dataset is None:
            self._dataset = self.load_dataset()
        return self._dataset
    
    def get_sample(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            The sample at the given index.
        """
        dataset = self.get_dataset()
        return dataset[idx]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.get_dataset())
