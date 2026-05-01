"""
MMAU-Pro Dataset Implementation.

Provides the MMAUProBenchmark class for loading and evaluating on MMAU-Pro dataset.
"""

import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from benchmarking.base import BaseBenchmark
from benchmarking.metrics import choice_match, normalize_answer


class MMAUProBenchmark(BaseBenchmark):
    """
    MMAU-Pro benchmark implementation.
    
    MMAU-Pro is a comprehensive benchmark for audio understanding with:
    - 5,305 question-answer pairs
    - Multiple choice and open-ended questions
    - Categories: sound, music, speech
    - Task types: sound (MCQ), open (open-ended)
    
    Example:
        benchmark = MMAUProBenchmark(
            dataset_dir="/lihaoyu/datasets/MMAU-Pro",
            split="test"
        )
        
        # Get formatted question
        sample = benchmark.get_sample(0)
        question, audio_paths = benchmark.format_question(sample)
        
        # Evaluate prediction
        result = benchmark.evaluate_answer(
            prediction="Boba tea",
            ground_truth=sample["answer"],
            choices=sample.get("choices", []),
        )
    """
    
    DATASET_NAME = "gamma-lab-umd/MMAU-Pro"
    
    def __init__(
        self,
        dataset_dir: str | None = None,
        split: str = "test",
        cache_dir: str | None = None,
    ):
        """
        Initialize MMAU-Pro benchmark.
        
        Args:
            dataset_dir: Directory for dataset cache (audio files will be here).
            split: Dataset split to use (default: "test").
            cache_dir: Cache directory for HuggingFace datasets.
        """
        super().__init__(dataset_dir=dataset_dir)
        self.split = split
        self.cache_dir = cache_dir
        self._audio_base_path: Path | None = None
    
    def load_dataset(self) -> Dataset:
        """
        Load the MMAU-Pro dataset.
        
        First tries to load from local metadata directory, then falls back
        to HuggingFace Hub.
        
        Returns:
            HuggingFace Dataset object.
        """
        print(f"Loading MMAU-Pro dataset (split: {self.split})...")
        
        # Try loading from local metadata directory first
        if self.dataset_dir:
            local_metadata_path = Path(self.dataset_dir) / "metadata"
            if local_metadata_path.exists():
                print(f"Loading from local metadata: {local_metadata_path}")
                dataset = load_from_disk(str(local_metadata_path))
                print(f"Loaded {len(dataset)} samples from local storage")
                self._audio_base_path = Path(self.dataset_dir)
                self._dataset = dataset
                return dataset
        
        # Fall back to HuggingFace
        print(f"Loading from HuggingFace Hub: {self.DATASET_NAME}")
        dataset = load_dataset(
            self.DATASET_NAME,
            split=self.split,
            cache_dir=self.cache_dir,
        )
        
        print(f"Loaded {len(dataset)} samples")
        
        # Determine audio base path
        if self.dataset_dir:
            self._audio_base_path = Path(self.dataset_dir)
        else:
            # Use HuggingFace cache directory
            self._audio_base_path = Path(dataset.cache_files[0]["filename"]).parent
        
        self._dataset = dataset
        return dataset
    
    def format_question(self, sample: dict) -> tuple[str, list[str]]:
        """
        Format a dataset sample into agent inputs.
        
        For MCQ questions, formats as:
            {question}
            
            Options:
            A. {choice_0}
            B. {choice_1}
            ...
            
            Please answer with exactly one of the options above.
        
        Args:
            sample: A dataset sample.
            
        Returns:
            Tuple of (formatted_question, audio_paths).
        """
        question = sample["question"]
        choices = sample.get("choices", [])
        
        # Format MCQ with choices
        if choices and len(choices) > 0:
            question = self._format_mcq(question, choices)
        
        # Resolve audio paths
        audio_paths = sample.get("audio_path", [])
        if not audio_paths:
            raise ValueError(f"No audio paths found for sample {sample.get('id')}")
        
        # Resolve relative paths to absolute paths
        resolved_paths = []
        for path in audio_paths:
            if not os.path.isabs(path) and self._audio_base_path:
                full_path = self._audio_base_path / path
                
                # Download audio file if it doesn't exist locally
                if not full_path.exists():
                    self._download_audio_file(path, full_path)
                
                path = str(full_path)
            resolved_paths.append(path)
        
        return question, resolved_paths
    
    def _download_audio_file(self, audio_path: str, local_path: Path):
        """
        Download an audio file from HuggingFace data.zip if not available locally.
        
        The MMAU-Pro dataset stores audio files in a large data.zip file.
        This method extracts individual files on-demand.
        
        Args:
            audio_path: Relative path to audio file (e.g., "data/xxx.wav").
            local_path: Full local path where file should be saved.
        """
        from huggingface_hub import hf_hub_download
        from tqdm import tqdm
        import zipfile
        import shutil
        
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get filename
        filename = audio_path.replace("data/", "") if audio_path.startswith("data/") else audio_path
        
        print(f"\nDownloading audio: {filename}")
        print("Note: Audio files are in data.zip. Downloading may take a moment...")
        
        try:
            # Download the data.zip file to a temporary location if not already present
            zip_cache_dir = Path(self.dataset_dir) / ".zip_cache"
            zip_cache_dir.mkdir(exist_ok=True)
            zip_path = zip_cache_dir / "data.zip"
            
            if not zip_path.exists():
                print(f"Downloading data.zip (this is a one-time ~47GB download)...")
                zip_path = hf_hub_download(
                    repo_id="gamma-lab-umd/MMAU-Pro",
                    filename="data.zip",
                    local_dir=zip_cache_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                zip_path = Path(zip_path)
            
            # Extract the specific file from the zip
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # The file is stored as "data/filename.wav" in the zip
                zip_internal_path = f"data/{filename}"
                
                # Extract to temp location then move
                temp_extract_path = zip_cache_dir / filename
                with zf.open(zip_internal_path) as source:
                    with open(temp_extract_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                
                # Move to final location
                shutil.move(temp_extract_path, local_path)
            
            print(f"Saved to: {local_path}")
            
        except Exception as e:
            print(f"Warning: Failed to download {filename}: {e}")
            print("You can pre-download all audio files with:")
            print("  python -m benchmarking.mmau_pro.download --download-audio")
            # Don't raise - let the agent handle the error if file is truly missing
            if local_path.exists():
                local_path.unlink()  # Clean up partial download
    
    def _format_mcq(self, question: str, choices: list[str]) -> str:
        """
        Format an MCQ question with choices.
        
        Args:
            question: The question text.
            choices: List of choice strings.
            
        Returns:
            Formatted question string.
        """
        lines = [question, "", "Options:"]
        
        # Add lettered choices (A, B, C, ...)
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A=65, B=66, etc.
            lines.append(f"{letter}. {choice}")
        
        lines.extend([
            "",
            "Please answer with exactly one of the options above (just the option text, not the letter).",
        ])
        
        return "\n".join(lines)
    
    def evaluate_answer(
        self,
        prediction: str | None,
        ground_truth: str,
        **kwargs
    ) -> dict:
        """
        Evaluate a prediction against ground truth.
        
        For MCQ questions, uses choice extraction and matching.
        For open-ended questions, uses normalized string matching.
        
        Args:
            prediction: The agent's predicted answer (or None if failed).
            ground_truth: The correct answer.
            **kwargs: Additional context (choices for MCQ).
            
        Returns:
            Dictionary with evaluation results.
        """
        choices = kwargs.get("choices", [])
        
        # Handle None prediction (agent failed)
        if prediction is None:
            return {
                "correct": False,
                "match_method": "no_prediction",
                "extracted_choice": None,
            }
        
        # MCQ evaluation with choices
        if choices and len(choices) > 0:
            return choice_match(prediction, ground_truth, choices)
        
        # Open-ended evaluation
        pred_norm = normalize_answer(prediction)
        target_norm = normalize_answer(ground_truth)
        
        # Try exact match first
        if pred_norm == target_norm:
            return {
                "correct": True,
                "match_method": "exact",
                "extracted_choice": prediction,
            }
        
        # Try contains match
        if target_norm in pred_norm or pred_norm in target_norm:
            return {
                "correct": True,
                "match_method": "contains",
                "extracted_choice": prediction,
            }
        
        # No match
        return {
            "correct": False,
            "match_method": "mismatch",
            "extracted_choice": prediction,
        }
    
    def compute_metrics(self, results: list[dict]) -> dict:
        """
        Compute aggregate metrics from evaluation results.
        
        Args:
            results: List of evaluation results.
            
        Returns:
            Dictionary with aggregate metrics including:
            - accuracy: Overall accuracy
            - correct: Number of correct predictions
            - total: Total number of samples
            - failed: Number of failed predictions
            - by_category: Accuracy broken down by category
            - by_task_type: Accuracy broken down by task type
        """
        if not results:
            return {"accuracy": 0.0, "correct": 0, "total": 0}
        
        total = len(results)
        correct = sum(1 for r in results if r.get("correct", False))
        failed = sum(1 for r in results if r.get("prediction") is None)
        
        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "failed": failed,
        }
        
        # Compute accuracy by category
        by_category = {}
        category_groups: dict[str, list] = {}
        
        for result in results:
            category = result.get("category", "unknown")
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(result)
        
        for category, group in category_groups.items():
            cat_correct = sum(1 for r in group if r.get("correct", False))
            by_category[category] = cat_correct / len(group) if group else 0.0
        
        metrics["by_category"] = by_category
        
        # Compute accuracy by task type
        by_task_type = {}
        task_groups: dict[str, list] = {}
        
        for result in results:
            task_type = result.get("task_type", "unknown")
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(result)
        
        for task_type, group in task_groups.items():
            task_correct = sum(1 for r in group if r.get("correct", False))
            by_task_type[task_type] = task_correct / len(group) if group else 0.0
        
        metrics["by_task_type"] = by_task_type
        
        return metrics
    
    def get_sample_metadata(self, sample: dict) -> dict:
        """
        Extract relevant metadata from a sample for result tracking.
        
        Args:
            sample: A dataset sample.
            
        Returns:
            Dictionary of metadata fields.
        """
        return {
            "id": sample.get("id"),
            "category": sample.get("category"),
            "task_type": sample.get("task_classification"),
            "length_type": sample.get("length_type"),
            "question": sample.get("question"),
            "ground_truth": sample.get("answer"),
            "choices": sample.get("choices", []),
        }
