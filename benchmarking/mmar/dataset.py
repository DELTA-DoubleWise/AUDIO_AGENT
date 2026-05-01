"""
MMAR Dataset Implementation.

Provides the MMARBenchmark class for loading and evaluating on MMAR dataset.
MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix.
"""

import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from benchmarking.base import BaseBenchmark
from benchmarking.metrics import choice_match


class MMARBenchmark(BaseBenchmark):
    """
    MMAR benchmark implementation.
    
    MMAR is a challenging benchmark for deep reasoning in speech, audio, music,
    and their mix. It contains 1,000 audio-question-answer triplets with
    multiple-choice questions requiring multi-step reasoning.
    
    Dataset: https://huggingface.co/datasets/BoJack/MMAR
    Paper: MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, 
           Music, and Their Mix (arXiv:2505.13032)
    
    Example:
        benchmark = MMARBenchmark(
            dataset_dir="/lihaoyu/datasets/MMAR",
            split="test"
        )
        
        # Get formatted question
        sample = benchmark.get_sample(0)
        question, audio_paths = benchmark.format_question(sample)
        
        # Evaluate prediction
        result = benchmark.evaluate_answer(
            prediction="Parrot",
            ground_truth=sample["answer"],
            choices=sample.get("choices", []),
        )
    """
    
    DATASET_NAME = "BoJack/MMAR"
    
    def __init__(
        self,
        dataset_dir: str | None = None,
        split: str = "test",
        cache_dir: str | None = None,
    ):
        """
        Initialize MMAR benchmark.
        
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
        Load the MMAR dataset.
        
        First tries to load from local metadata directory, then falls back
        to HuggingFace Hub.
        
        Returns:
            HuggingFace Dataset object.
        """
        print(f"Loading MMAR dataset (split: {self.split})...")
        
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
        
        # Resolve audio path
        audio_path = sample.get("audio_path", "")
        if not audio_path:
            raise ValueError(f"No audio path found for sample {sample.get('id')}")
        
        # Resolve relative path to absolute path
        # audio_path format: "./audio/filename.wav"
        resolved_paths = []
        if not os.path.isabs(audio_path) and self._audio_base_path:
            # Remove leading "./" if present
            if audio_path.startswith("./"):
                audio_path = audio_path[2:]
            
            full_path = self._audio_base_path / audio_path
            
            # Download audio file if it doesn't exist locally
            if not full_path.exists():
                self._download_audio_file(audio_path, full_path)
            
            audio_path = str(full_path)
        
        resolved_paths.append(audio_path)
        
        return question, resolved_paths
    
    def _download_audio_file(self, audio_path: str, local_path: Path):
        """
        Download an audio file from HuggingFace tar.gz archive if not available locally.
        
        The MMAR dataset stores audio files in mmar-audio.tar.gz.
        This method extracts individual files on-demand.
        
        Args:
            audio_path: Relative path to audio file (e.g., "audio/xxx.wav").
            local_path: Full local path where file should be saved.
        """
        import subprocess
        import tarfile
        import shutil
        
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get filename (remove audio/ prefix if present)
        filename = Path(audio_path).name
        
        print(f"\nDownloading audio: {filename}")
        print("Note: Audio files are in mmar-audio.tar.gz. Extracting...")
        
        try:
            # Download the tar.gz file to a temporary location if not already present
            tar_cache_dir = Path(self.dataset_dir) / ".tar_cache"
            tar_cache_dir.mkdir(exist_ok=True)
            tar_path = tar_cache_dir / "mmar-audio.tar.gz"
            
            if not tar_path.exists():
                print(f"Downloading mmar-audio.tar.gz (one-time ~3GB download)...")
                # Use hf CLI for better authentication handling
                result = subprocess.run(
                    [
                        "hf", "download", "BoJack/MMAR", "mmar-audio.tar.gz",
                        "--repo-type", "dataset",
                        "--local-dir", str(tar_cache_dir)
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                tar_path = tar_cache_dir / "mmar-audio.tar.gz"
            
            # Extract the specific file from the tar.gz
            with tarfile.open(tar_path, 'r:gz') as tf:
                # The file is stored as "audio/filename.wav" in the tar
                tar_internal_path = f"audio/{filename}"
                
                # Find the member
                member = None
                for m in tf.getmembers():
                    if m.name == tar_internal_path or m.name.endswith(filename):
                        member = m
                        break
                
                if member is None:
                    raise FileNotFoundError(f"File {filename} not found in archive")
                
                # Extract to temp location then move
                temp_extract_path = tar_cache_dir / filename
                with tf.extractfile(member) as source:
                    with open(temp_extract_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                
                # Move to final location
                shutil.move(temp_extract_path, local_path)
            
            print(f"Saved to: {local_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to download {filename}: {e}")
            print("Make sure you are authenticated with HuggingFace:")
            print("  hf auth login")
            # Don't raise - let the agent handle the error if file is truly missing
            if local_path.exists():
                local_path.unlink()  # Clean up partial download
        except Exception as e:
            print(f"Warning: Failed to extract {filename}: {e}")
            print("You can pre-download all audio files with:")
            print("  python -m benchmarking.mmar.download --download-audio")
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
        
        All MMAR questions are multiple choice, so this uses choice extraction
        and matching.
        
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
        
        # Fallback to exact match if no choices (shouldn't happen for MMAR)
        from benchmarking.metrics import normalize_answer
        pred_norm = normalize_answer(prediction)
        target_norm = normalize_answer(ground_truth)
        
        return {
            "correct": pred_norm == target_norm,
            "match_method": "exact_fallback",
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
            - by_modality: Accuracy broken down by modality
            - by_category: Accuracy broken down by category
            - by_sub_category: Accuracy broken down by sub-category
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
        
        # Compute accuracy by modality
        by_modality = {}
        modality_groups: dict[str, list] = {}
        
        for result in results:
            modality = result.get("modality", "unknown")
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(result)
        
        for modality, group in modality_groups.items():
            mod_correct = sum(1 for r in group if r.get("correct", False))
            by_modality[modality] = mod_correct / len(group) if group else 0.0
        
        metrics["by_modality"] = by_modality
        
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
        
        # Compute accuracy by sub-category
        by_sub_category = {}
        sub_category_groups: dict[str, list] = {}
        
        for result in results:
            sub_category = result.get("sub_category", "unknown")
            if sub_category not in sub_category_groups:
                sub_category_groups[sub_category] = []
            sub_category_groups[sub_category].append(result)
        
        for sub_category, group in sub_category_groups.items():
            sub_correct = sum(1 for r in group if r.get("correct", False))
            by_sub_category[sub_category] = sub_correct / len(group) if group else 0.0
        
        metrics["by_sub_category"] = by_sub_category
        
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
            "modality": sample.get("modality"),
            "category": sample.get("category"),
            "sub_category": sample.get("sub-category"),
            "language": sample.get("language"),
            "source": sample.get("source"),
            "question": sample.get("question"),
            "ground_truth": sample.get("answer"),
            "choices": sample.get("choices", []),
        }
