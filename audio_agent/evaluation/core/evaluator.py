"""
Tool evaluator for running benchmarks and computing metrics.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from audio_agent.core.logging import get_logger
from audio_agent.core.schemas import ToolCallRequest, ToolResult
from audio_agent.evaluation.core.rps import RPSRegistry

logger = get_logger()


class ToolEvaluator:
    """
    Evaluates tools on benchmark datasets.
    
    Provides unified interface for:
    - Running tools on test samples
    - Computing evaluation metrics
    - Recording RPS scores
    """
    
    def __init__(
        self,
        registry: RPSRegistry | None = None,
        data_dir: Path | None = None,
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            registry: RPS registry for recording results
            data_dir: Directory containing benchmark data
        """
        self._registry = registry or RPSRegistry()
        self._data_dir = data_dir or Path("data")
    
    def evaluate_sample(
        self,
        tool,
        audio_path: Path,
        reference: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a tool on a single audio sample.
        
        Args:
            tool: Tool instance (with invoke method)
            audio_path: Path to audio file
            reference: Optional reference transcription
            
        Returns:
            Evaluation result dict
        """
        start_time = time.time()
        
        # Build tool call request
        request = ToolCallRequest(
            tool_name=tool.spec.name,
            args={"audio_path": str(audio_path)},
        )
        
        # Execute
        try:
            result: ToolResult = tool.invoke(request)
            latency = time.time() - start_time
            
            return {
                "success": result.success,
                "text": result.output.get("text", ""),
                "latency": latency,
                "error": result.error_message,
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "latency": time.time() - start_time,
                "error": str(e),
            }
    
    def evaluate_dataset(
        self,
        tool,
        dataset: str,
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a tool on a dataset.
        
        Args:
            tool: Tool instance
            dataset: Dataset name (e.g., "aishell1")
            max_samples: Maximum samples to evaluate
            
        Returns:
            Evaluation summary
        """
        logger.info(f"Evaluating {tool.spec.name} on {dataset}")
        
        # Find dataset
        dataset_dir = self._find_dataset_dir(dataset)
        if not dataset_dir:
            return {"error": f"Dataset {dataset} not found"}
        
        # Find audio files and references
        samples = self._load_dataset_samples(dataset_dir)
        if max_samples:
            samples = samples[:max_samples]
        
        # Evaluate each sample
        results = []
        for audio_path, reference in samples:
            result = self.evaluate_sample(tool, audio_path, reference)
            results.append(result)
        
        # Compute metrics
        successful = [r for r in results if r["success"]]
        avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0
        
        summary = {
            "tool": tool.spec.name,
            "dataset": dataset,
            "total": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "avg_latency": avg_latency,
            "results": results,
        }
        
        logger.info(
            f"Evaluation complete: {len(successful)}/{len(results)} successful, "
            f"avg latency: {avg_latency:.2f}s"
        )
        
        return summary
    
    def record_metric(
        self,
        tool_name: str,
        dataset: str,
        score: float,
        metric: str | None = None,
    ) -> None:
        """
        Record a metric and compute RPS.
        
        Args:
            tool_name: Tool name
            dataset: Dataset name
            score: Score value
            metric: Metric type
        """
        self._registry.record(tool_name, dataset, score, metric)
    
    def recommend_tool(self, dataset: str) -> str | None:
        """
        Recommend best tool for a dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Recommended tool name
        """
        return self._registry.get_best_tool(dataset)
    
    def _find_dataset_dir(self, dataset: str) -> Path | None:
        """Find dataset directory."""
        for track in ["asr", "multitask"]:
            path = self._data_dir / track / dataset
            if path.exists():
                return path
        return None
    
    def _load_dataset_samples(
        self, 
        dataset_dir: Path,
    ) -> list[tuple[Path, str | None]]:
        """
        Load audio samples from dataset directory.
        
        Returns:
            List of (audio_path, reference) tuples
        """
        samples = []
        gt_file = dataset_dir / "gt.jsonl"
        
        if gt_file.exists():
            # Load from ground truth file
            with open(gt_file) as f:
                for line in f:
                    import json
                    data = json.loads(line)
                    audio_path = dataset_dir / data["path"]
                    reference = data.get("target")
                    if audio_path.exists():
                        samples.append((audio_path, reference))
        else:
            # Fallback: scan directory
            for ext in ["*.wav", "*.mp3", "*.flac"]:
                for audio_path in dataset_dir.rglob(ext):
                    samples.append((audio_path, None))
        
        return samples
