"""
RPS (Relative Performance Score) calculation and management.

RPS measures tool performance relative to SOTA on benchmark datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_agent.core.logging import get_logger

logger = get_logger()


@dataclass
class SOTARecord:
    """Record of SOTA achievement for a dataset."""
    
    dataset: str
    score: float
    metric: str  # "cer", "wer", "accuracy", "bleu"
    higher_is_better: bool = False


@dataclass 
class ToolPerformance:
    """Performance record of a tool on a dataset."""
    
    tool_name: str
    dataset: str
    score: float
    metric: str
    rps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RPSCalculator:
    """
    Calculate RPS (Relative Performance Score).
    
    RPS = 1.0: Matches SOTA
    RPS > 1.0: Beats SOTA
    RPS < 1.0: Below SOTA
    """
    
    # Built-in SOTA benchmarks
    DEFAULT_SOTA = {
        # ASR Track
        "aishell1": SOTARecord("aishell1", 0.80, "cer", False),
        "aishell5": SOTARecord("aishell5", 24.74, "cer", False),
        "cs_dialogue": SOTARecord("cs_dialogue", 7.00, "mer", False),
        "kespeech": SOTARecord("kespeech", 3.81, "cer", False),
        "voxpopuli_en": SOTARecord("voxpopuli_en", 6.72, "wer", False),
        "contextasr_en": SOTARecord("contextasr_en", 3.47, "wer", False),
        "contextasr_zh": SOTARecord("contextasr_zh", 2.50, "cer", False),
        # Multitask Track
        "librispeech_clean": SOTARecord("librispeech_clean", 1.70, "wer", False),
        "librispeech_gr": SOTARecord("librispeech_gr", 92.02, "accuracy", True),
        "covost2_en2zh": SOTARecord("covost2_en2zh", 46.25, "bleu", True),
        "covost2_zh2en": SOTARecord("covost2_zh2en", 60.14, "bleu", True),
        "iemocap": SOTARecord("iemocap", 69.38, "accuracy", True),
        "mmsu_reason": SOTARecord("mmsu_reason", 89.07, "accuracy", True),
    }
    
    def __init__(self) -> None:
        """Initialize calculator with default SOTA values."""
        self._sota: dict[str, SOTARecord] = dict(self.DEFAULT_SOTA)
    
    def calculate(self, dataset: str, score: float) -> float | None:
        """
        Calculate RPS for a score on a dataset.
        
        Args:
            dataset: Dataset name
            score: Tool's score on the dataset
            
        Returns:
            RPS value or None if dataset not in SOTA
        """
        sota = self._sota.get(dataset)
        if not sota:
            logger.warning(f"No SOTA found for dataset: {dataset}")
            return None
        
        if sota.higher_is_better:
            rps = score / sota.score if sota.score > 0 else None
        else:
            rps = sota.score / score if score > 0 else None
        
        return rps
    
    def get_sota(self, dataset: str) -> SOTARecord | None:
        """Get SOTA record for a dataset."""
        return self._sota.get(dataset)
    
    def list_datasets(self) -> list[str]:
        """List all datasets with SOTA records."""
        return list(self._sota.keys())


class RPSRegistry:
    """
    Registry for tracking tool performances and RPS scores.
    
    Persists data to JSON for durability.
    """
    
    def __init__(self, db_path: Path | None = None) -> None:
        """
        Initialize registry.
        
        Args:
            db_path: Path to JSON database file
        """
        self._db_path = db_path or Path("evaluation_results.json")
        self._calculator = RPSCalculator()
        self._performances: dict[str, list[ToolPerformance]] = {}
        self._load()
    
    def record(
        self, 
        tool_name: str, 
        dataset: str, 
        score: float,
        metric: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolPerformance:
        """
        Record a tool's performance on a dataset.
        
        Args:
            tool_name: Name of the tool
            dataset: Dataset name
            score: Performance score
            metric: Metric type (auto-detected from dataset if not provided)
            metadata: Additional metadata
            
        Returns:
            ToolPerformance record with RPS
        """
        # Auto-detect metric from dataset
        if metric is None:
            sota = self._calculator.get_sota(dataset)
            metric = sota.metric if sota else "unknown"
        
        # Calculate RPS
        rps = self._calculator.calculate(dataset, score)
        
        record = ToolPerformance(
            tool_name=tool_name,
            dataset=dataset,
            score=score,
            metric=metric,
            rps=rps,
            metadata=metadata or {},
        )
        
        # Store
        key = f"{tool_name}:{dataset}"
        if key not in self._performances:
            self._performances[key] = []
        self._performances[key].append(record)
        
        # Persist
        self._save()
        
        rps_str = f"{rps:.3f}" if rps else "N/A"
        logger.info(
            f"Recorded: {tool_name} on {dataset}: "
            f"{metric}={score:.4f}, RPS={rps_str}"
        )
        
        return record
    
    def get_best_tool(self, dataset: str) -> str | None:
        """
        Get the best tool for a dataset based on RPS.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Name of best tool or None
        """
        best_tool = None
        best_rps = 0.0
        
        for key, performances in self._performances.items():
            tool_name = key.split(":")[0]
            for perf in performances:
                if perf.dataset == dataset and perf.rps is not None:
                    if perf.rps > best_rps:
                        best_rps = perf.rps
                        best_tool = tool_name
        
        return best_tool
    
    def get_tool_rps(self, tool_name: str, dataset: str) -> float | None:
        """Get RPS for a specific tool on a dataset."""
        key = f"{tool_name}:{dataset}"
        performances = self._performances.get(key, [])
        
        if performances:
            # Return most recent RPS
            return performances[-1].rps
        return None
    
    def list_tool_performances(self, tool_name: str) -> list[ToolPerformance]:
        """List all recorded performances for a tool."""
        results = []
        for key, performances in self._performances.items():
            if key.startswith(f"{tool_name}:"):
                results.extend(performances)
        return results
    
    def _load(self) -> None:
        """Load from JSON file."""
        if not self._db_path.exists():
            return
        
        try:
            with open(self._db_path, "r") as f:
                data = json.load(f)
            
            for key, records in data.items():
                self._performances[key] = [
                    ToolPerformance(**record) for record in records
                ]
            
            logger.info(f"Loaded {len(self._performances)} performance records")
        except Exception as e:
            logger.warning(f"Failed to load RPS registry: {e}")
    
    def _save(self) -> None:
        """Save to JSON file."""
        try:
            data = {
                key: [
                    {
                        "tool_name": p.tool_name,
                        "dataset": p.dataset,
                        "score": p.score,
                        "metric": p.metric,
                        "rps": p.rps,
                        "metadata": p.metadata,
                    }
                    for p in performances
                ]
                for key, performances in self._performances.items()
            }
            
            with open(self._db_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save RPS registry: {e}")
