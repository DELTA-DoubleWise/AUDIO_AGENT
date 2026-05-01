"""MMAU Dataset Implementation."""

import json
import os
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

from benchmarking.base import BaseBenchmark
from benchmarking.metrics import choice_match


class MMAUBenchmark(BaseBenchmark):
    """MMAU benchmark implementation."""

    def __init__(
        self,
        dataset_dir: str | None = None,
        split: str = "test",
        cache_dir: str | None = None,
    ):
        super().__init__(dataset_dir=dataset_dir)
        self.split = split
        self.cache_dir = cache_dir
        self._audio_base_path: Path | None = None

    def load_dataset(self) -> Dataset:
        print("Loading MMAU dataset...")
        if not self.dataset_dir:
            raise ValueError("dataset_dir is required for MMAU")
        meta_path = Path(self.dataset_dir) / "MMAU-meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"MMAU metadata not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize mixed-type fields before creating Dataset
        for item in data:
            cat = item.get("category")
            if isinstance(cat, list):
                item["category"] = cat[0] if cat else "unknown"
            elif isinstance(cat, str) and cat.startswith("["):
                try:
                    parsed = json.loads(cat.replace("'", '"'))
                    item["category"] = parsed[0] if parsed else "unknown"
                except Exception:
                    pass

        dataset = Dataset.from_list(data)
        print(f"Loaded {len(dataset)} samples")
        self._audio_base_path = Path(self.dataset_dir)
        self._dataset = dataset
        return dataset

    def format_question(self, sample: dict) -> tuple[str, list[str]]:
        question = sample["question"]
        choices = sample.get("choices", [])
        if choices and len(choices) > 0:
            clean_choices = [self._clean_choice(c) for c in choices]
            question = self._format_mcq(question, clean_choices)
        audio_path = sample.get("audio_path", "")
        if not audio_path:
            raise ValueError(f"No audio path found for sample {sample.get('id')}")
        resolved_paths = []
        if not os.path.isabs(audio_path) and self._audio_base_path:
            if audio_path.startswith("./"):
                audio_path = audio_path[2:]
            full_path = self._audio_base_path / audio_path
            audio_path = str(full_path)
        resolved_paths.append(audio_path)
        return question, resolved_paths

    @staticmethod
    def _clean_choice(choice: str) -> str:
        return re.sub(r'^\([A-Da-d]\)\s*', '', choice).strip()

    def _format_mcq(self, question: str, choices: list[str]) -> str:
        lines = [question, "", "Options:"]
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            lines.append(f"{letter}. {choice}")
        lines.extend(["", "Please answer with exactly one of the options above (just the option text, not the letter)."])
        return "\n".join(lines)

    def evaluate_answer(self, prediction: str | None, ground_truth: str, **kwargs) -> dict:
        raw_choices = kwargs.get("choices", [])
        if prediction is None:
            return {"correct": False, "match_method": "no_prediction", "extracted_choice": None}
        if raw_choices and len(raw_choices) > 0:
            clean_choices = [self._clean_choice(c) for c in raw_choices]
            clean_gt = self._clean_choice(ground_truth)
            return choice_match(prediction, clean_gt, clean_choices)
        from benchmarking.metrics import normalize_answer
        pred_norm = normalize_answer(prediction)
        target_norm = normalize_answer(ground_truth)
        return {"correct": pred_norm == target_norm, "match_method": "exact_fallback", "extracted_choice": prediction}

    def compute_metrics(self, results: list[dict]) -> dict:
        if not results:
            return {"accuracy": 0.0, "correct": 0, "total": 0}
        total = len(results)
        correct = sum(1 for r in results if r.get("correct", False))
        failed = sum(1 for r in results if r.get("prediction") is None)
        metrics = {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total, "failed": failed}
        by_modality = {}
        modality_groups: dict[str, list] = {}
        for result in results:
            modality = result.get("modality", "unknown")
            modality_groups.setdefault(modality, []).append(result)
        for modality, group in modality_groups.items():
            mod_correct = sum(1 for r in group if r.get("correct", False))
            by_modality[modality] = mod_correct / len(group) if group else 0.0
        metrics["by_modality"] = by_modality
        by_category = {}
        category_groups: dict[str, list] = {}
        for result in results:
            category = result.get("category", "unknown")
            category_groups.setdefault(category, []).append(result)
        for category, group in category_groups.items():
            cat_correct = sum(1 for r in group if r.get("correct", False))
            by_category[category] = cat_correct / len(group) if group else 0.0
        metrics["by_category"] = by_category
        by_sub_category = {}
        sub_category_groups: dict[str, list] = {}
        for result in results:
            sub_category = result.get("sub_category", "unknown")
            sub_category_groups.setdefault(sub_category, []).append(result)
        for sub_category, group in sub_category_groups.items():
            sub_correct = sum(1 for r in group if r.get("correct", False))
            by_sub_category[sub_category] = sub_correct / len(group) if group else 0.0
        metrics["by_sub_category"] = by_sub_category
        return metrics

    def get_sample_metadata(self, sample: dict) -> dict:
        raw_category = sample.get("category", "unknown")
        if isinstance(raw_category, list):
            category = raw_category[0] if raw_category else "unknown"
        elif isinstance(raw_category, str) and raw_category.startswith("["):
            try:
                parsed = json.loads(raw_category.replace("'", '"'))
                category = parsed[0] if parsed else "unknown"
            except Exception:
                category = raw_category
        else:
            category = raw_category
        return {
            "id": sample.get("id"),
            "modality": sample.get("modality"),
            "category": category,
            "sub_category": sample.get("sub-category", "unknown"),
            "language": sample.get("language"),
            "source": sample.get("source"),
            "question": sample.get("question"),
            "ground_truth": sample.get("answer"),
            "choices": sample.get("choices", []),
        }
