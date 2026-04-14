"""
Qwen2.5 local-text planner implementation.

This planner follows the sample inference flow for Qwen2.5 Instruct:
- load model and tokenizer from transformers
- use BaseModelPlanner local-model input messages
- apply the tokenizer chat template
- generate text output

Output contracts remain the framework planner schemas:
- InitialPlan
- PlannerDecision

This adapter is intentionally constrained to:
- local text-only inference
- raw text output only
"""

from __future__ import annotations

from typing import Any

from audio_agent.core.errors import PlannerError
from audio_agent.planner.model_planner import (
    BaseModelPlanner,
    PlannerInputFormat,
    UnifiedPlannerInput,
)
from audio_agent.utils.model_downloader import DEFAULT_QWEN25_PATH


# Use local model path by default, fallback to HuggingFace Hub if not available
DEFAULT_QWEN25_MODEL_PATH = DEFAULT_QWEN25_PATH


class Qwen25Planner(BaseModelPlanner):
    """Qwen2.5-7B-Instruct planner adapter."""

    def __init__(
        self,
        model_path: str = DEFAULT_QWEN25_MODEL_PATH,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        generation_kwargs: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_config=model_config, max_retries=max_retries)

    @property
    def name(self) -> str:
        return "qwen25_planner"

    @property
    def input_format(self) -> PlannerInputFormat:
        return PlannerInputFormat.LOCAL_MODEL

    def initialize_model(self) -> dict[str, Any]:
        """Load Qwen2.5 model + tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise PlannerError(
                "Missing Qwen2.5 planner dependencies",
                details={
                    "required_packages": ["transformers", "torch"],
                    "hint": "Install model dependencies in your cluster environment",
                },
            ) from e

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            raise PlannerError(
                f"Failed to initialize Qwen2.5 model/tokenizer: {type(e).__name__}: {e}",
                details={"model_path": self.model_path},
            ) from e

        return {
            "model": model,
            "tokenizer": tokenizer,
        }

    def call_model(self, model_input: UnifiedPlannerInput) -> str:
        """Run Qwen2.5 generation and return raw text output."""
        model = self.model_handle["model"]
        tokenizer = self.model_handle["tokenizer"]
        messages = model_input.messages

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generate_kwargs = {"max_new_tokens": self.max_new_tokens}
            generate_kwargs.update(self.generation_kwargs)

            generated_ids = model.generate(**model_inputs, **generate_kwargs)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
        except Exception as e:
            raise PlannerError(
                f"Qwen2.5 inference failed: {type(e).__name__}: {e}",
                details={
                    "model_path": self.model_path,
                    "task_type": model_input.task_type,
                },
            ) from e

        if not response_texts:
            raise PlannerError("Qwen2.5 returned empty response list")

        raw_response = response_texts[0].strip()
        if not raw_response:
            raise PlannerError("Qwen2.5 returned empty response text")

        return raw_response
