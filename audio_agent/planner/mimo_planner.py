"""MiMo API planner implementation.

Uses Xiaomi MiMo's OpenAI-compatible API for planning and decision making.
Supports mimo-v2.5-pro and other MiMo text models.

The API endpoint is: https://token-plan-cn.xiaomimimo.com/v1
"""

from __future__ import annotations

from typing import Any

from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner


class MimoPlanner(OpenAICompatiblePlanner):
    """
    Planner for Xiaomi MiMo's OpenAI-compatible API.

    This planner works with MiMo's text models (e.g., mimo-v2.5-pro)
    via their OpenAI-compatible chat completions endpoint.

    By default, forces JSON output via ``response_format={"type": "json_object"}``
    so that plan/decide/answer stages produce reliably parseable structured output.

    Args:
        model: Model name (default: "mimo-v2.5-pro")
        api_key: API key. If None, reads from MIMO_API_KEY environment variable.
        base_url: API base URL. Defaults to MiMo's endpoint.
        api_key_env: Environment variable name to read API key from.
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        enable_thinking: Enable thinking/reasoning mode
        timeout: API request timeout in seconds
        response_format: Output format constraint. Defaults to JSON object mode.
                         Set to None to let the model reply in free-form text.
    """

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "MIMO_API_KEY",
        temperature: float = 0.05,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        timeout: float = 120.0,
        max_retries: int = 3,
        response_format: dict[str, str] | None = {"type": "json_object"},
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://token-plan-cn.xiaomimimo.com/v1",
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._response_format = response_format

    @property
    def name(self) -> str:
        return f"mimo_planner_{self._model}"

    def call_model(self, model_input) -> str:
        """
        Call the MiMo API model and return the response text.

        Overrides the base to inject ``response_format={"type": "json_object"}``
        (or the configured format) so that plan/decide/answer outputs are
        guaranteed valid JSON when required.
        """
        from audio_agent.core.errors import PlannerError

        client = self.model_handle

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": model_input.messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        # MiMo-specific: JSON output by default (can be overridden or disabled)
        if self._response_format is not None:
            kwargs["response_format"] = self._response_format

        if self._enable_thinking:
            kwargs.setdefault("extra_body", {})
            if isinstance(kwargs["extra_body"], dict):
                kwargs["extra_body"]["enable_thinking"] = True

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise PlannerError(
                f"API call failed for model {self._model}: {e}",
                details={"model": self._model, "error_type": type(e).__name__},
            ) from e

        if not response.choices or len(response.choices) == 0:
            raise PlannerError(
                "Empty response from API",
                details={"model": self._model},
            )

        message = response.choices[0].message
        return self._extract_content(message)
