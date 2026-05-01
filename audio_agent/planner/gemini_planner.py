"""
Gemini API planner implementation.

Uses Google's Gemini API (3 Flash / 2.5 Pro) via the company proxy endpoint.
Converts OpenAI-style messages from BaseModelPlanner into Gemini's native
generateContent payload format.

Tested with runway.devops.rednote.life proxy.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

from audio_agent.core.errors import PlannerError
from audio_agent.planner.model_planner import (
    BaseModelPlanner,
    PlannerInputFormat,
    UnifiedPlannerInput,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://runway.devops.rednote.life/openai/google/v1:generateContent"

# Known API keys for convenience (model selection is controlled by which key is used)
_DEFAULT_API_KEYS = {
    "gemini-3-flash": "0c4daa7ef46d453fb91d7ac3ceefaf25",
    "gemini-2.5-pro": "035267d99c1f4f2e8ae03cedbfd61732",
}


class GeminiPlanner(BaseModelPlanner):
    """Planner using Google Gemini API.

    This planner works with any Gemini model accessible through the proxy
    endpoint, including Gemini 3 Flash and Gemini 2.5 Pro.

    Args:
        api_key: API key for the Gemini endpoint. If None, reads from
            GEMINI_API_KEY env var, then falls back to known default keys.
        base_url: Gemini API endpoint URL.
        model_name: Model identifier for logging (e.g. "gemini-3-flash").
            The actual model is determined by the API key used.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum output tokens.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for model output parsing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "gemini-3-flash",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_BASE_URL
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        super().__init__(max_retries=max_retries)

    @property
    def name(self) -> str:
        return f"gemini_planner_{self._model_name}"

    @property
    def input_format(self) -> PlannerInputFormat:
        return PlannerInputFormat.API_MODEL

    def initialize_model(self) -> Any:
        """No persistent client needed; we use requests directly."""
        return None

    def _resolve_api_key(self) -> str:
        """Resolve API key from constructor arg, env var, or defaults."""
        if self._api_key:
            return self._api_key

        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            return env_key

        # Fallback to known defaults based on model name
        default_key = _DEFAULT_API_KEYS.get(self._model_name)
        if default_key:
            logger.warning(
                f"Using hardcoded default API key for {self._model_name}. "
                "Set GEMINI_API_KEY env var or pass api_key explicitly."
            )
            return default_key

        raise PlannerError(
            "API key required for Gemini planner",
            details={
                "hint": "Provide api_key parameter or set GEMINI_API_KEY environment variable",
                "model_name": self._model_name,
            },
        )

    def _convert_messages_to_gemini(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert OpenAI-style messages to Gemini format.

        Returns:
            (system_instruction_text, gemini_contents_list)
        """
        system_instruction: str | None = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # Gemini uses system_instruction, not a system message in contents
                system_instruction = content
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })
            else:
                # Unknown role: treat as user message
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })

        return system_instruction, contents

    def _build_payload(
        self,
        system_instruction: str | None,
        contents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the Gemini generateContent request payload."""
        generation_config: dict[str, Any] = {
            "temperature": self._temperature,
        }
        if self._max_tokens > 0:
            generation_config["maxOutputTokens"] = self._max_tokens

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_instruction:
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }

        return payload

    def _call_gemini_api(self, payload: dict[str, Any]) -> str:
        """Send request to Gemini endpoint and extract text response."""
        api_key = self._resolve_api_key()

        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self._base_url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout as e:
            raise PlannerError(
                f"Gemini API request timed out after {self._timeout}s",
                details={"base_url": self._base_url, "timeout": self._timeout},
            ) from e
        except requests.exceptions.RequestException as e:
            raise PlannerError(
                f"Gemini API request failed: {type(e).__name__}: {e}",
                details={"base_url": self._base_url},
            ) from e

        if response.status_code >= 400:
            raise PlannerError(
                f"Gemini API returned HTTP {response.status_code}",
                details={
                    "status_code": response.status_code,
                    "response": response.text[:1000],
                    "base_url": self._base_url,
                },
            )

        try:
            result = response.json()
        except Exception as e:
            raise PlannerError(
                f"Failed to parse Gemini API response as JSON: {e}",
                details={"response": response.text[:1000]},
            ) from e

        # Error code handling (e.g. token limit)
        if "Code" in result and "Error" in result:
            message = str(result.get("Error", ""))
            code = result.get("Code")
            if code == 10001 or "too many tokens" in message.lower():
                raise PlannerError(
                    f"Gemini API token limit error (Code {code}): {message}",
                    details={"response": result},
                )
            raise PlannerError(
                f"Gemini API error (Code {code}): {message}",
                details={"response": result},
            )

        # Validate response structure
        if "candidates" not in result or not result["candidates"]:
            raise PlannerError(
                "Gemini API returned no candidates",
                details={"response": result},
            )

        try:
            parts = result["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as e:
            raise PlannerError(
                f"Malformed Gemini API response: {e}",
                details={"response": json.dumps(result)[:1000]},
            ) from e

        # Extract text from parts
        # If thinking is present, it's usually in parts[0]; actual answer in parts[1]
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])

        if not text_parts:
            raise PlannerError(
                "Gemini API returned no text parts",
                details={"response": json.dumps(result)[:1000]},
            )

        # If multiple parts, prefer the last one (actual answer after thinking)
        # This matches the behavior in GeminiFrontend
        if len(text_parts) > 1:
            combined = text_parts[-1].strip()
        else:
            combined = text_parts[0].strip()

        if not combined:
            raise PlannerError(
                "Gemini API returned empty text",
                details={"response": json.dumps(result)[:1000]},
            )

        return combined

    def call_model(self, model_input: UnifiedPlannerInput) -> str:
        """Call Gemini API and return raw text output."""
        system_instruction, contents = self._convert_messages_to_gemini(
            model_input.messages
        )

        if not contents:
            raise PlannerError(
                "No valid messages to send to Gemini API",
                details={"messages": model_input.messages},
            )

        payload = self._build_payload(system_instruction, contents)
        return self._call_gemini_api(payload)
