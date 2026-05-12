"""MiMo API frontend implementation.

Uses Xiaomi MiMo's OpenAI-compatible API for audio understanding.
Supports mimo-v2.5 and other MiMo audio-capable models.

The API endpoint is: https://token-plan-cn.xiaomimimo.com/v1
"""

from __future__ import annotations

from typing import Any

from audio_agent.frontend.openai_compatible_frontend import OpenAICompatibleFrontend


class MimoFrontend(OpenAICompatibleFrontend):
    """
    Frontend for Xiaomi MiMo's OpenAI-compatible API with audio support.

    This frontend works with MiMo's audio-capable models (e.g., mimo-v2.5)
    via their OpenAI-compatible chat completions endpoint.

    Args:
        model: Model name (default: "mimo-v2.5")
        api_key: API key. If None, reads from MIMO_API_KEY environment variable.
        base_url: API base URL. Defaults to MiMo's endpoint.
        api_key_env: Environment variable name to read API key from.
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        timeout: API request timeout in seconds
        response_format: Optional output format constraint (e.g.
                         {"type": "json_object"}). Defaults to None (free text).
    """

    def __init__(
        self,
        model: str = "mimo-v2.5",
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "MIMO_API_KEY",
        temperature: float = 0.05,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
        response_format: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://token-plan-cn.xiaomimimo.com/v1",
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._response_format = response_format

    @property
    def name(self) -> str:
        return f"mimo_frontend_{self._model}"

    def _build_api_kwargs(self, messages: list[dict[str, Any]], stream: bool = False) -> dict[str, Any]:
        """Build API call kwargs with optional response format."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": stream,
        }
        if self._response_format is not None:
            kwargs["response_format"] = self._response_format
        if not stream:
            kwargs["modalities"] = ["text"]
        else:
            kwargs["stream_options"] = {"include_usage": True}
            kwargs["modalities"] = ["text"]
        return kwargs

    def call_model(self, model_input) -> str:
        """
        Call the MiMo API model and return the caption text.

        Overrides the base to support an optional ``response_format``.
        """
        from audio_agent.core.errors import FrontendError

        client = self.model_handle
        kwargs = self._build_api_kwargs(model_input.messages, stream=True)

        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise FrontendError(
                f"API call failed for model {self._model}: {e}",
                details={"model": self._model, "error_type": type(e).__name__},
            ) from e

        text_response = ""
        try:
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_response += chunk.choices[0].delta.content
        except Exception as e:
            raise FrontendError(
                f"Error processing API response: {e}",
                details={"model": self._model},
            ) from e

        if not text_response.strip():
            raise FrontendError(
                "API returned empty response",
                details={"model": self._model},
            )

        return text_response.strip()

    def generate_final_answer(self, question, audio_paths, context):
        """
        Generate final answer with optional response format.
        """
        from audio_agent.core.errors import FrontendError
        from audio_agent.frontend.model_frontend import FrontendInputFormat
        from audio_agent.utils.prompt_io import load_prompt

        self.validate_inputs(question, audio_paths)
        stripped_paths = [p.strip() for p in audio_paths]

        model_input = self.build_final_answer_model_input(question.strip(), stripped_paths, context)
        if not hasattr(model_input, "messages"):
            raise FrontendError(
                "Malformed model input: builder must return UnifiedFrontendInput",
                details={"returned_type": type(model_input).__name__},
            )

        original_messages = model_input.messages
        messages: list[dict[str, Any]] = []
        for msg in original_messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                new_content: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "audio":
                        audio_path = block.get("audio")
                        audio_data_url, audio_format = self._encode_audio(audio_path)
                        new_content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_data_url,
                                    "format": audio_format,
                                },
                            }
                        )
                    else:
                        new_content.append(block)
                messages.append({"role": "user", "content": new_content})
            else:
                messages.append(msg)

        model_input.messages = messages
        model_input.metadata["input_format"] = FrontendInputFormat.API_MODEL.value

        client = self.model_handle
        kwargs = self._build_api_kwargs(messages, stream=False)

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise FrontendError(
                f"API call failed for final answer: {e}",
                details={"model": self._model, "error_type": type(e).__name__},
            ) from e

        if not response.choices or len(response.choices) == 0:
            raise FrontendError(
                "Empty response from API",
                details={"model": self._model},
            )

        text_response = response.choices[0].message.content or ""
        if not text_response.strip():
            raise FrontendError(
                "API returned empty final answer",
                details={"model": self._model},
            )

        return text_response.strip()
