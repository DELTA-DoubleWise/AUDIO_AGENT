"""
Gemini API frontend implementation.

Uses Google's Gemini 2.5 Pro via a custom API endpoint.
Tested with runway.devops.rednote.life proxy.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import requests

from audio_agent.core.errors import FrontendError
from audio_agent.core.schemas import FrontendOutput
from audio_agent.frontend.model_frontend import (
    BaseModelFrontend,
    FrontendInputFormat,
    UnifiedFrontendInput,
)
from audio_agent.utils.prompt_io import load_prompt


class GeminiFrontend(BaseModelFrontend):
    """
    Frontend for Google Gemini API with audio support.

    This frontend uses the native Gemini generateContent API format:
    - generationConfig for parameters
    - system_instruction for system prompt
    - contents with inline_data for audio
    - Handles thinking responses (parts[0] = thoughts, parts[1] = answer)

    Args:
        api_key: API key for the Gemini proxy endpoint.
        base_url: API endpoint URL.
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum output tokens
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://runway.devops.rednote.life/openai/google/v1:generateContent",
        temperature: float = 0.01,
        max_tokens: int = 40960,
        timeout: float = 60.0,
        max_retries: int = 30,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        super().__init__(max_retries=max_retries)

    @property
    def name(self) -> str:
        return "gemini_frontend"

    @property
    def input_format(self) -> FrontendInputFormat:
        return FrontendInputFormat.API_MODEL

    def initialize_model(self) -> Any:
        """No persistent client needed; we use requests directly."""
        return None

    def _encode_audio(self, audio_path: str) -> str:
        """Read and base64-encode an audio file."""
        path = Path(audio_path)
        if not path.exists():
            raise FrontendError(
                f"Audio file not found: {audio_path}",
                details={"audio_path": audio_path},
            )
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise FrontendError(
                f"Failed to read/encode audio file: {e}",
                details={"audio_path": audio_path},
            ) from e

    def _validate_built_model_input(self, model_input: UnifiedFrontendInput) -> None:
        """Override to skip message-sequence validation; Gemini uses its own payload."""
        if not model_input.system_prompt.strip():
            raise FrontendError("Malformed model input: empty system_prompt")
        if not model_input.question.strip():
            raise FrontendError("Malformed model input: empty question")
        if not model_input.audio_paths or len(model_input.audio_paths) == 0:
            raise FrontendError("Malformed model input: empty audio_paths")
        for i, path in enumerate(model_input.audio_paths):
            if not path or not path.strip():
                raise FrontendError(f"Malformed model input: empty audio path at index {i}")

    def _build_gemini_payload(
        self,
        system_prompt: str,
        user_text: str,
        audio_paths: list[str],
    ) -> dict[str, Any]:
        """Build the Gemini generateContent request payload."""
        parts: list[dict[str, Any]] = []
        for audio_path in audio_paths:
            audio_b64 = self._encode_audio(audio_path)
            parts.append({
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": audio_b64,
                }
            })
        parts.append({"text": user_text})

        return {
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_tokens,
                "thinkingConfig": {"includeThoughts": True},
            },
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
        }

    def _call_gemini_api(self, payload: dict[str, Any]) -> str:
        """Send request to Gemini endpoint and extract text response."""
        api_key = self._api_key
        if not api_key:
            raise FrontendError(
                "API key required for Gemini frontend",
                details={"hint": "Provide api_key parameter"},
            )

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
        except Exception as e:
            raise FrontendError(
                f"Gemini API request failed: {type(e).__name__}: {e}",
                details={"base_url": self._base_url},
            ) from e

        if response.status_code != 200:
            raise FrontendError(
                f"Gemini API returned status {response.status_code}",
                details={
                    "status_code": response.status_code,
                    "response": response.text[:500],
                },
            )

        try:
            result = response.json()
        except Exception as e:
            raise FrontendError(
                f"Failed to parse Gemini API response: {e}",
                details={"response": response.text[:500]},
            ) from e

        # Error code handling (token limit etc.)
        if "Code" in result and result["Code"] == 10001:
            raise FrontendError(
                "Gemini API token limit error (Code 10001)",
                details={"response": result},
            )

        if "candidates" not in result or not result["candidates"]:
            raise FrontendError(
                "Gemini API returned no candidates",
                details={"response": result},
            )

        try:
            parts = result["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as e:
            raise FrontendError(
                f"Malformed Gemini API response: {e}",
                details={"response": result},
            ) from e

        # Prefer parts[1] (actual answer) if thinking is in parts[0]
        if len(parts) > 1:
            text = parts[1].get("text", "")
        elif len(parts) > 0:
            text = parts[0].get("text", "")
        else:
            text = ""

        text = text.strip()
        if not text:
            raise FrontendError(
                "Gemini API returned empty text",
                details={"response": result},
            )

        return text

    def build_api_model_input(
        self,
        question: str,
        audio_paths: list[str],
        question_oriented_prompt: str | None = None,
    ) -> UnifiedFrontendInput:
        """Build Gemini-format input."""
        system_prompt = load_prompt("frontend_system")
        user_text = self.build_frontend_task_instruction(question, audio_paths, question_oriented_prompt)

        payload = self._build_gemini_payload(system_prompt, user_text, audio_paths)

        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_paths=audio_paths,
            user_payload={
                "question": question,
                "audio": {"kind": "paths", "value": audio_paths, "count": len(audio_paths)},
                "task": "question_guided_audio_captioning",
                "output_format": "plain_text_caption",
                "question_oriented_prompt": question_oriented_prompt,
            },
            messages=[],  # Not used for Gemini; payload is in metadata
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.API_MODEL.value,
                "model": "gemini-2.5-pro",
                "gemini_payload": payload,
                "question_oriented_prompt": question_oriented_prompt,
            },
        )

    def build_followup_api_model_input(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
    ) -> UnifiedFrontendInput:
        """Build Gemini follow-up input; only payload packaging is provider-specific."""
        common = self.build_followup_common_fields(
            question, audio_paths, followup_prompt, FrontendInputFormat.API_MODEL
        )
        payload = self._build_gemini_payload(
            common["system_prompt"], common["user_text"], audio_paths
        )

        return UnifiedFrontendInput(
            system_prompt=common["system_prompt"],
            question=question,
            audio_paths=audio_paths,
            user_payload=common["user_payload"],
            messages=[
                {"role": "system", "content": common["system_prompt"]},
                {"role": "user", "content": common["user_text"]},
            ],
            metadata={
                **common["metadata"],
                "model": "gemini-2.5-pro",
                "gemini_payload": payload,
            },
        )

    def call_model(self, model_input: UnifiedFrontendInput) -> str:
        """Call Gemini API and return caption text."""
        payload = model_input.metadata.get("gemini_payload")
        if not payload:
            raise FrontendError(
                "Missing gemini_payload in model input metadata",
                details={"frontend": self.name},
            )
        return self._call_gemini_api(payload)

    def build_final_answer_model_input(
        self,
        question: str,
        audio_paths: list[str],
        context: dict[str, Any],
    ) -> UnifiedFrontendInput:
        """Build Gemini-format input for final answer generation."""
        system_prompt = load_prompt("frontend_final_answer_system")

        evidence_summary = context.get("evidence_summary")
        evidence_log = context.get("evidence_log", [])
        evidence_text = "\n".join(
            f"[{item.source}] {item.content}"
            for item in evidence_log
        ) if evidence_log else "No evidence collected."

        planner_trace = context.get("planner_trace", [])
        planner_trace_text = "\n".join(
            f"Step {i+1}: {d.action.value} - {d.rationale}"
            for i, d in enumerate(planner_trace)
        ) if planner_trace else "No planner decisions yet."

        tool_history = context.get("tool_call_history", [])
        tool_history_text = "\n".join(
            f"- {record.request.tool_name}: success={record.result.success}"
            for record in tool_history
        ) if tool_history else "No tools called."

        initial_plan = context.get("initial_plan")
        initial_plan_text = initial_plan.approach if initial_plan else "No initial plan."

        initial_frontend_output = context.get("initial_frontend_output")
        frontend_direct_text = (
            initial_frontend_output.question_guided_caption
            if initial_frontend_output else "No frontend direct output."
        )

        audio_summary = "\n".join(
            f"- {a.audio_id}: {a.description}"
            for a in context.get("audio_list", [])
        ) if context.get("audio_list") else "No audio information."

        expected_output_format = context.get("expected_output_format") or "No specific format required."
        format_critique = context.get("format_critique")
        format_critique_section = (
            f"\n## Format Critique (previous attempt failed)\n{format_critique}\n"
            if format_critique else ""
        )

        if evidence_summary:
            evidence_and_history_text = (
                f"## Evidence and Reasoning Summary\n"
                f"{evidence_summary}\n\n"
            )
        else:
            evidence_and_history_text = (
                f"## Evidence Log\n"
                f"{evidence_text}\n\n"
                f"## Planner Reasoning Trace\n"
                f"{planner_trace_text}\n\n"
                f"## Tool Call History\n"
                f"{tool_history_text}\n\n"
            )

        user_text = load_prompt("frontend_final_answer_user").format(
            question=question,
            expected_output_format=expected_output_format,
            initial_plan_text=initial_plan_text,
            frontend_direct_text=frontend_direct_text,
            evidence_and_history_text=evidence_and_history_text,
            audio_summary=audio_summary,
            format_critique_section=format_critique_section,
        )

        payload = self._build_gemini_payload(system_prompt, user_text, audio_paths)

        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_paths=audio_paths,
            user_payload={
                "question": question,
                "audio": {"kind": "paths", "value": audio_paths, "count": len(audio_paths)},
                "task": "final_answer_generation",
            },
            messages=[],
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.API_MODEL.value,
                "model": "gemini-2.5-pro",
                "gemini_payload": payload,
                "task": "final_answer_generation",
                "audio_count": len(audio_paths),
            },
        )

    def generate_final_answer(
        self,
        question: str,
        audio_paths: list[str],
        context: dict[str, Any],
    ) -> str:
        """Generate final answer using the Gemini frontend model."""
        self.validate_inputs(question, audio_paths)
        stripped_paths = [p.strip() for p in audio_paths]

        model_input = self.build_final_answer_model_input(
            question.strip(), stripped_paths, context
        )
        if not isinstance(model_input, UnifiedFrontendInput):
            raise FrontendError(
                "Malformed model input: builder must return UnifiedFrontendInput",
                details={"returned_type": type(model_input).__name__},
            )

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except FrontendError:
                raise
            except Exception as e:
                raise FrontendError(
                    f"Final answer generation failed: {type(e).__name__}: {e}",
                    details={"frontend": self.name},
                ) from e
            answer = raw_output.strip()
            if not answer:
                raise FrontendError(
                    "Frontend returned empty final answer",
                    details={"frontend": self.name},
                )
            return answer

        return self._call_with_retries(_call, "generate_final_answer()")

    def run_direct_answer(
        self,
        question: str,
        audio_paths: list[str],
        question_oriented_prompt: str | None = None,
    ) -> FrontendOutput:
        """
        Run frontend in direct-answer (observer) mode.

        Uses `frontend_direct_system.md` instead of `frontend_system.md`.
        The returned FrontendOutput.question_guided_caption contains the direct answer text.
        """
        self.validate_inputs(question, audio_paths)
        stripped_paths = [p.strip() for p in audio_paths]

        system_prompt = load_prompt("frontend_direct_system")
        user_text = self.build_frontend_task_instruction(question, stripped_paths, question_oriented_prompt)
        payload = self._build_gemini_payload(system_prompt, user_text, stripped_paths)

        def _call():
            raw_output = self._call_gemini_api(payload)
            text = raw_output.strip()
            if not text:
                raise FrontendError("Gemini API returned empty direct answer")
            return FrontendOutput(question_guided_caption=text)

        return self._call_with_retries(_call, "run_direct_answer()")
