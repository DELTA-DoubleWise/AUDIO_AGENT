"""Tests for shared model I/O utilities."""

import pytest

from audio_agent.core.errors import FrontendError, PlannerError
from audio_agent.utils.model_io import parse_json_object_text, validate_message_sequence


class TestParseJsonObjectText:
    """Tests for strict JSON object text parsing."""

    def test_parses_plain_json_object_text(self):
        parsed = parse_json_object_text(
            '{"question_guided_caption": "caption"}',
            error_cls=FrontendError,
            subject="Model",
        )

        assert parsed == {"question_guided_caption": "caption"}

    def test_parses_fenced_json_object_text(self):
        parsed = parse_json_object_text(
            """```json
{
  "action": "fail",
  "rationale": "insufficient evidence"
}
```""",
            error_cls=PlannerError,
            subject="Planner",
        )

        assert parsed == {
            "action": "fail",
            "rationale": "insufficient evidence",
        }

    def test_rejects_non_string_input(self):
        with pytest.raises(FrontendError, match="Model text output must be a string"):
            parse_json_object_text(
                {"not": "a string"},
                error_cls=FrontendError,
                subject="Model",
            )

    def test_rejects_empty_text(self):
        with pytest.raises(PlannerError, match="Planner text output is empty"):
            parse_json_object_text(
                "   ",
                error_cls=PlannerError,
                subject="Planner",
            )

    def test_rejects_non_object_json(self):
        with pytest.raises(FrontendError, match="Model output JSON must be an object"):
            parse_json_object_text(
                '["not", "an", "object"]',
                error_cls=FrontendError,
                subject="Model",
            )

    def test_rejects_invalid_json_text(self):
        with pytest.raises(PlannerError, match="Planner output is not valid JSON object text"):
            parse_json_object_text(
                "not valid json",
                error_cls=PlannerError,
                subject="Planner",
            )


class TestValidateMessageSequence:
    """Tests for message list validation."""

    def test_rejects_empty_message_list(self):
        with pytest.raises(FrontendError, match="messages cannot be empty"):
            validate_message_sequence(
                [],
                error_cls=FrontendError,
                context="Malformed model input",
            )

    def test_rejects_non_dict_message(self):
        with pytest.raises(PlannerError, match="each message must be an object"):
            validate_message_sequence(
                ["bad"],
                error_cls=PlannerError,
                context="Malformed planner model input",
            )

    def test_rejects_message_missing_role_or_content(self):
        with pytest.raises(FrontendError, match="must include role and content"):
            validate_message_sequence(
                [{"role": "user"}],
                error_cls=FrontendError,
                context="Malformed model input",
            )
