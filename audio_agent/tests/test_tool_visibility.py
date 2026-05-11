"""Tests for planner-facing tool visibility."""

import pytest

from audio_agent.core.schemas import ToolSpec
from audio_agent.tools.visibility import CORE_PLANNER_TOOL_NAMES, filter_tool_specs


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"{name} description")


class TestToolVisibility:
    """Tests for planner tool scope filtering."""

    def test_core_scope_returns_only_allowlisted_specs(self):
        """Core scope should hide registered tools outside the core allowlist."""
        specs = [
            _spec("trim_audio"),
            _spec("dummy_asr"),
            _spec("recognize_chords_large_vocab"),
        ]

        filtered = filter_tool_specs(specs, scope="core")

        assert [spec.name for spec in filtered] == ["trim_audio", "recognize_chords_large_vocab"]

    def test_all_scope_returns_every_spec(self):
        """All scope should preserve the full registered tool list."""
        specs = [
            _spec("trim_audio"),
            _spec("dummy_asr"),
            _spec("add_echo"),
        ]

        filtered = filter_tool_specs(specs, scope="all")

        assert filtered == specs

    def test_invalid_scope_raises(self):
        """Invalid scopes should fail fast."""
        with pytest.raises(ValueError, match="Invalid planner tool scope"):
            filter_tool_specs([_spec("trim_audio")], scope="invalid")

    def test_unregistered_core_names_are_harmless(self):
        """Runtime filtering should not require every core tool to be registered."""
        specs = [_spec("trim_audio")]

        filtered = filter_tool_specs(specs, scope="core")

        assert [spec.name for spec in filtered] == ["trim_audio"]

    def test_core_allowlist_size(self):
        """The default planner core inventory is intentionally fixed."""
        assert len(CORE_PLANNER_TOOL_NAMES) == 38
        assert "inspect_audio_plots" in CORE_PLANNER_TOOL_NAMES
        assert "lyric_asr" not in CORE_PLANNER_TOOL_NAMES
        assert "omni_caption" not in CORE_PLANNER_TOOL_NAMES
