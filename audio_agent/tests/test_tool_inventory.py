"""Tests for standalone planner tool inventory overrides."""

from __future__ import annotations

import pytest

from audio_agent.core.errors import ToolRegistryError
from audio_agent.core.schemas import ToolSpec
from audio_agent.tools.inventory import (
    PLANNER_TOOL_CATEGORY_ORDER,
    apply_planner_tool_inventory,
    load_planner_tool_category_definitions,
    load_planner_tool_inventory,
)


def _inventory_yaml(tool_entries: str) -> str:
    """Wrap tool entries in the planner inventory schema."""
    category_definitions = "\n".join(
        f"  {category}:\n"
        f"    definition: Definition for {category}.\n"
        f"    guideline: Guideline for {category}."
        for category in PLANNER_TOOL_CATEGORY_ORDER
    )
    category_order = "\n".join(f"  - {category}" for category in PLANNER_TOOL_CATEGORY_ORDER)
    return (
        f"category_order:\n{category_order}\n"
        f"category_definitions:\n{category_definitions}\n"
        f"tools:\n{tool_entries}"
    )


def test_load_planner_tool_inventory(tmp_path):
    """Inventory files should load as entries keyed by name."""
    path = tmp_path / "tools.yaml"
    path.write_text(
        _inventory_yaml(
            """
  - name: trim_audio
    category: audio_derivation
    description:
      function: Cut a selected time range.
    input_schema: {}
    tags: [mcp, ffmpeg]
"""
        ),
        encoding="utf-8",
    )

    inventory = load_planner_tool_inventory(path)

    assert sorted(inventory) == ["trim_audio"]
    assert inventory["trim_audio"]["description"]["function"] == "Cut a selected time range."

    definitions = load_planner_tool_category_definitions(path)
    assert definitions["audio_derivation"]["definition"] == "Definition for audio_derivation."


def test_apply_planner_tool_inventory_overrides_registered_spec(tmp_path):
    """Inventory descriptions and tags should override registered planner specs."""
    path = tmp_path / "tools.yaml"
    path.write_text(
        _inventory_yaml(
            """
  - name: trim_audio
    category: audio_derivation
    description:
      function: Cut a focused clip.
      recommended_use:
        - Use before frontend follow-up.
      not_recommended_use:
        - Do not use as analysis.
    input_schema:
      type: object
      properties:
        audio_path:
          type: string
      required: [audio_path]
    tags: [inventory, trim]
"""
        ),
        encoding="utf-8",
    )
    specs = [
        ToolSpec(
            name="trim_audio",
            description="server description",
            input_schema={"type": "object"},
            tags=["server"],
        ),
        ToolSpec(name="dummy_tool", description="dummy description", tags=["dummy"]),
    ]

    updated = apply_planner_tool_inventory(specs, inventory_path=path)

    assert updated[0].description.startswith("Function: Cut a focused clip.")
    assert "Category: audio_derivation" in updated[0].description
    assert "Recommended use: Use before frontend follow-up." in updated[0].description
    assert "Not recommended: Do not use as analysis." in updated[0].description
    assert updated[0].input_schema == {
        "type": "object",
        "properties": {"audio_path": {"type": "string"}},
        "required": ["audio_path"],
    }
    assert updated[0].tags == ["inventory", "trim"]
    assert [spec.name for spec in updated] == ["trim_audio"]


def test_apply_planner_tool_inventory_raises_for_missing_file(tmp_path):
    """Missing inventory files should fail fast."""
    with pytest.raises(ToolRegistryError, match="Planner tool inventory file not found"):
        apply_planner_tool_inventory([], inventory_path=tmp_path / "missing.yaml")


def test_apply_planner_tool_inventory_requires_input_schema(tmp_path):
    """Inventory entries used by the planner should be self-contained."""
    path = tmp_path / "tools.yaml"
    path.write_text(
        _inventory_yaml(
            """
  - name: trim_audio
    category: audio_derivation
    description:
      function: Cut a selected time range.
    tags: [inventory]
"""
        ),
        encoding="utf-8",
    )
    specs = [ToolSpec(name="trim_audio", description="server description")]

    with pytest.raises(ToolRegistryError, match="missing input_schema"):
        apply_planner_tool_inventory(specs, inventory_path=path)


def test_apply_planner_tool_inventory_requires_valid_category(tmp_path):
    """Inventory categories should be explicit and from the planner-facing taxonomy."""
    path = tmp_path / "tools.yaml"
    path.write_text(
        _inventory_yaml(
            """
  - name: trim_audio
    category: generic
    description:
      function: Cut a selected time range.
    input_schema: {}
    tags: [inventory]
"""
        ),
        encoding="utf-8",
    )
    specs = [ToolSpec(name="trim_audio", description="server description")]

    with pytest.raises(ToolRegistryError, match="category is missing or invalid"):
        apply_planner_tool_inventory(specs, inventory_path=path)


def test_load_planner_tool_inventory_requires_category_metadata(tmp_path):
    """Inventory files should include category definitions for LLM-facing organization."""
    path = tmp_path / "tools.yaml"
    path.write_text(
        """
tools:
  - name: trim_audio
    category: audio_derivation
    description:
      function: Cut a selected time range.
    input_schema: {}
    tags: [inventory]
""",
        encoding="utf-8",
    )

    with pytest.raises(ToolRegistryError, match="category_order is missing or invalid"):
        load_planner_tool_inventory(path)
