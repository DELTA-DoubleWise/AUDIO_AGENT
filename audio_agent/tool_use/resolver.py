"""Resolve abstract tool-use skills into concrete tool chains.

The resolver intentionally separates two concerns:
- task_skills.yaml defines fine-grained tasks and abstract slots
- reports.yaml supplies benchmark-derived rankings for filling those slots

If report coverage is missing, or the ranked tool is unavailable in the current
catalog, the resolver falls back to defaults extracted from reference flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "tool_use"
DEFAULT_SKILLS_PATH = CONFIG_DIR / "task_skills.yaml"
DEFAULT_REPORTS_PATH = CONFIG_DIR / "reports.yaml"


@dataclass(frozen=True)
class ResolutionContext:
    """Optional runtime context for selecting report rows."""

    dataset_id: str | None = None
    available_tools: frozenset[str] | None = None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return data


def _agent_tool_name(tool: str | None, meta: dict[str, Any] | None = None) -> str | None:
    if meta:
        explicit = meta.get("agent_tool_name") or meta.get("fallback_agent_tool_name")
        if explicit:
            return str(explicit)
    if not tool:
        return None
    return tool.rsplit(".", 1)[-1]


def _full_tool(row: dict[str, Any], tool_keys: dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
    tool_key = row.get("tool_key")
    if not tool_key:
        return None, None, {}
    meta = tool_keys.get(tool_key, {})
    if not isinstance(meta, dict):
        meta = {}
    full_tool = meta.get("catalog_tool") or meta.get("fallback_catalog_tool")
    agent_tool = meta.get("agent_tool_name") or meta.get("fallback_agent_tool_name") or _agent_tool_name(full_tool)
    return full_tool, agent_tool, meta


def _is_available(
    catalog_tool: str | None,
    available_tools: frozenset[str] | None,
    agent_tool_name: str | None = None,
) -> bool:
    if not catalog_tool and not agent_tool_name:
        return False
    if available_tools is None:
        return True
    return bool(
        (catalog_tool and catalog_tool in available_tools)
        or (agent_tool_name and agent_tool_name in available_tools)
    )


class ToolUseResolver:
    """Resolve a skill into concrete tools using reports first, defaults second."""

    def __init__(
        self,
        skills_path: str | Path = DEFAULT_SKILLS_PATH,
        reports_path: str | Path = DEFAULT_REPORTS_PATH,
    ) -> None:
        self.skills_path = Path(skills_path)
        self.reports_path = Path(reports_path)
        self.skills_doc = _load_yaml(self.skills_path)
        self.reports_doc = _load_yaml(self.reports_path)

    def get_skill(self, skill_id: str) -> dict[str, Any]:
        for skill in self.skills_doc.get("skills", []):
            if skill.get("id") == skill_id:
                return skill
        raise KeyError(f"Unknown tool-use skill: {skill_id}")

    def matching_rankings(
        self,
        skill_id: str,
        slot: str,
        dataset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        exact = []
        fallback = []
        for ranking in self.reports_doc.get("tool_rankings", []):
            if ranking.get("skill_id") != skill_id or ranking.get("slot") != slot:
                continue
            if dataset_id and ranking.get("dataset_id") == dataset_id:
                exact.append(ranking)
            else:
                fallback.append(ranking)
        return exact or fallback

    def _select_from_report(
        self,
        skill_id: str,
        slot: str,
        context: ResolutionContext,
    ) -> dict[str, Any] | None:
        tool_keys = self.reports_doc.get("tool_keys", {})
        for ranking in self.matching_rankings(skill_id, slot, context.dataset_id):
            for index, row in enumerate(ranking.get("rows", []), start=1):
                catalog_tool, agent_tool_name, meta = _full_tool(row, tool_keys)
                if not _is_available(catalog_tool, context.available_tools, agent_tool_name):
                    continue
                selected_by = "report"
                warning = None
                if meta.get("catalog_tool") is None and meta.get("fallback_catalog_tool"):
                    selected_by = "report_fallback_catalog_tool"
                    warning = (
                        f"Report winner {meta.get('display_name') or row.get('tool_key')} "
                        f"is unavailable; using fallback {catalog_tool}."
                    )
                return {
                    "slot": slot,
                    "tool": catalog_tool,
                    "agent_tool_name": agent_tool_name,
                    "selected_by": selected_by,
                    "ranking_id": ranking.get("id"),
                    "dataset_id": ranking.get("dataset_id"),
                    "rank": index,
                    "metric": ranking.get("metric"),
                    "direction": ranking.get("direction"),
                    "score": row.get("score"),
                    "rps": row.get("rps"),
                    "tool_key": row.get("tool_key"),
                    "display_name": row.get("display_name") or meta.get("display_name"),
                    "warning": warning,
                }
        return None

    def _select_from_defaults(
        self,
        skill: dict[str, Any],
        slot: str,
        context: ResolutionContext,
    ) -> dict[str, Any] | None:
        defaults = skill.get("default_bindings_from_ref_flows", {}).get(slot, [])
        for index, candidate in enumerate(defaults, start=1):
            tool = candidate.get("tool")
            agent_tool_name = candidate.get("agent_tool_name") or _agent_tool_name(tool)
            if _is_available(tool, context.available_tools, agent_tool_name):
                return {
                    "slot": slot,
                    "tool": tool,
                    "agent_tool_name": agent_tool_name,
                    "selected_by": "skill_default",
                    "rank": index,
                    "reference": candidate.get("reference"),
                }
        return None

    def _select_from_slot_defaults(
        self,
        slot: str,
        context: ResolutionContext,
    ) -> dict[str, Any] | None:
        slot_meta = self.skills_doc.get("abstract_slots", {}).get(slot, {})
        defaults = slot_meta.get("current_catalog_defaults", [])
        for index, tool in enumerate(defaults, start=1):
            agent_tool_name = _agent_tool_name(tool)
            if _is_available(tool, context.available_tools, agent_tool_name):
                return {
                    "slot": slot,
                    "tool": tool,
                    "agent_tool_name": agent_tool_name,
                    "selected_by": "abstract_slot_default",
                    "rank": index,
                    "contract": slot_meta.get("contract"),
                }
        return None

    def resolve(
        self,
        skill_id: str,
        dataset_id: str | None = None,
        available_tools: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Resolve a skill to a concrete chain.

        Args:
            skill_id: Fine-grained task skill ID.
            dataset_id: Optional dataset/scenario ID for exact report matching.
            available_tools: Optional set of catalog tools in ``server.tool`` form.

        Returns:
            A structured resolution with concrete chain items and unresolved slots.
        """
        skill = self.get_skill(skill_id)
        context = ResolutionContext(
            dataset_id=dataset_id,
            available_tools=frozenset(available_tools) if available_tools is not None else None,
        )

        chain = []
        unresolved = []
        for step in skill.get("chain", []):
            slot = step.get("slot")
            if not slot:
                continue
            selected = self._select_from_report(skill_id, slot, context)
            if selected is None:
                selected = self._select_from_defaults(skill, slot, context)
            if selected is None:
                selected = self._select_from_slot_defaults(slot, context)
            if selected is None:
                item = {
                    "slot": slot,
                    "optional": bool(step.get("optional", False)),
                    "reason": "No report candidate or default binding is available.",
                }
                unresolved.append(item)
                continue
            selected["optional"] = bool(step.get("optional", False))
            chain.append(selected)

        return {
            "skill_id": skill_id,
            "task": skill.get("task"),
            "dataset_id": dataset_id,
            "chain": chain,
            "unresolved_slots": unresolved,
            "guardrails": skill.get("guardrails", []),
        }


def resolve_tool_chain(
    skill_id: str,
    dataset_id: str | None = None,
    available_tools: set[str] | list[str] | tuple[str, ...] | None = None,
    skills_path: str | Path = DEFAULT_SKILLS_PATH,
    reports_path: str | Path = DEFAULT_REPORTS_PATH,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`ToolUseResolver`."""
    return ToolUseResolver(skills_path, reports_path).resolve(
        skill_id=skill_id,
        dataset_id=dataset_id,
        available_tools=available_tools,
    )
