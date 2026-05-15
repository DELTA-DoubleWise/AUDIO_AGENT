#!/usr/bin/env python3
"""Validation harness for the Option 2 (native function calling) migration.

Runs a small battery of test prompts against the real DashScope
``qwen3.5-plus`` planner and captures the per-round tool emission for
inspection. Each test saves a markdown report under
``.artifacts/option2_validation/<test_id>.md`` containing:

- The system message and the user message of EACH planner round
- The model's raw emitted tool_calls per round
- The resulting PlannerDecision per round
- A summary (round count, tool count, pass/fail vs expectation)

Usage::

    export DASHSCOPE_API_KEY=sk-...
    python -m audio_agent.examples.option2_validation \
        --suite cpu \
        --tests t1 t2 t3 t4

    # GPU-only suite (requires a GPU node + GPU MCP tool envs):
    python -m audio_agent.examples.option2_validation --suite gpu

By default runs the CPU suite only. The CPU tests exercise tools that
ship with ffmpeg / librosa MCP servers (no GPU model weights needed).

Test fixtures live at::

    /itet-stor/yuchwang/net_scratch/workspace/AUDIO_AGENT/.artifacts/sample_audios/

The harness uses ``OpenAICompatibleFrontend`` (``qwen3.5-omni-plus``) for
the perception step. The frontend itself doesn't need GPU — DashScope
provides the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow direct execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from audio_agent.config.settings import AgentConfig
from audio_agent.core.constants import AgentStatus
from audio_agent.core.logging import setup_logger, set_debug_mode
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
from audio_agent.frontend.openai_compatible_frontend import OpenAICompatibleFrontend
from audio_agent.main import AudioAgent
from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner
from audio_agent.tools.catalog import register_all_mcp_tools
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.tools.registry import ToolRegistry


FIXTURES_DIR = Path(
    "/itet-stor/yuchwang/net_scratch/workspace/AUDIO_AGENT/.artifacts/sample_audios"
)
ARTIFACTS_DIR = (
    Path(__file__).parent.parent.parent
    / ".artifacts"
    / "option2_validation"
)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------


@dataclass
class Expectation:
    """Loose contract for a test's expected emission pattern.

    All bounds are intentionally generous (the model has some latitude in
    how it sequences tools) — we want to catch gross regressions, not
    over-specify the planner's choices.
    """

    min_rounds: int
    max_rounds: int
    # Tools the planner should call at some point. Order doesn't matter;
    # presence is required.
    must_call_tools: list[str] = field(default_factory=list)
    # The action that must terminate the loop (typically "answer").
    final_action: str = "answer"
    # If True, at least one round must contain >1 parallel tool_calls.
    requires_parallel_round: bool = False


@dataclass
class TestCase:
    test_id: str
    description: str
    audio_files: list[Path]
    question: str
    tools: list[str]  # MCP tool catalogs to register (e.g. ["ffmpeg", "librosa"])
    expectation: Expectation
    suite: str  # "cpu" or "gpu"


def _cpu_tests() -> list[TestCase]:
    return [
        TestCase(
            test_id="t1_smoke_single_tool",
            description=(
                "Single tool → answer. Verifies basic tool_call emission "
                "and that emit_final_answer terminates the loop."
            ),
            audio_files=[FIXTURES_DIR / "male_talking.wav"],
            question=(
                "Use the get_audio_info tool to report the sample rate and "
                "duration of this audio, then answer with both values."
            ),
            tools=["librosa"],
            expectation=Expectation(
                min_rounds=2,
                max_rounds=3,
                must_call_tools=["get_audio_info"],
                final_action="answer",
            ),
            suite="cpu",
        ),
        TestCase(
            test_id="t2_parallel_independent",
            description=(
                "Parallel independent investigations on the same audio. "
                "Verifies parallel_tool_calls=True works and the model "
                "groups independent calls into one round."
            ),
            audio_files=[FIXTURES_DIR / "sample_music.mp3"],
            question=(
                "Get the audio metadata (with get_audio_info), the musical "
                "key (with detect_key), and the pitch statistics (with "
                "analyze_pitch) of this music in parallel. Then answer "
                "with all three findings."
            ),
            tools=["librosa"],
            expectation=Expectation(
                min_rounds=2,
                max_rounds=4,
                must_call_tools=["get_audio_info", "detect_key", "analyze_pitch"],
                final_action="answer",
                requires_parallel_round=True,
            ),
            suite="cpu",
        ),
        TestCase(
            test_id="t3_cross_round_dependency",
            description=(
                "Cross-round dependency (segment → trim → analysis). "
                "Verifies the model splits dependent calls into separate "
                "rounds and references new audio_ids correctly."
            ),
            audio_files=[FIXTURES_DIR / "first_beep_then_music.wav"],
            question=(
                "First use segment_audio on this audio to find the boundary "
                "between the initial beep and the music that follows. Then "
                "trim the music portion of the audio into a new clip. Then "
                "estimate the musical key of the trimmed clip. Finally "
                "answer with the boundary time (in seconds) and the "
                "estimated key."
            ),
            tools=["librosa", "ffmpeg"],
            expectation=Expectation(
                min_rounds=3,
                max_rounds=6,
                must_call_tools=["segment_audio", "trim_audio", "detect_key"],
                final_action="answer",
            ),
            suite="cpu",
        ),
        TestCase(
            test_id="t4_fan_out_fan_in",
            description=(
                "Fan-out / fan-in. Round 1 emits 3 parallel trim calls; "
                "round 2 emits 3 parallel detect_key calls on the trimmed "
                "segments. This is the strongest argument for parallel calls."
            ),
            audio_files=[FIXTURES_DIR / "sample_music.mp3"],
            question=(
                "Trim this audio into three 5-second segments using "
                "trim_audio: 0–5s, 5–10s, and 10–15s. Then in the next "
                "round run detect_key on each of the three trimmed "
                "segments in parallel. Finally answer with the musical "
                "key estimated for each segment."
            ),
            tools=["librosa", "ffmpeg"],
            expectation=Expectation(
                min_rounds=3,
                max_rounds=5,
                must_call_tools=["trim_audio", "detect_key"],
                final_action="answer",
                requires_parallel_round=True,
            ),
            suite="cpu",
        ),
    ]


def _gpu_tests() -> list[TestCase]:
    return [
        TestCase(
            test_id="t5_asr_resample_chain",
            description=(
                "Resample → ASR chain. Tests cross-round dependency with "
                "GPU-backed ASR (transcribe_qwenasr)."
            ),
            audio_files=[FIXTURES_DIR / "asr_en.wav"],
            question=(
                "Use resample_audio to resample this audio to 16 kHz, then "
                "transcribe the resampled audio with transcribe_qwenasr. "
                "Answer with the transcript."
            ),
            tools=["ffmpeg", "asr_qwen3"],
            expectation=Expectation(
                min_rounds=3,
                max_rounds=5,
                must_call_tools=["resample_audio", "transcribe_qwenasr"],
                final_action="answer",
            ),
            suite="gpu",
        ),
        TestCase(
            test_id="t6_multi_speaker_parallel",
            description=(
                "Multi-speaker analysis with diarization + ASR-with-timestamps "
                "in parallel. Tests mixed parallel-then-sequential."
            ),
            audio_files=[FIXTURES_DIR / "conversation.wav"],
            question=(
                "Run diarize and transcribe_qwenasr_with_timestamps on this "
                "audio in parallel. Then answer with the number of speakers "
                "(from diarization) and the total speech duration in seconds "
                "(sum of speech segment lengths)."
            ),
            tools=["diarizen", "asr_qwen3"],
            expectation=Expectation(
                min_rounds=2,
                max_rounds=4,
                must_call_tools=["diarize", "transcribe_qwenasr_with_timestamps"],
                final_action="answer",
                requires_parallel_round=True,
            ),
            suite="gpu",
        ),
        TestCase(
            test_id="t7_full_gpu_chain",
            description=(
                "Full GPU chain: locate music boundary, trim, run chord "
                "recognition on the trimmed clip."
            ),
            audio_files=[FIXTURES_DIR / "first_beep_then_music.wav"],
            question=(
                "First use segment_audio to find the boundary between the "
                "initial beep and the music that follows. Then trim_audio "
                "to extract the music portion as a new clip. Then run "
                "recognize_chords_large_vocab on the trimmed clip. Finally "
                "answer with the chord progression."
            ),
            tools=["librosa", "ffmpeg", "lv_chordia"],
            expectation=Expectation(
                min_rounds=3,
                max_rounds=6,
                must_call_tools=[
                    "segment_audio",
                    "trim_audio",
                    "recognize_chords_large_vocab",
                ],
                final_action="answer",
            ),
            suite="gpu",
        ),
    ]


ALL_TESTS: dict[str, TestCase] = {
    t.test_id: t for t in _cpu_tests() + _gpu_tests()
}


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def _serialize_decision(decision: Any) -> dict:
    """Concise JSON-able view of a PlannerDecision."""
    tcs = getattr(decision, "selected_tool_calls", None) or []
    return {
        "action": getattr(decision.action, "value", str(decision.action)),
        "rationale": decision.rationale,
        "tool_calls": [
            {"tool_name": tc.tool_name, "args": tc.args}
            for tc in tcs
        ],
        "selected_audio_id": getattr(decision, "selected_audio_id", None),
        "selected_audio_ids": list(getattr(decision, "selected_audio_ids", []) or []),
        "frontend_followup_prompt": getattr(decision, "frontend_followup_prompt", None),
        "confidence": float(getattr(decision, "confidence", 0.0)),
    }


def _verify_expectation(
    tc: TestCase, trace: list[Any]
) -> tuple[bool, list[str]]:
    """Compare a recorded planner trace against the test's expectation.

    Returns ``(ok, failure_messages)``. ``ok`` is False if any check fails.
    """
    failures: list[str] = []
    exp = tc.expectation

    rounds = len(trace)
    if rounds < exp.min_rounds:
        failures.append(
            f"round count {rounds} < min {exp.min_rounds}"
        )
    if rounds > exp.max_rounds:
        failures.append(
            f"round count {rounds} > max {exp.max_rounds}"
        )

    tools_called: list[str] = []
    parallel_round_seen = False
    for d in trace:
        tcs = getattr(d, "selected_tool_calls", None) or []
        if len(tcs) >= 2:
            parallel_round_seen = True
        tools_called.extend(tc.tool_name for tc in tcs)

    for tool in exp.must_call_tools:
        if tool not in tools_called:
            failures.append(f"required tool {tool!r} never called")

    if exp.requires_parallel_round and not parallel_round_seen:
        failures.append("no round emitted 2+ parallel tool_calls")

    last_action = (
        getattr(trace[-1].action, "value", str(trace[-1].action))
        if trace
        else None
    )
    if last_action != exp.final_action:
        failures.append(
            f"final action was {last_action!r}, expected {exp.final_action!r}"
        )

    return (len(failures) == 0, failures)


def _write_report(
    tc: TestCase,
    final_state: dict,
    system_message: str | None,
    user_messages: list[str],
    raw_tool_calls_per_round: list[list[dict]],
    error: str | None,
    elapsed_seconds: float,
) -> Path:
    """Write a markdown report for one test run."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / f"{tc.test_id}.md"

    planner_trace = final_state.get("planner_trace", []) if final_state else []
    ok, failures = (False, ["test errored before completion"])
    if not error and planner_trace:
        ok, failures = _verify_expectation(tc, planner_trace)

    lines: list[str] = []
    lines.append(f"# {tc.test_id} — {tc.description}")
    lines.append("")
    lines.append(f"- **Status**: {'PASS' if ok else 'FAIL'}")
    lines.append(f"- **Elapsed**: {elapsed_seconds:.1f}s")
    lines.append(f"- **Audio**: {', '.join(str(p) for p in tc.audio_files)}")
    lines.append(f"- **Tools registered**: {', '.join(tc.tools)}")
    lines.append(f"- **Question**: {tc.question}")
    lines.append("")
    lines.append("## Expectation")
    exp = tc.expectation
    lines.append(f"- Round bounds: {exp.min_rounds}–{exp.max_rounds}")
    lines.append(f"- Must call tools: {exp.must_call_tools}")
    lines.append(f"- Final action: `{exp.final_action}`")
    lines.append(f"- Requires parallel round: {exp.requires_parallel_round}")
    if failures:
        lines.append("")
        lines.append("### Failures")
        for f in failures:
            lines.append(f"- {f}")
    if error:
        lines.append("")
        lines.append("### Error")
        lines.append("```")
        lines.append(error)
        lines.append("```")

    lines.append("")
    lines.append(f"## Rounds executed: {len(planner_trace)}")
    if planner_trace:
        lines.append("")
        for i, decision in enumerate(planner_trace, start=1):
            payload = _serialize_decision(decision)
            lines.append(f"### Round {i}: action=`{payload['action']}`")
            lines.append("")
            lines.append(f"- Rationale: {payload['rationale'] or '(empty)'}")
            if payload["tool_calls"]:
                lines.append(f"- Tool calls ({len(payload['tool_calls'])}):")
                for j, tc_dump in enumerate(payload["tool_calls"], start=1):
                    lines.append(
                        f"  - {j}. `{tc_dump['tool_name']}({json.dumps(tc_dump['args'], ensure_ascii=False)})`"
                    )
            if payload["selected_audio_ids"]:
                lines.append(f"- Frontend audio_ids: {payload['selected_audio_ids']}")
            if payload["frontend_followup_prompt"]:
                lines.append(f"- Frontend prompt: {payload['frontend_followup_prompt']}")
            lines.append("")

    lines.append("## Final state summary")
    lines.append(f"- status: {final_state.get('status') if final_state else 'unknown'}")
    lines.append(f"- step_count: {final_state.get('step_count') if final_state else 'unknown'}")
    audio_list = (final_state or {}).get("audio_list", []) or []
    lines.append(f"- audio_list ({len(audio_list)}):")
    for a in audio_list:
        lines.append(f"  - {a.audio_id}: {a.description} (source: {a.source})")
    fa = (final_state or {}).get("final_answer")
    if fa is not None:
        lines.append(f"- final_answer: {fa.answer if hasattr(fa, 'answer') else fa}")

    if system_message:
        lines.append("")
        lines.append("## System message (first round, truncated)")
        lines.append("```")
        lines.append(system_message[:6000])
        if len(system_message) > 6000:
            lines.append("... (truncated)")
        lines.append("```")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


async def _run_one_test(
    tc: TestCase,
    api_key: str,
    frontend_model: str,
    planner_model: str,
    base_url: str,
    max_steps: int,
) -> tuple[bool, Path]:
    """Run one TestCase end-to-end and write its report."""
    print(f"\n{'=' * 70}")
    print(f"  TEST: {tc.test_id}")
    print(f"  {tc.description}")
    print(f"{'=' * 70}")

    # Quick check that fixtures exist before we boot any servers.
    for p in tc.audio_files:
        if not p.exists():
            err = f"Fixture not found: {p}"
            print(f"\n[SKIP] {err}")
            report = _write_report(
                tc=tc,
                final_state={},
                system_message=None,
                user_messages=[],
                raw_tool_calls_per_round=[],
                error=err,
                elapsed_seconds=0.0,
            )
            return False, report

    config = AgentConfig(max_steps=max_steps, debug=True)
    frontend = OpenAICompatibleFrontend(
        model=frontend_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.05,
        max_tokens=4096,
    )
    planner = OpenAICompatiblePlanner(
        model=planner_model,
        api_key=api_key,
        base_url=base_url,
        enable_thinking=False,
        temperature=0.05,
        max_tokens=4096,
    )
    registry = ToolRegistry()
    fuser = DefaultEvidenceFuser()
    server_manager = MCPServerManager()

    captured_messages: dict[str, Any] = {"system": None, "users": []}

    # Hook the planner's build_decision_system/user methods so we can save
    # what the model actually saw on each round (for the markdown report).
    orig_sys = planner.build_decision_system_prompt
    orig_user = planner.build_decision_user_instruction

    def wrapped_system(state, available_tools):
        s = orig_sys(state, available_tools)
        if captured_messages["system"] is None:
            captured_messages["system"] = s
        return s

    def wrapped_user(state, available_tools):
        u = orig_user(state, available_tools)
        captured_messages["users"].append(u)
        return u

    planner.build_decision_system_prompt = wrapped_system  # type: ignore[assignment]
    planner.build_decision_user_instruction = wrapped_user  # type: ignore[assignment]

    error_msg: str | None = None
    final_state: dict = {}
    start = datetime.now()
    try:
        print(f"\nRegistering MCP tools from catalogs: {tc.tools}")
        await register_all_mcp_tools(
            registry=registry,
            server_manager=server_manager,
            tool_names=tc.tools,
            verbose=False,
        )
        print(f"Registered {len(registry.list_specs())} tools.")

        agent = AudioAgent(
            frontend=frontend,
            planner=planner,
            registry=registry,
            fuser=fuser,
            config=config,
        )

        audio_paths = [str(p.resolve()) for p in tc.audio_files]
        final_state = await agent.arun(
            question=tc.question,
            audio_paths=audio_paths,
            max_steps=max_steps,
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        print(f"\n[ERROR] {error_msg.splitlines()[0]}")
    finally:
        try:
            await server_manager.shutdown_all()
        except Exception as e:
            print(f"(MCP shutdown failed: {e})")

    elapsed = (datetime.now() - start).total_seconds()

    report = _write_report(
        tc=tc,
        final_state=final_state or {},
        system_message=captured_messages["system"],
        user_messages=captured_messages["users"],
        raw_tool_calls_per_round=[],
        error=error_msg,
        elapsed_seconds=elapsed,
    )

    ok = error_msg is None and final_state.get("status") == AgentStatus.ANSWERED
    print(f"\n→ report: {report}")
    print(f"→ result: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
    return ok, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Option 2 (native function calling) validation harness.",
    )
    parser.add_argument(
        "--suite",
        choices=["cpu", "gpu", "all"],
        default="cpu",
        help="Which test suite to run. CPU-only by default.",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=None,
        help="Optional subset of test_ids to run (e.g. t1_smoke_single_tool).",
    )
    parser.add_argument(
        "--frontend-model",
        default="qwen3.5-omni-plus",
        help="DashScope frontend model name.",
    )
    parser.add_argument(
        "--planner-model",
        default="qwen3.5-plus",
        help="DashScope planner model name.",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="DashScope OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DashScope API key. Reads DASHSCOPE_API_KEY if not given.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Per-run planner round budget.",
    )
    return parser.parse_args()


async def _amain() -> int:
    args = _parse_args()
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: provide --api-key or set DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    setup_logger()
    set_debug_mode(True)

    # Filter test list by suite / explicit names.
    selected: list[TestCase] = []
    for t in ALL_TESTS.values():
        if args.suite != "all" and t.suite != args.suite:
            continue
        if args.tests and t.test_id not in args.tests:
            continue
        selected.append(t)

    if not selected:
        print("No tests selected.", file=sys.stderr)
        return 2

    print(f"Running {len(selected)} test(s) → {', '.join(t.test_id for t in selected)}")
    print(f"Artifacts → {ARTIFACTS_DIR}")

    results: list[tuple[TestCase, bool, Path]] = []
    for tc in selected:
        ok, report = await _run_one_test(
            tc,
            api_key=api_key,
            frontend_model=args.frontend_model,
            planner_model=args.planner_model,
            base_url=args.base_url,
            max_steps=args.max_steps,
        )
        results.append((tc, ok, report))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    pass_count = sum(1 for _, ok, _ in results if ok)
    for tc, ok, report in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {tc.test_id} → {report}")
    print(f"\n  {pass_count}/{len(results)} tests passed")
    return 0 if pass_count == len(results) else 1


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
