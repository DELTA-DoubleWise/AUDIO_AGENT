#!/usr/bin/env python3
"""
Run API-based audio agent on selected MMAR samples and compare with mmar_v3 results.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from audio_agent.config.settings import AgentConfig
from audio_agent.core.constants import AgentStatus
from audio_agent.core.logging import setup_logger, set_debug_mode
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
from audio_agent.frontend.openai_compatible_frontend import OpenAICompatibleFrontend
from audio_agent.frontend.gemini_frontend import GeminiFrontend
from audio_agent.main import AudioAgent
from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.tools.catalog import register_all_mcp_tools

API_KEY = "sk-f8ae3fc37bdd4953977e813f77b7324f"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
FRONTEND_MODEL = "qwen3.5-omni-plus"
PLANNER_MODEL = "qwen3.5-plus"

MMAR_AUDIO_DIR = "/cpfs/user/jingpeng/workspace/nfs/data/test/MMAR/audio"
MMAR_META_PATH = "/cpfs/user/jingpeng/workspace/nfs/data/test/MMAR/MMAR-meta.json"
MMAR_V3_RESULTS_PATH = "/cpfs/user/jingpeng/workspace/AUDIO_AGENT/test_result/mmar_v3_extracted/results.json"


def load_mmar_v3_results():
    with open(MMAR_V3_RESULTS_PATH) as f:
        data = json.load(f)
    return {r["id"]: r for r in data["results"]}


def load_mmar_meta():
    with open(MMAR_META_PATH) as f:
        return json.load(f)


def extract_choice_from_answer(answer: str, choices: list[str]) -> str | None:
    """Extract which choice the answer corresponds to."""
    answer_clean = answer.strip().lower()
    for choice in choices:
        if choice.lower() in answer_clean:
            return choice
    # Try exact match
    for choice in choices:
        if answer_clean == choice.lower():
            return choice
    return None


USE_GEMINI = True  # Set to True to use Gemini frontend
GEMINI_API_KEY = "1561cda555764bed97c8acba1ac46f92"
GEMINI_BASE_URL = "https://runway.devops.rednote.life/openai/google/v1:generateContent"


async def run_single_sample(sample: dict, v3_result: dict, registry, server_manager) -> dict:
    """Run agent on a single sample."""
    audio_path = os.path.join(MMAR_AUDIO_DIR, os.path.basename(sample["audio_path"]))
    question = sample["question"]
    choices = sample["choices"]
    ground_truth = sample["answer"]
    sample_id = sample["id"]

    if not os.path.exists(audio_path):
        return {
            "id": sample_id,
            "error": f"Audio not found: {audio_path}",
            "correct": False,
        }

    if USE_GEMINI:
        frontend = GeminiFrontend(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
            temperature=0.01,
            max_tokens=40960,
        )
    else:
        frontend = OpenAICompatibleFrontend(
            model=FRONTEND_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.05,
            max_tokens=4096,
        )
    planner = OpenAICompatiblePlanner(
        model=PLANNER_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        enable_thinking=True,
        temperature=0.05,
        max_tokens=4096,
    )
    fuser = DefaultEvidenceFuser()

    config = AgentConfig(max_steps=5, debug=True)
    agent = AudioAgent(
        frontend=frontend,
        planner=planner,
        registry=registry,
        fuser=fuser,
        config=config,
    )

    # Build question with choices
    question_with_choices = (
        f"{question}\n\n"
        f"Please choose the best answer from the following options:\n" +
        "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    )

    try:
        final_state = await agent.arun(
            question=question_with_choices,
            audio_paths=[audio_path],
            max_steps=10,
        )
    except Exception as e:
        return {
            "id": sample_id,
            "error": str(e),
            "correct": False,
            "agent_status": "error",
            "prediction_raw": "",
            "extracted_choice": None,
            "ground_truth": ground_truth,
            "step_count": 0,
            "tool_calls": 0,
            "confidence": 0.0,
            "v3_prediction": v3_result.get("prediction"),
            "v3_correct": v3_result.get("correct"),
            "v3_extracted_choice": v3_result.get("extracted_choice"),
        }

    status = final_state.get("status", AgentStatus.RUNNING)
    final_answer = final_state.get("final_answer")
    answer_text = final_answer.answer if final_answer else ""
    confidence = final_answer.confidence if final_answer else 0.0

    extracted = extract_choice_from_answer(answer_text, choices)
    correct = extracted == ground_truth if extracted else False

    return {
        "id": sample_id,
        "question": question,
        "choices": choices,
        "ground_truth": ground_truth,
        "prediction_raw": answer_text,
        "extracted_choice": extracted,
        "correct": correct,
        "agent_status": status.value if hasattr(status, "value") else str(status),
        "step_count": final_state.get("step_count", 0),
        "tool_calls": len(final_state.get("tool_call_history", [])),
        "confidence": confidence,
        "v3_prediction": v3_result.get("prediction"),
        "v3_correct": v3_result.get("correct"),
        "v3_extracted_choice": v3_result.get("extracted_choice"),
    }


async def main():
    setup_logger()
    set_debug_mode(True)

    meta = load_mmar_meta()
    v3_results = load_mmar_v3_results()

    # Select 3 diverse samples:
    # 1. First sample (Parrot) - v3 correct
    # 2. BV1j1ABeqEyp - v3 wrong (interesting)
    # 3. BV1gt4y1e7U5 (gunshots counting) - v3 correct, uses tools
    selected_ids = [
        "f0VchKwpMAk_00-11-10_00-11-30",
        "BV1gt4y1e7U5_00-00-52_00-01-16",
    ]

    selected_samples = [m for m in meta if m["id"] in selected_ids]

    print("=" * 70)
    print("MMAR API Agent Test - Selected Samples")
    print("=" * 70)
    for s in selected_samples:
        v3 = v3_results.get(s["id"], {})
        print(f"\nID: {s['id']}")
        print(f"Question: {s['question']}")
        print(f"Choices: {s['choices']}")
        print(f"Ground Truth: {s['answer']}")
        print(f"V3 Prediction: {v3.get('prediction')} (correct={v3.get('correct')})")

    # Set up shared components
    registry = ToolRegistry()
    server_manager = MCPServerManager()

    print("\n" + "=" * 70)
    print("Registering MCP tools...")
    print("=" * 70)
    registered_tools = await register_all_mcp_tools(
        registry=registry,
        server_manager=server_manager,
        verbose=True,
    )
    print(f"Registered {len(registered_tools)} tools")

    results = []
    for sample in selected_samples:
        print("\n" + "=" * 70)
        print(f"Running sample: {sample['id']}")
        print("=" * 70)
        v3 = v3_results.get(sample["id"], {})
        result = await run_single_sample(sample, v3, registry, server_manager)
        results.append(result)

        print(f"\nStatus: {result['agent_status']}")
        print(f"Raw Answer: {result['prediction_raw']}")
        print(f"Extracted Choice: {result['extracted_choice']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Correct: {result['correct']}")
        print(f"Steps: {result['step_count']}, Tool Calls: {result['tool_calls']}")
        print(f"V3 Prediction: {result['v3_prediction']} (correct={result['v3_correct']})")

    await server_manager.shutdown_all()

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    correct_count = sum(1 for r in results if r["correct"])
    v3_correct_count = sum(1 for r in results if r["v3_correct"])

    print(f"\nOur Agent: {correct_count}/{len(results)} correct")
    print(f"V3 Agent:  {v3_correct_count}/{len(results)} correct")

    print("\n" + "-" * 70)
    print(f"{'ID':<45} {'Ours':<20} {'V3':<20} {'GT':<20}")
    print("-" * 70)
    for r in results:
        ours = r["extracted_choice"] or "(none)"
        v3 = r["v3_extracted_choice"] or "(none)"
        gt = r["ground_truth"]
        print(f"{r['id']:<45} {ours:<20} {v3:<20} {gt:<20}")

    # Save results
    output_dir = "/cpfs/user/jingpeng/workspace/AUDIO_AGENT/test_result/examples"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mmar_gemini_api_comparison.json")
    with open(output_path, "w") as f:
        json.dump({
            "frontend_model": "gemini-2.5-pro",
            "planner_model": PLANNER_MODEL,
            "results": results,
            "summary": {
                "our_correct": correct_count,
                "v3_correct": v3_correct_count,
                "total": len(results),
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
