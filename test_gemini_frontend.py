#!/usr/bin/env python3
"""
Validation tests for GeminiFrontend.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from audio_agent.frontend.gemini_frontend import GeminiFrontend
from audio_agent.core.errors import FrontendError

TEST_AUDIO = "/cpfs/user/jingpeng/tmp/000000.wav"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "1561cda555764bed97c8acba1ac46f92")


def test_import_and_init():
    print("=" * 60)
    print("TEST 1: Import and Initialization")
    print("=" * 60)
    frontend = GeminiFrontend(api_key=GEMINI_API_KEY)
    assert frontend.name == "gemini_frontend"
    assert frontend.model_handle is None
    print("✓ GeminiFrontend initialized successfully")
    print(f"  name: {frontend.name}")
    print(f"  base_url: {frontend._base_url}")
    print()


def test_encode_audio():
    print("=" * 60)
    print("TEST 2: Audio Base64 Encoding")
    print("=" * 60)
    frontend = GeminiFrontend(api_key=GEMINI_API_KEY)

    # Test existing file
    b64 = frontend._encode_audio(TEST_AUDIO)
    assert len(b64) > 0
    print(f"✓ Encoded {TEST_AUDIO}")
    print(f"  base64 length: {len(b64)} chars")

    # Test missing file
    try:
        frontend._encode_audio("/nonexistent/file.wav")
        assert False, "Should have raised FrontendError"
    except FrontendError as e:
        print(f"✓ Correctly raised FrontendError for missing file")
        print(f"  error: {e.message}")
    print()


def test_build_gemini_payload():
    print("=" * 60)
    print("TEST 3: Build Gemini Payload")
    print("=" * 60)
    frontend = GeminiFrontend(api_key=GEMINI_API_KEY)
    payload = frontend._build_gemini_payload(
        system_prompt="You are a helpful assistant.",
        user_text="Describe this audio.",
        audio_paths=[TEST_AUDIO],
    )

    assert "generationConfig" in payload
    assert "system_instruction" in payload
    assert "contents" in payload
    assert len(payload["contents"]) == 1
    assert payload["contents"][0]["role"] == "user"
    parts = payload["contents"][0]["parts"]
    assert len(parts) == 2  # audio + text
    assert "inline_data" in parts[0]
    assert parts[0]["inline_data"]["mime_type"] == "audio/wav"
    assert "text" in parts[1]
    print("✓ Payload structure validated")
    print(f"  generationConfig.temperature: {payload['generationConfig']['temperature']}")
    print(f"  generationConfig.maxOutputTokens: {payload['generationConfig']['maxOutputTokens']}")
    print(f"  system_instruction: {payload['system_instruction']['parts'][0]['text'][:50]}...")
    print(f"  parts count: {len(parts)} (audio + text)")
    print()


def test_api_call():
    print("=" * 60)
    print("TEST 4: Real Gemini API Call")
    print("=" * 60)
    frontend = GeminiFrontend(api_key=GEMINI_API_KEY)
    payload = frontend._build_gemini_payload(
        system_prompt="You are an audio analysis assistant. Listen to the audio and describe what you hear in one sentence.",
        user_text="What is in this audio?",
        audio_paths=[TEST_AUDIO],
    )

    try:
        response = frontend._call_gemini_api(payload)
        assert len(response) > 0
        print("✓ API call succeeded")
        print(f"  response length: {len(response)} chars")
        print(f"  response preview: {response[:200]}...")
    except FrontendError as e:
        print(f"✗ API call failed: {e.message}")
        print(f"  details: {e.details}")
        raise
    print()


def test_call_model_full_pipeline():
    print("=" * 60)
    print("TEST 5: Full Pipeline (build_model_input + call_model)")
    print("=" * 60)
    frontend = GeminiFrontend(api_key=GEMINI_API_KEY)

    model_input = frontend.build_api_model_input(
        question="What is in this audio?",
        audio_paths=[TEST_AUDIO],
        question_oriented_prompt="Focus on identifying the main sounds.",
    )

    assert model_input.metadata["frontend_name"] == "gemini_frontend"
    assert "gemini_payload" in model_input.metadata
    print("✓ Model input built successfully")

    try:
        raw_output = frontend.call_model(model_input)
        assert isinstance(raw_output, str)
        assert len(raw_output) > 0
        print("✓ call_model returned text")
        print(f"  output length: {len(raw_output)} chars")
        print(f"  output preview: {raw_output[:200]}...")

        normalized = frontend.normalize_model_output(raw_output, model_input)
        assert normalized.question_guided_caption == raw_output.strip()
        print("✓ normalize_model_output succeeded")
    except FrontendError as e:
        print(f"✗ Pipeline failed: {e.message}")
        print(f"  details: {e.details}")
        raise
    print()


def test_final_answer_generation():
    print("=" * 60)
    print("TEST 6: Final Answer Generation")
    print("=" * 60)
    frontend = GeminiFrontend(api_key=GEMINI_API_KEY)

    # Minimal context for final answer
    context = {
        "evidence_summary": "The audio contains speech.",
        "evidence_log": [],
        "planner_trace": [],
        "tool_call_history": [],
        "initial_plan": None,
        "initial_frontend_output": None,
        "audio_list": [],
        "expected_output_format": "One sentence.",
    }

    try:
        model_input = frontend.build_final_answer_model_input(
            question="What is in this audio?",
            audio_paths=[TEST_AUDIO],
            context=context,
        )
        assert "gemini_payload" in model_input.metadata
        print("✓ Final answer model input built successfully")

        raw_output = frontend.call_model(model_input)
        assert isinstance(raw_output, str)
        assert len(raw_output) > 0
        print("✓ Final answer API call succeeded")
        print(f"  output length: {len(raw_output)} chars")
        print(f"  output preview: {raw_output[:200]}...")
    except FrontendError as e:
        print(f"✗ Final answer generation failed: {e.message}")
        print(f"  details: {e.details}")
        raise
    print()


def main():
    print("\n" + "=" * 60)
    print("GeminiFrontend Validation Tests")
    print("=" * 60 + "\n")

    global TEST_AUDIO
    if not Path(TEST_AUDIO).exists():
        print(f"⚠ Test audio not found: {TEST_AUDIO}")
        print("Searching for alternative...")
        alt = "/cpfs/user/jingpeng/workspace/nfs/data/test/MMAR/audio/f0VchKwpMAk_00-11-10_00-11-30.wav"
        if Path(alt).exists():
            TEST_AUDIO = alt
            print(f"Using: {TEST_AUDIO}\n")
        else:
            print("No suitable audio found. Tests requiring audio will be skipped.")
            return

    tests = [
        test_import_and_init,
        test_encode_audio,
        test_build_gemini_payload,
        test_api_call,
        test_call_model_full_pipeline,
        test_final_answer_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    if failed == 0:
        print("\n🎉 All tests passed! GeminiFrontend is working correctly.")
    else:
        print(f"\n⚠ {failed} test(s) failed. Check output above for details.")


def test_final_answer_integration():
    print("=" * 60)
    print("TEST 6: Final Answer Generation (generate_final_answer)")
    print("=" * 60)
    frontend = GeminiFrontend(api_key=GEMINI_API_KEY)

    context = {
        "evidence_summary": "The audio contains speech.",
        "evidence_log": [],
        "planner_trace": [],
        "tool_call_history": [],
        "initial_plan": None,
        "initial_frontend_output": None,
        "audio_list": [],
        "expected_output_format": "One sentence.",
    }

    try:
        answer = frontend.generate_final_answer(
            question="What is in this audio?",
            audio_paths=[TEST_AUDIO],
            context=context,
        )
        assert isinstance(answer, str)
        assert len(answer) > 0
        print("✓ generate_final_answer succeeded")
        print(f"  answer length: {len(answer)} chars")
        print(f"  answer preview: {answer[:200]}...")
    except FrontendError as e:
        print(f"✗ Final answer generation failed: {e.message}")
        print(f"  details: {e.details}")
        raise
    print()


if __name__ == "__main__":
    main()
