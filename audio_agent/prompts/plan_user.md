Question: {question}

Frontend Caption:
{frontend_caption}

Produce an InitialPlan JSON object with these keys:

- `approach` (str): High-level strategy. Include task mode: `direct_perception`, `verified_perception`, or `decomposed_evidence_construction`.
- `focus_points` (list[str]): Concrete evidence gaps or audio aspects to inspect.
- `possible_tool_types` (list[str]): Relevant tool categories only, such as `asr`, `diarization`, `vad`, `chord_recognition`, `audio_processing`, `acoustic_analysis`, or `frontend_followup`.
- `clarified_intent` (str | null): What the question is asking.
- `expected_output_format` (str | null): Expected final answer format.
- `requires_audio_output` (bool): Whether the user asks for a processed/generated audio file.
- `notes` (str, optional): Concise rationale, known vulnerability, or operation chain.
- `detailed_plan` (list[ExecutionStep], optional): Sequential plan only when useful.

## Planning Steps

1. Diagnose task mode.
   - Use `direct_perception` for holistic recognition or semantic perception.
   - Use `verified_perception` for direct questions involving known frontend weaknesses such as exact speech, speaker count, timestamps, chords/key, tempo, pitch, loudness, or duration.
   - Use `decomposed_evidence_construction` when the question needs locating, separating, transforming, extracting, measuring, or comparing intermediate evidence.

2. Use frontend evidence.
   - If the caption is clear and the task is direct perception, keep `detailed_plan` empty.
   - If the caption is uncertain or conflicts with the observer direct answer appended below, plan targeted verification.
   - Do not trust self-reported confidence blindly.

3. Choose tool use policy.
   - Direct perception: avoid tools unless there is a specific evidence gap.
   - Verified perception: use one or a few narrow expert tools as incremental evidence.
   - Decomposed evidence construction: create an operation-level chain using `locate`, `separate`, `transform`, `symbolic_extraction`, `acoustic_measurement`, and/or `compare`.

4. Decide whether derived audio helps.
   - If trimming, separation, denoising, normalization, or filtering would produce better evidence, include it in `detailed_plan`.
   - If a derived audio artifact should be re-perceived by the frontend, include `frontend_followup` as a possible tool type or step.

## Audio Output Detection

Set `requires_audio_output: true` only when the final deliverable is processed audio, such as trim, cut, merge, mix, denoise, normalize, filter, convert, extract, or separate.

Set `requires_audio_output: false` for questions asking for information about audio, such as transcription, content, speaker identity, scene, metadata, or analysis.

## Detailed Plan Rules

For simple direct perception tasks:

```json
{{
  "detailed_plan": []
}}
```

For verified perception, keep the plan short:

```json
{{
  "detailed_plan": [
    {{
      "step_number": 1,
      "description": "Verify the vulnerable aspect with a narrow expert tool",
      "tool_type": "asr",
      "expected_output": "Transcript evidence for the exact spoken phrase"
    }}
  ]
}}
```

For decomposed evidence construction, use operation-level steps:

```json
{{
  "detailed_plan": [
    {{
      "step_number": 1,
      "description": "Locate the alarm event and the following speech segment",
      "tool_type": "audio_localization",
      "expected_output": "Relevant time span"
    }},
    {{
      "step_number": 2,
      "description": "Separate or trim the target segment for focused analysis",
      "tool_type": "audio_processing",
      "expected_output": "Derived audio artifact for the target region"
    }},
    {{
      "step_number": 3,
      "description": "Extract symbolic evidence from the derived audio",
      "tool_type": "asr",
      "expected_output": "Transcript of the target region"
    }}
  ]
}}
```

If the intent is unclear, state the uncertainty in `focus_points` or `notes` rather than inventing details.
