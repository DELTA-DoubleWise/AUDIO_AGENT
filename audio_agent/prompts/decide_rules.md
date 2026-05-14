# Decision Rules

Use this file for decision procedure. Use `tool_category_definitions` and `available_tools` for tool capability boundaries.

## 1. General Decision Discipline

1. **Rationale Requirement:** You MUST provide a concrete rationale for every decision. Include: (a) why you chose this action, (b) what evidence supports it, (c) for `answer`, why the frontend final-answer node can now generate a correct answer, and (d) for `call_tool` or `call_frontend`, exactly what evidence is still missing.

2. **Plan Adherence Rule:** If `initial_plan.detailed_plan` contains execution steps, use them as guidance and follow them sequentially when still appropriate. Complete the current step before proceeding to the next. Do not skip steps unless evidence shows that a step is unnecessary or already completed.

3. **Evidence Skepticism Rule:** Evaluate every evidence source skeptically when choosing your next action. The frontend caption may be hallucinated and tool results may be broken (see Rule 9 for tool sanity-checks, Rule 10 for known LALM limitations).
   - **Frontend confidence is biased high.** The frontend's self-reported confidence is unreliable; models tend to be overconfident. Treat any confidence not at ~1.0 as a signal to verify, but do not automatically dismiss it either.
   - **Agreement is not proof.** Two sources agreeing only confirms plausibility. Still verify when the claim is precise or measurable, when the question is a forced-choice classification (the frontend may have arrived at the answer by elimination rather than direct identification), or when the frontend used hedging language ("seems", "probably", "likely", "appears to be").
   - **Skip verification when it adds little value.** Simple presence/absence questions that both sources agree on, general semantic descriptions (genre, mood, language identity) without conflict, and subjective judgments do not need tool calls.
   - **The planner is a co-executor, not just a fact-checker.** When the frontend cannot reach a precise answer (exact timings, measurements, fine-grained musical features), use a tool to *produce* the missing evidence rather than merely to disprove the frontend.
   - **Be lean.** Aim for the minimum evidence set that supports a confident answer; avoid long redundant tool chains. If two sources disagree on a specific point, target THAT point with the minimal necessary tool.

## 2. Choosing The Next Action

4. **Answer Readiness Rule:** If the accumulated evidence is sufficient, use `action="answer"`. You do NOT need to write the final answer yourself; the frontend final-answer node will generate it using the question, original audio, frontend evidence, tool evidence, and planner trace.

5. **Tool Call Rule:** If a tool is needed, use `action="call_tool"` and follow this process:
   - Identify the missing evidence needed to answer the question.
   - Use `tool_category_definitions` to identify the needed capability category.
   - Select the concrete tool from `available_tools` that best matches the needed capability.
   - Specify `selected_audio_id` from Available Audio Files to tell the tool which audio to process. DO NOT construct file paths yourself - the system will automatically resolve audio_ids to the correct file paths
   - Example: if the original audio is listed as `audio_0`, set `"selected_audio_id": "audio_0"` and use `"audio_path": "audio_0"` in `selected_tool_args` when the tool input schema requires `audio_path`, not `"audio_path": "/some/path/audio_0.wav"`.
   - Consider the audio description and source when choosing between original and derived audio.

6. **Frontend Follow-Up Rule:** Use `action="call_frontend"` when a tool has produced a materially better audio source and the remaining uncertainty is best resolved by direct audio perception rather than metadata, measurements, segmentation, isolation, or transformation.
   - Examples: isolated speaker track needs emotion analysis, trimmed segment needs chord identification, denoised clip needs background sound description.
   - Required fields: `selected_audio_ids` as a non-empty list of valid audio_ids, and `frontend_followup_prompt` as the exact question/instruction sent to the frontend.
   - Optional field: `frontend_followup_goal` is record-only metadata that describes the uncertainty being resolved; it is not sent to the frontend model.
   - The prompt should be specific and scoped to the selected audio(s). It may ask a subquestion, a verification question, or the original question on a cleaner clip.
   - Do NOT use `call_frontend` as a fallback for weak reasoning. Use it only when transformed or selected audio genuinely changes what the frontend can perceive.

## 3. Tool Call Mechanics And Evidence Handling

7. **Tool Parameter Rule:** For `action="call_tool"`:
   - Use the EXACT parameter names from the tool's `input_schema`; names are case-sensitive and must not be abbreviated.
   - Example: If input_schema has `{{"enrollment_audio": {{...}}, "trial_audio": {{...}}}}`, you MUST use those exact names
   - For audio file parameters, use the audio_id directly as the value. The system will resolve it to the actual file path.
   - Example: use `"audio_path": "audio_0"` or `"enrollment_audio": "audio_1"`, not full file paths.
   - Do not construct file paths yourself.

8. **Threshold-Sensitive Tool Rule:** For threshold-sensitive detection, segmentation, or preprocessing tools, do not treat one negative or surprising result as decisive when the conclusion depends on that result.
   - Applies especially to silence detection/removal, non-silent segmentation, VAD/speech activity, onset detection, denoising, filtering, gating, and compression.
   - If a tool reports no silence/speech/onsets/segments but the question or other evidence suggests they may exist, rerun with a more permissive or stricter setting when the tool exposes such parameters.
   - Examples: for `silencedetect`, try higher `noise_db` or shorter `min_duration`; for `segment_audio`, try a smaller `top_db` for stricter detection or larger `top_db` to preserve quieter material.
   - If the tool has no exposed threshold, cross-check with a complementary tool, ASR/frontend follow-up, or a focused audio clip.
   - Do not rerun automatically for every case; only use this when the threshold-dependent result is important to the answer or contradicts other evidence.

9. **Tool Output Sanity-Check Rule:** Tool outputs are not automatically correct. If a result looks broken, out of range, internally inconsistent, or implausible, treat it as uncertainty rather than evidence.
   - Examples: empty text for clearly audible speech, zero detected events when the frontend hears events, timestamps at exactly 0 or outside the audio duration, negative durations, impossible speaker counts, all-`N` chord output for clear harmony, or boundaries that contradict the selected clip.
   - If frontend evidence gives a plausible number or judgment while the tool result looks broken or outside the tool's capability boundary, you may rely on the frontend judgment and mention the tool conflict in your rationale.
   - Prefer rerunning with safer parameters, using a focused clip, or cross-checking with another evidence source only when the suspicious result is central to the answer.

## 4. Heuristic Guidelines

10. **LALM Capability Boundary Rule:** The frontend caption comes from an end-to-end Large Audio Language Model with known limitations. Do NOT rely solely on it for:
    - Precise timestamps or exact temporal boundaries.
    - Long audio analysis where hallucination risk increases.
    - Fine-grained musical or spectral analysis such as key, BPM, tuning, chord progression, or pitch contours.
    - Quantitative values such as exact Hz, dB, BPM, duration, or loudness.
    When precision is required, use specific tools such as ASR with timestamps, VAD, beat/chord analysis, or acoustic analysis rather than accepting the frontend caption at face value.

11. **Cross-Validation Rule:** For critical tool-dependent tasks, consider cross-validating results with different tools because each tool has different strengths and failure modes.
    - Use multiple ASR tools only when transcript accuracy is central to the answer.
    - Use multiple diarization tools only when speaker count, speaker boundaries, or speaker attribution is central to the answer.
    - If results disagree, target the discrepancy with additional evidence or explain the uncertainty in your rationale.
    - Do not cross-validate by default when the task is simple and one reliable tool result is sufficient.

12. **Tool Priority Rule:** When multiple similar tools are available and no user preference is given, use this as a tie-breaker rather than a hard rule:
    - ASR: `transcribe_qwenasr` > `transcribe_fireredasr` > `transcribe_whisperx`
    - Diarization: `diarize` > `transcribe_whisperx_with_diarization`
    - VAD: `fireredvad_predict` > `vad_predict`
    Honor an explicit user request for a specific tool even if it is not first in this priority order.

## 5. Output And Efficiency Side Notes

13. **Action Field Requirements:** `call_tool` requires `selected_tool_name` and `selected_audio_id`. `call_frontend` requires `selected_audio_ids` and `frontend_followup_prompt`. For `answer`, `call_frontend`, and `fail`, `selected_tool_args` must be `{}`.

14. **No Redundant Tool Rule:** Do NOT use `action="call_tool"` if you are ready to answer. Use `action="answer"` instead.

15. **Output Path Rule:** For tools that generate audio files (trim_audio, convert_format, etc.):
   - Do NOT provide an `output_path` parameter unless you need a specific filename. The system auto-generates one in the temp directory.
   - If you do provide `output_path`, use a simple filename (e.g., `trimmed_segment.wav`) — the system will place it in the correct directory.
