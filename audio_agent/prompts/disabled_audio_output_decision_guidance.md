# Disabled Audio Output Decision Guidance

This file preserves older decision-stage guidance for audio-output tasks.
It is not loaded by `load_prompt()` unless explicitly referenced by code.

Reason disabled:
The current workflow routes `action="answer"` to `frontend_final_answer_node`.
The decision stage no longer directly generates the final user-facing answer,
so audio-output delivery needs a redesign before this guidance is active again.

## Previous Decide System Guidance

**Important Guidelines for Audio Output Tasks:**

1. **When to Answer:** If the initial plan indicates `requires_audio_output: true`,
   do NOT answer until the audio has been successfully generated. You should see
   a new audio file in Available Audio Files (e.g., audio_1, audio_2) that was
   produced by a tool.

2. **How to Reference Audio in Answer:** When providing draft_answer for tasks
   that produce audio output:
   - Reference the output audio by its ID (e.g., "audio_1")
   - Say "the output audio file" or "the processed audio"
   - Do NOT include raw file paths (like /tmp/... or /output/...) in draft_answer
   - The exact file path will be provided separately in the structured output

3. **What to Include:** In your draft_answer, focus on:
   - Confirming the processing was completed as requested
   - Describing what was done (e.g., "converted from stereo to mono")
   - Mentioning the audio ID (e.g., "available as audio_1")
   - Including relevant technical details (duration, sample rate, channels)

## Previous Decide Rules Guidance

**Audio Output Rule:** If the task requires producing an audio file
(`requires_audio_output` is true), verify that a new audio file has been
generated before answering. Check Available Audio Files for entries with
source != "original". Only answer when the output audio exists.

**Answer Content Rule:** When `action="answer"`, do NOT include raw file paths
in `draft_answer`. Reference output audio by ID, such as "available as audio_1",
or say "the output audio file". The exact path will be provided separately.
