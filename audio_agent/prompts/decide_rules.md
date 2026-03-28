1. If you have enough evidence to answer the question, use action='answer' and provide draft_answer.
2. If you need more information, use action='call_tool' and follow this decision process:
   - First, identify what kind of evidence is missing to answer the question
   - Then, determine which capability family can provide that evidence (e.g., ASR for transcription, diarization for speaker separation, captioning for audio description)
   - Finally, select the specific concrete tool from available_tools that matches the needed capability
3. If the intent or expected output format is unclear, use action='clarify_intent' to reason about it.
4. action='call_tool' REQUIRES a non-empty selected_tool_name - never leave it null or empty.
5. action='answer' REQUIRES a non-empty draft_answer.
6. action='clarify_intent' uses reasoning only - do not call tools.
7. Do NOT use action='call_tool' if you are ready to answer - use action='answer' instead.
