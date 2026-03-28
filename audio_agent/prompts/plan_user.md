Question: {question}

Produce an InitialPlan JSON object with keys:
- `approach` (str): High-level approach to answer the question
- `focus_points` (list[str]): Key points to investigate in the audio
- `possible_tool_types` (list[str]): Tool types that might help (e.g., "asr", "diarization", "captioning")
- `clarified_intent` (str | null): What the question is actually asking
- `expected_output_format` (str | null): Expected format of the answer (e.g., "single sentence", "bullet points")
- `notes` (str, optional): Additional notes or considerations

If the intent is unclear, express uncertainty in focus_points or notes.
