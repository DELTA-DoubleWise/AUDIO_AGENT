Question: {question}

Produce an InitialPlan JSON object with keys:
- `approach` (str): High-level approach to answer the question
- `focus_points` (list[str]): Key points to investigate in the audio
- `possible_tool_types` (list[str]): Tool types that might help (e.g., "asr", "diarization", "captioning")
- `clarified_intent` (str | null): What the question is actually asking
- `expected_output_format` (str | null): Expected format of the answer (e.g., "single sentence", "bullet points")
- `requires_audio_output` (bool): Whether this task requires/produces an audio file as output
- `notes` (str, optional): Additional notes or considerations

**Audio Output Detection:**
Set `requires_audio_output: true` when the user asks for:
- Audio processing/transformation (trim, cut, merge, mix, etc.)
- Audio enhancement (denoise, normalize, filter, etc.)
- Format conversion (convert to MP3, WAV, etc.)
- Audio extraction (extract from video, separate stems, etc.)
- Any task where the deliverable is a processed audio file

Set `requires_audio_output: false` when the user asks for:
- Information about the audio (transcription, caption, analysis)
- Questions about content ("what is being said?", "who is speaking?")
- Metadata extraction (duration, sample rate, etc.)

If the intent is unclear, express uncertainty in focus_points or notes.
