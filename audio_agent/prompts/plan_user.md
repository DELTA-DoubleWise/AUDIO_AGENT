Question: {question}

Produce an InitialPlan JSON object with keys:
- `approach` (str): High-level approach to answer the question
- `focus_points` (list[str]): Key points to investigate in the audio
- `possible_tool_types` (list[str]): Tool types that might help (e.g., "asr", "diarization", "captioning")
- `clarified_intent` (str | null): What the question is actually asking
- `expected_output_format` (str | null): Expected format of the answer (e.g., "single sentence", "bullet points")
- `requires_audio_output` (bool): Whether this task requires/produces an audio file as output
- `notes` (str, optional): Additional notes or considerations
- `detailed_plan` (list[ExecutionStep], optional): Todo list for complex questions (see below)

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

**Detailed Plan (for complex questions only):**
For simple questions, use: `"detailed_plan": []`
For complex questions requiring multiple steps, provide an array of ExecutionStep objects:

```json
{{
  "step_number": 1,
  "description": "Transcribe audio to get speaker content and timing",
  "tool_type": "asr",
  "expected_output": "Transcript with speaker turn timestamps"
}}
```

**Example Simple Question:**
Question: "What is the sample rate of this audio?"
Output: `{{ "detailed_plan": [] }}`

**Example Complex Question:**
Question: "What emotions does each speaker express?"
Output:
```json
{{
  "detailed_plan": [
    {{
      "step_number": 1,
      "description": "Transcribe audio to identify speaker turns and content",
      "tool_type": "asr",
      "expected_output": "Transcript with speaker timestamps"
    }},
    {{
      "step_number": 2,
      "description": "Separate speakers to isolate individual audio streams",
      "tool_type": "diarization",
      "expected_output": "Speaker segments with labels"
    }},
    {{
      "step_number": 3,
      "description": "Analyze emotional tone of each speaker's segments",
      "tool_type": "emotion_analysis",
      "expected_output": "Emotion labels per speaker per segment"
    }},
    {{
      "step_number": 4,
      "description": "Synthesize emotion findings into final answer",
      "tool_type": null,
      "expected_output": "Summary of emotions per speaker"
    }}
  ]
}}
```

If the intent is unclear, express uncertainty in focus_points or notes.
