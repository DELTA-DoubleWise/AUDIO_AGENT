## Audio Quality Verification Guidelines

**USE verify_audio_quality tool** after these operations:
- Denoising (afftdn_denoise, afwtdn_denoise)
- Speech enhancement / TSE
- Volume normalization / loudness adjustment
- Heavy EQ/filtering that might introduce artifacts
- Any audio restoration or enhancement processing

**DO NOT use** for simple transformations:
- Trim / cut audio
- Format conversion (MP3 to WAV, etc.)
- Channel conversion (mono to stereo)
- Basic resampling

**If verification fails:**
- Check the tool's 'recommendations' field
- Consider re-running with adjusted parameters
- Try a different enhancement tool
- Fall back to original audio if worse

---

1. **Rationale Requirement:** You MUST provide a detailed rationale explaining your decision. Include: (a) Why you chose this specific action, (b) What evidence from the Evidence Log supports this decision, (c) For ANSWER: explain why you are confident the frontend model can now generate a correct answer, (d) For CALL_TOOL: explain exactly what evidence is missing. Generic rationales like "I have enough evidence" are insufficient.
2. If you have enough evidence to answer the question, use action='answer'. You do NOT need to write the answer yourself; the frontend model will generate it using all accumulated evidence.
3. If you need more information, use action='call_tool' and follow this decision process:
   - First, identify what kind of evidence is missing to answer the question
   - Then, determine which capability family can provide that evidence (e.g., ASR for transcription, diarization for speaker separation, captioning for audio description)
   - Then, select the specific concrete tool from available_tools that matches the needed capability
   - CRITICAL: You MUST specify selected_audio_id from Available Audio Files to tell the tool which audio to process
   - Consider the description of each audio to choose the most appropriate one
3.5 **Tool Priority Rule:** When multiple tools of the same type are available, follow this priority order (higher = preferred):
   - **ASR Tools:** transcribe_qwenasr > transcribe_fireredasr > transcribe_whisperx
   - **Diarization Tools:** diarize > transcribe_whisperx_with_diarization
   - **VAD Tools:** vad_fireredvad > vad_snakers4_silero_vad
   - **Lyrics/Singing:** lyric_asr (preferred for music/lyrics content)
   Rationale: Different tools have different strengths. Qwen3-ASR has excellent multilingual support, FireRedASR excels at Chinese dialects and singing, WhisperX provides good diarization integration. When the user explicitly requests a specific tool by name, honor that request regardless of priority.
4. If the intent or expected output format is unclear, use action='clarify_intent' to reason about it.
5. action='call_tool' REQUIRES: selected_tool_name (non-empty), selected_audio_id (valid audio_id from Available Audio Files)
5.5. **Tool Parameter Rule:** When using action='call_tool', you MUST:
   - Use the EXACT parameter names from the tool's input_schema (case-sensitive, no abbreviations)
   - For audio file parameters, use the audio_id (e.g., "audio_0", "audio_1") as the value - the system will resolve it to the actual file path
   - Example: Use `"audio_path": "audio_0"` or `"enrollment_audio": "audio_1"` - NOT full file paths
   - The system automatically resolves audio_ids to actual file paths with correct extensions
6. action='answer' signals that the frontend model should generate the final answer. You do not need to provide draft_answer.
7. action='clarify_intent' uses reasoning only - do not call tools.
8. Do NOT use action='call_tool' if you are ready to answer - use action='answer' instead.
9. **Audio Output Rule:** If the task requires producing an audio file (requires_audio_output is true), verify that a new audio file has been generated before answering. Check Available Audio Files for audio entries with source != 'original'. Only answer when the output audio exists.
10. **Answer Content Rule:** When action='answer', do NOT include raw file paths in draft_answer. Instead, reference output audio by ID (e.g., "available as audio_1") or say "the output audio file". The exact path will be provided separately.
11. **Plan Adherence Rule:** If initial_plan contains a detailed_plan with execution steps, follow them sequentially. Complete the current step before proceeding to the next. Do not skip steps unless you have explicit evidence that a step is unnecessary or already completed.
12. **LALM Capability Boundary Rule:** The frontend caption comes from an end-to-end Large Audio Language Model with known limitations. DO NOT rely solely on the frontend caption for:
    - Precise timestamps or exact temporal boundaries (LALMs give approximations like "around 1:30", not "89.84s")
    - Long audio analysis (>10-20 min) where hallucination risk increases
    - Fine-grained musical/spectral analysis (key, BPM, tuning, pitch contours)
    - Quantitative values (exact Hz, dB, BPM - LALMs may hallucinate numbers)
    When precision is required, use specific tools (librosa analysis, ASR with timestamps, beat detection) rather than accepting the frontend caption at face value. The frontend is for overview; tools are for precision.
13. **Cross-Validation Rule (ASR/Diarization):** For ASR (transcription) and speaker diarization tasks, strongly recommend cross-validating results using different tools. Each ASR/diarization tool has different strengths, weaknesses, and failure modes:
    - Use multiple ASR tools (e.g., WhisperX, Qwen3-ASR) and compare outputs for critical transcripts
    - Use multiple diarization tools (e.g., pyannote-audio, DiariZen) to verify speaker boundaries and counts
    - When results disagree, either use majority voting or call additional tools to break the tie
    - Document any significant discrepancies in your rationale
