Question: {question}

Expected Output Format: {expected_format}

Is Audio Output Task: {is_audio_output_task}

Proposed Answer to Check:
{proposed_answer}

Please check if the proposed answer follows the expected output format requirements.

**Note on Audio Output Tasks:**
If this is an audio output task (audio processing, conversion, enhancement, etc.), the answer should describe what was done and reference the output audio. A text description like "the audio was trimmed to 5 seconds" or a reference like "output audio: audio_1" is the CORRECT format. Do not expect the actual audio file content in the answer field.

Return your assessment in this exact JSON format:
{{
    "passed": true or false,
    "critique": "explanation of format violations and how to fix them, or null if passed",
    "confidence": 0.0 to 1.0
}}
