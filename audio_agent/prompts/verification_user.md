Question: {question}

Proposed Answer to Review:
{proposed_answer}

Please listen to the audio and review the proposed answer for any obvious flaws or contradictions.

Return your assessment in this exact JSON format:
{{
    "passed": true or false,
    "critique": "explanation of flaws if failed, or null if passed",
    "confidence": 0.0 to 1.0
}}
