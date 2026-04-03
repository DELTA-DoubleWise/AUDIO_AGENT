Question: {question}

Expected Output Format: {expected_format}

Proposed Answer to Check:
{proposed_answer}

Please check if the proposed answer follows the expected output format requirements.

Return your assessment in this exact JSON format:
{{
    "passed": true or false,
    "critique": "explanation of format violations and how to fix them, or null if passed",
    "confidence": 0.0 to 1.0
}}
