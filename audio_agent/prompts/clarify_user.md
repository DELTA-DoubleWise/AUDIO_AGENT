Original Question: {question}
Current Clarified Intent: {clarified_intent}
Current Expected Format: {expected_format}

Accumulated Evidence:
{evidence_text}

Based on the question and accumulated evidence, clarify the intent and expected output format.
Return a JSON object with keys:
  `clarified_intent` (str): What the question is actually asking
  `expected_output_format` (str | null): Expected format (e.g., 'bullet points', 'single sentence', 'detailed paragraph')
