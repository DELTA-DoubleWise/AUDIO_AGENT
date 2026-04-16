Question: {question}

---

Task-Oriented Caption Skills Reference (for your internal reference only — the front-end model will NOT see this):
{caption_skills_reference}

---

Generate a question-oriented prompt for the front-end audio model.

Requirements:
- Output ONLY the prompt string (plain text, no JSON).
- The prompt must be fully self-contained. Do not use skill or modifier names as shorthand.
- Include: clarified question, decomposed tasks, and focus points with concrete watchouts.
- Embed actual focus points, thinking patterns, and cues directly into the prompt text.
- Be as detailed and concrete as needed, while staying focused on the question.