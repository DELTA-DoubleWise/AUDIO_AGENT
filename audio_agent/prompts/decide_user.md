Question: {question}
Frontend Caption: {frontend_caption}
Initial Plan: {initial_plan}
Evidence Log: {evidence_log}
Tool Call History: {tool_call_history}
Available Tools: {available_tools}
Step Count: {step_count}
Max Steps: {max_steps}

Decision Rules:
{decision_rules}

Required Output Format:
{{
    "action": "answer | call_tool | clarify_intent | fail",
    "rationale": "str - explain your decision",
    "selected_tool_name": "str | null - REQUIRED for call_tool, must be a valid tool name",
    "selected_tool_args": "dict - arguments for the tool when using call_tool",
    "draft_answer": "str | null - REQUIRED for answer, your final response to the question",
    "confidence": "float - 0.0 to 1.0"
}}
