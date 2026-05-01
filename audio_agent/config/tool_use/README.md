# Tool-Use Skill/Report Routing

This directory keeps tool-use routing separate from prompt tuning.

## Files

- `task_skills.yaml`: fine-grained task skills. Each skill defines an abstract
  chain of slots such as `speech_asr`, `diarization`, `beat_onset`, or
  `omni_perception`. It does not hard-code one model as the only answer.
- `reports.yaml`: benchmark-derived model/tool rankings and dataset
  descriptions. Rankings fill abstract slots when there is relevant evidence.

## Resolution Rule

1. Classify the user question into a fine-grained `skill_id`.
2. For every abstract slot in that skill, look for report rows matching
   `(skill_id, dataset_id, slot)`.
3. Pick the best available catalog tool from the ranking.
4. If no ranked tool is available, use the skill default extracted from the
   reference workflows.
5. Preserve guardrails from the skill. For example, onset count cannot be used
   as an event count without event-class confirmation.

## Example

```python
from audio_agent.tool_use import resolve_tool_chain

available = {
    "transcribe_qwenasr",
    "transcribe_fireredasr",
    "segment_audio",
    "silencedetect",
    "omni_caption",
}

result = resolve_tool_chain(
    "speech.asr.codeswitch_zh_en_dialogue",
    dataset_id="cs_dialogue",
    available_tools=available,
)
```

The resolver will select `transcribe_qwenasr` from Table 3 for
CS-Dialogue because it is the best available report-ranked tool. If Qwen3-ASR is
not available, it falls back to the next available report candidate or the
reference-flow default.

## Reference Workflow Defaults

`ARC-Agent-NO1` contributes category/tool-chain defaults:

- counting: beat/onset, diarization, spectral, energy
- music: chord/melody/rhythm/instrument/tempo/harmonic tools
- speaker: diarization, vocal emotion, speech LLM
- emotion: vocal emotion, dialogue structure, speech events
- scene: scene context, speech LLM, stereo/spectral/energy
- timing: beat/onset, event sequence, tempo, segment analysis
- content: dialogue, speech events, scene context, correlation
- signal: quality/effects/stereo/anomaly tools

`xlance-audio-reasoning-agent` contributes the ensemble/arbitration pattern:

- Qwen-Omni and Step-Audio-R1 are treated as independent perception sources.
- SED, caption, audio features, and visual解析结果 are auxiliary arbitration
  evidence.
- Agreement cases can be lighter; disagreement cases need stronger verification.
