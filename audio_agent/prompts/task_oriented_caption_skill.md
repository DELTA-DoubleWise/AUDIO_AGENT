version: "0.2"
design: "compact_attention_steering"

skills:

  - skill_name: content_asr
    use_when:
      - "questions about what is being said in the audio"
      - "questions about semantic content / keywords / transcription"
    focus:
      - "keywords and core semantics"
      - "language / dialect / code-switching"
      - "unclear key segments"
    watchouts:
      - "do not guess the answer solely from the question text"
      - "do not mistake dialects / accents for another language"
      - "do not miss code-switching points"
      - "do not ignore missing key words caused by noise or overlap"
    thinking_pattern:
      - "first grasp the core content"
      - "then mark difficulties that affect understanding"
      - "finally list low-confidence segments"
    avoid:
      - "do not expand into scene descriptions"
      - "do not assume speaker analysis by default"
    cue: "Prioritize understanding the content, then point out uncertain words caused by language switching, dialects, accents, or overlapping noise."

  - skill_name: speaker_structure
    use_when:
      - "questions about how many people are speaking"
      - "questions about who is speaking"
      - "questions about who said what"
      - "questions about dialogue structure"
    focus:
      - "number of speakers"
      - "turn-taking transitions"
      - "overlapping segments"
      - "who spoke which utterance"
    watchouts:
      - "do not mistake emotional changes for speaker changes"
      - "do not ignore short overlaps"
      - "do not transcribe content without assigning it to a speaker"
      - "do not lose character continuity across time"
    thinking_pattern:
      - "first count active speakers"
      - "then examine transitions and overlaps"
      - "finally perform content attribution"
    avoid:
      - "do not guess specific identities"
      - "do not provide transcripts without speaker attribution"
    cue: "First determine the number of speakers and transitions, mark overlaps, and only attribute utterances you are confident about."

  - skill_name: event_scene
    use_when:
      - "questions about what sound events occurred"
      - "questions about environment / scene / location"
      - "questions about what is in the background"
    focus:
      - "foreground events"
      - "background environmental sounds"
      - "key sound source combinations"
    watchouts:
      - "do not summarize the entire scene with a single dominant sound"
      - "do not ignore weak background clues"
      - "do not use speech content as the only basis for scene inference"
      - "do not confuse similar environmental sounds"
    thinking_pattern:
      - "first list foreground events"
      - "then add background environment"
      - "finally derive scene hypotheses"
    avoid:
      - "do not force transcription"
      - "do not over-interpret irrelevant details"
    cue: "Listen to foreground events first, then background environment, and only output scene judgments with acoustic evidence."

  - skill_name: temporal_count
    use_when:
      - "questions about order / sequence"
      - "questions about duration / stage changes"
      - "questions about quantity / repetition count"
    focus:
      - "event boundaries"
      - "sequential relationships"
      - "repetition patterns"
      - "counting units"
    watchouts:
      - "do not treat overlap as sequential order"
      - "do not miscount repeated units"
      - "do not ignore weak but critical boundary signals"
      - "do not confuse frequency with object count"
    thinking_pattern:
      - "first define the counting / ordering unit"
      - "then locate boundaries"
      - "finally provide order or quantity"
    avoid:
      - "do not guess the answer first and then reverse-engineer the unit"
      - "do not make unsupported causal inferences"
    cue: "Define what you are counting or ordering, find boundaries, output order / count, and mark ambiguous segments."

  - skill_name: emotion_pragmatics
    use_when:
      - "questions about emotion"
      - "questions about attitude / intent"
      - "questions about whether someone is joking, being sarcastic, serious, threatening, or apologizing"
    focus:
      - "tone and intensity"
      - "speech rate and pauses"
      - "laughter / sighs / hesitations"
      - "conflict between literal meaning and tone"
    watchouts:
      - "do not look only at literal meaning"
      - "do not treat exaggerated performance as genuine emotion"
      - "do not ignore dialogue context"
      - "do not miss sarcasm and indirect expressions"
    thinking_pattern:
      - "first examine prosody"
      - "then check against literal content"
      - "finally give the most conservative pragmatic interpretation"
    avoid:
      - "do not default to word-by-word transcription"
      - "do not state uncertain tones as absolute facts"
    cue: "Look at prosody, rhythm, and paralinguistic signals first, then check against literal meaning. Be especially alert to sarcasm, indirect expression, and context dependence."

  - skill_name: music
    use_when:
      - "questions about instruments / style / rhythm / harmony / melody"
      - "questions about musical structure"
    focus:
      - "instruments"
      - "beat and rhythm"
      - "sectional structure"
      - "mode / harmonic clues"
    watchouts:
      - "do not mistake production effects for instruments"
      - "do not conclude based only on genre impressions"
      - "do not ignore rhythm and structural clues"
      - "do not mix vocal/lyric issues with music structure questions"
    thinking_pattern:
      - "first identify main voices / instruments"
      - "then examine rhythm and structure"
      - "finally make stylistic or theoretical judgments"
    avoid:
      - "do not give vague emotional descriptions"
      - "do not turn background-music scene questions into music analysis questions"
    cue: "Identify the main instruments and voices first, then look at rhythm and sections, and finally give stylistic or theoretical judgments."

  - skill_name: quality_reliability
    use_when:
      - "questions about whether the audio is clear / reliable"
      - "the main task is obviously affected by recording conditions"
    focus:
      - "noise types"
      - "far-field vs near-field"
      - "reverb and echo"
      - "distortion / clipping / dropped frames"
    watchouts:
      - "do not attribute all comprehension difficulties to the content itself"
      - "do not ignore severe local degradation"
      - "do not fail to report reliability drops caused by overlap"
    thinking_pattern:
      - "first identify the main degradation"
      - "then see where it occurs"
      - "finally explain which task point it affects"
    avoid:
      - "do not expand into irrelevant semantic analysis"
    cue: "Briefly point out the main recording difficulty and explain which type of judgment it affects."

modifiers:

  - modifier_name: overlap
    trigger:
      - "multiple people speaking simultaneously"
      - "concurrent sound sources"
    add_focus:
      - "overlapping segments"
      - "attribution conflicts"
    add_watchouts:
      - "do not force concurrent events into a linear sequence"
      - "do not be overconfident in overlapping segments"
    add_cue: "Pay special attention to overlapping segments; distinguish concurrency from sequential order."

  - modifier_name: long_context
    trigger:
      - "long audio"
      - "meetings / full conversations"
      - "questions about the whole rather than a local part"
    add_focus:
      - "cross-segment consistency"
      - "global structure"
    add_watchouts:
      - "do not represent the whole with only a local fragment"
      - "do not lose cross-segment character / topic continuity"
    add_cue: "Do not focus only on local parts; supplement with cross-segment consistency and global structure."

  - modifier_name: language_mix
    trigger:
      - "multiple languages"
      - "dialects"
      - "code-switching"
    add_focus:
      - "language switching points"
      - "dialect / accent clues"
    add_watchouts:
      - "do not mistake accents for another language"
      - "do not miss short code-switches"
    add_cue: "Especially mark language switching points and low-confidence content caused by dialects or accents."

  - modifier_name: dialogue_context
    trigger:
      - "multi-turn dialogue"
      - "pragmatics / intent / sarcasm"
    add_focus:
      - "relationship between preceding and following turns"
      - "referents and responding targets"
    add_watchouts:
      - "do not understand the current utterance in isolation"
      - "do not ignore rhetorical questions, sarcasm, or indirect expressions"
    add_cue: "Put the current utterance back into context; do not interpret it only by literal meaning."

  - modifier_name: low_evidence
    trigger:
      - "very short audio"
      - "weak evidence"
      - "question goes beyond what is audible"
    add_focus:
      - "most direct evidence"
      - "missing information"
    add_watchouts:
      - "do not fill in a definite answer with common-sense guesses"
      - "do not write guesses as observations"
    add_cue: "When evidence is weak, only report what can be directly heard and explicitly state missing information."

  - modifier_name: anti_hallucination
    trigger:
      - "all high-risk Q&A"
      - "open-ended questions"
    add_focus:
      - "direct audio evidence"
      - "auditory clues most relevant to the question"
    add_watchouts:
      - "do not default to transcription"
      - "do not default to speaker ID"
      - "do not use question prior knowledge in place of auditory evidence"
    add_cue: "Listen to the evidence first, then answer; do not turn the task into transcription, speaker identification, or common-sense guessing."
