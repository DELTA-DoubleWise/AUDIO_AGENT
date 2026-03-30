# TOOL_INPUT Template for Tool Onboarding Agent

Replace `TOOL_INPUT` in the system prompt with a concrete YAML block following this template.

---

## Template

```yaml
tool_id: owner/model-name
tool_name: ToolName
task_type: asr|s2tt|sd|ser|speech_enhancement|vad|omni|utility|...
deployment_type: local|api

repo:
  url: https://github.com/owner/repo
  commit: null  # or specific commit hash

weights:
  source: huggingface|pip|release_or_pypi|local|api
  local_path: null
  required: true

environment_hint:
  preferred_backend: uv|conda|docker
  python_version: "3.11"
  requires_gpu: true|false
  system_packages: [ffmpeg, libsndfile1]

phase1_runtime_target:
  Validate the minimal callable path only:
  - confirm package is importable
  - load model with minimal config
  - run inference on fixture
  This phase does NOT require accuracy evaluation or production validation.

entrypoints:
  import_test: "import package"
  load_test: "model = package.load_model('tiny', 'cpu')"
  infer_test: "model.transcribe('tests/fixtures/shared/asr/en_16k.wav')"

fixture:
  audio: tests/fixtures/shared/asr/en_16k.wav
  task_specific: true|false
  fallback_allowed: true|false

io_contract:
  input_type: audio_path|text|json
  output_type: json|text
  primary_field: text|segments|labels
  required_fields: [field1, field2]
  nonempty_fields: [field1]
  json_serializable: true
```

---

## Example: WhisperX

```yaml
tool_id: m-bain/whisperX
tool_name: WhisperX
task_type: asr
deployment_type: local

repo:
  url: https://github.com/m-bain/whisperX
  commit: null

weights:
  source: huggingface
  local_path: null
  required: true

environment_hint:
  preferred_backend: uv
  python_version: "3.11"
  requires_gpu: true
  system_packages: [ffmpeg]

phase1_runtime_target:
  Validate the minimal callable path only:
  - confirm package is importable
  - load model with minimal config
  - run inference on fixture
  This phase does NOT require accuracy evaluation or production validation.

entrypoints:
  import_test: "import whisperx"
  load_test: "import whisperx; model = whisperx.load_model('tiny', 'cpu')"
  infer_test: "result = model.transcribe('tests/fixtures/shared/asr/en_16k.wav')"

fixture:
  audio: tests/fixtures/shared/asr/en_16k.wav
  task_specific: false
  fallback_allowed: true

io_contract:
  input_type: audio_path
  output_type: json
  primary_field: segments
  required_fields: [segments]
  nonempty_fields: [segments]
  json_serializable: true
```

---

## Example: API-Only Tool

```yaml
tool_id: qwen3-omni-flash
tool_name: Qwen3OmniAPI
task_type: omni
deployment_type: api

repo:
  url: https://github.com/QwenLM/Qwen3-Omni
  commit: null

weights:
  source: api
  local_path: null
  required: false

environment_hint:
  preferred_backend: api
  python_version: "3.11"
  requires_gpu: false
  system_packages: []

phase1_runtime_target:
  Validate the minimal callable path only:
  - confirm package is importable
  - load model with minimal config
  - run inference on fixture
  This phase does NOT require accuracy evaluation or production validation.

entrypoints:
  import_test: "import openai"
  load_test: "client = openai.OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'])"
  infer_test: "client.chat.completions.create(model='qwen3-omni-flash', messages=[...])"

fixture:
  audio: tests/fixtures/shared/asr/en_16k.wav
  task_specific: false
  fallback_allowed: true

io_contract:
  input_type: audio_path
  output_type: text
  primary_field: text
  required_fields: [text]
  nonempty_fields: [text]
  json_serializable: true
```
