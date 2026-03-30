# Model Spec Template Documentation

**Version**: 1.0  
**Scope**: Builder Agent, Harness Controller  
**Purpose**: Explain how to write and validate `model.spec.yaml`

---

## What is model.spec.yaml

`model.spec.yaml` is an **optional but strongly recommended** specification file for new tools in the Audio Agent Framework. It provides structured metadata that the agent harness uses to:

- Select the right backend
- Plan validation tests
- Verify output contracts
- Generate wrappers

While the Audio Agent Framework's catalog loader primarily reads `config.yaml`, `model.spec.yaml` serves as the **agent onboarding artifact** that makes the harness workflow possible.

---

## File Location

```
audio_agent/tools/catalog/{tool_name}/
├── model.spec.yaml    # This file
├── config.yaml        # Required: MCP config
├── server.py          # Required: MCP server
└── ...
```

---

## Template

See [`../templates/model.spec.yaml`](../templates/model.spec.yaml) for the raw template.

---

## Field Reference

### tool_id

**Required** | string

Unique identifier, typically `owner/model-name` or HuggingFace repo ID.

Example: `Qwen/Qwen3-ASR-1.7B`

### tool_name

**Required** | string

Human-readable name for the tool.

Example: `ASRQwen3`

### task_type

**Required** | string

The task this tool performs. Must be one of:

- `asr` — Automatic Speech Recognition
- `s2tt` — Speech-to-Text Translation
- `sd` — Speaker Diarization
- `ser` — Speech Emotion Recognition
- `vad` — Voice Activity Detection
- `speech_enhancement` — Noise suppression / enhancement
- `omni` — Omni-modal understanding
- `api` — API-only tool
- `utility` — General utility (e.g., ffmpeg)

### deployment_type

**Required** | string

Either `local` (runs in `.venv`) or `api` (calls remote service).

### repo

**Required** | object

```yaml
repo:
  url: "https://github.com/owner/repo"
  commit: null  # or specific commit hash
```

### weights

**Required** | object

```yaml
weights:
  source: "huggingface"  # huggingface | modelscope | pip | local
  local_path: null       # e.g., "/lihaoyu/workspace/AUDIO_AGENT/models/Qwen3-ASR-1.7B"
  required: true
```

If using HuggingFace models, register the model in `audio_agent/utils/model_downloader.py` and download via:

```bash
audio-agent-download-models --models <model-key>
```

### environment

**Required** | object

```yaml
environment:
  preferred_backend: "uv"      # uv | conda | docker | api
  python_version: "3.11"
  requires_gpu: true
  system_packages: [ffmpeg, libsndfile1]
```

**Notes**:
- `preferred_backend` is a hint, not a command. The actual backend is selected via [Backend Selection Policy](../policies/backend_selection.md).
- For conda-based tools (e.g., DiariZen), use `python_version: "3.10"`.
- For API tools, use `preferred_backend: "api"`.

### entrypoints

**Required** | object

Three test snippets that validate the minimal callable path:

```yaml
entrypoints:
  import_test: "import torch"
  load_test: "from model import ModelWrapper; m = ModelWrapper(device='cpu')"
  infer_test: "m.predict('tests/fixtures/shared/asr/en_16k.wav')"
```

These are used by the harness to generate actual validation scripts.

### fixture

**Required** | object

```yaml
fixture:
  audio: "tests/fixtures/shared/asr/en_16k.wav"
  task_specific: false
  fallback_allowed: true
```

See [Fixture Policy](../contracts/fixture_policy.md) for fixture selection rules.

### io_contract

**Required** | object

Defines the expected input/output contract for contract validation:

```yaml
io_contract:
  input_type: "audio_path"      # audio_path | text | json
  output_type: "text"           # text | json
  primary_field: "text"         # e.g., "text" for ASR, "labels" for SER
  required_fields: ["text"]     # Must exist in output
  nonempty_fields: ["text"]     # Must be non-empty
  json_serializable: true
```

### acceptance

**Required** | object

```yaml
acceptance:
  must_import: true
  must_load: true
  must_infer: true
  must_return_nonempty: true
```

---

## Validation

Before BUILD_ENV, the harness runs `VALIDATE_SPEC` which checks:

1. **Spec Completeness** — All required fields present
2. **Evidence Sufficiency** — Critical fields have supporting evidence
3. **Conflict Resolution** — Any evidence conflicts are resolved
4. **Build Plan Executability** — Steps can actually be executed
5. **Fixture Availability** — Test audio exists or fallback is allowed
6. **IO Contract Sufficiency** — Contract is detailed enough for testing
7. **Preflight Compatibility** — Backend choice matches host capabilities

See [Spec Validation Contract](../contracts/spec_validation.md) for full details.

---

## Example: ASR Tool

```yaml
tool_id: Qwen/Qwen3-ASR-1.7B
tool_name: ASRQwen3
task_type: asr
deployment_type: local

repo:
  url: https://github.com/QwenLM/Qwen3-ASR
  commit: null

weights:
  source: huggingface
  local_path: /lihaoyu/workspace/AUDIO_AGENT/models/Qwen3-ASR-1.7B
  required: true

environment:
  preferred_backend: uv
  python_version: "3.11"
  requires_gpu: true
  system_packages: [ffmpeg, libsndfile1]

entrypoints:
  import_test: "from qwen_asr import Qwen3ASRModel"
  load_test: "model = Qwen3ASRModel.from_pretrained('Qwen/Qwen3-ASR-1.7B', device_map='cpu')"
  infer_test: "model.transcribe('tests/fixtures/shared/asr/en_16k.wav')"

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

acceptance:
  must_import: true
  must_load: true
  must_infer: true
  must_return_nonempty: true
```

---

## Example: API Tool

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

environment:
  preferred_backend: api
  python_version: "3.11"
  requires_gpu: false
  system_packages: []

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

acceptance:
  must_import: true
  must_load: true
  must_infer: true
  must_return_nonempty: true
```
