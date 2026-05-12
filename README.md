# Audio Agent Framework

A LangGraph-based framework for audio understanding with iterative tool use.

## Overview

This framework provides a clean architecture for building audio understanding agents that:
- Process audio with a frontend LALM (Large Audio Language Model)
- Use an LLM planner to reason about evidence and decide next actions
- Invoke tools iteratively to gather more evidence
- Accumulate and fuse evidence from multiple sources
- Produce final answers with supporting evidence

## Architecture

```
START
  -> initial_prompt_node (planner generates question-oriented prompt)
  -> frontend_evidence_node (LALM processes audio, generates initial caption)
  -> initial_plan_node (audio-aware planning using question + caption)
  -> planner_decision_node (LLM decides action; on last step forces ANSWER)
  -> [conditional routing based on decision]
     - ANSWER -> evidence_summarization_node (neutral summary of all evidence)
       -> final_answer_node (frontend model generates answer from audio + context)
       -> format_check_node (mandatory validation)
         * Format OK -> answer_node -> END
         * Format Failed -> planner_decision_node (loop with critique as evidence)
     - CALL_TOOL -> tool_executor_node (auto-injects audio_path)
       -> evidence_fusion_node -> planner_decision_node (loop)
     - CALL_FRONTEND -> frontend_followup_node (frontend re-perceives selected audio with custom prompt)
       -> evidence_fusion_node -> planner_decision_node (loop)
     - FAIL -> failure_node -> END
```

**Key Behaviors:**
- **Initial Prompt Generation**: A dedicated `initial_prompt_node` uses the planner to craft a customized `question_oriented_prompt` from the user question, helping the frontend produce a richer, structured caption.
- **Initial Planning**: Planner generates a high-level approach using both the question and the frontend's structured caption, enabling audio-aware planning.
- **Frontend Follow-Up**: The planner can explicitly choose `CALL_FRONTEND` to have the frontend model re-perceive selected derived audio artifacts (isolated speakers, trimmed segments, denoised clips, etc.) with a custom prompt. This bridges tool-generated audio improvements with the strongest audio-capable model.
- **Evidence Summarization**: Before final answer, a text-LLM compresses all evidence, planner trace, and tool history into a single neutral narrative. This prevents the frontend model from being overwhelmed by verbose raw tool outputs.
- **Frontend Final Answer**: The frontend (audio-capable) model generates the final answer directly from the original audio(s) and summarized context, rather than the text planner producing the answer.
- **Format Checking**: Mandatory format validation occurs before finalizing. If the format is wrong, a critique is added as evidence and planning continues.
- **Planner Tool Scope**: `AgentConfig.planner_tool_scope` defaults to `core`, exposing a compact benchmark-oriented tool inventory to the planner. Use `all` to expose every registered tool for broader audio editing/effects workflows. `AgentConfig.planner_tool_inventory_path` points to the authoritative planner-facing inventory for category definitions, descriptions, schemas, and tags.
- **Tool Contracts**: Low-level signal/metadata tools cannot override the frontend's semantic judgments.

## Project Structure

```
audio_agent/
├── __init__.py
├── main.py                 # Main entry point and AudioAgent class
├── core/                   # Core types and utilities
│   ├── state.py           # AgentState definition
│   ├── schemas.py         # Pydantic schemas
│   ├── errors.py          # Custom exceptions
│   ├── constants.py       # Enums and constants
│   └── logging.py         # Logging utilities
├── frontend/              # Audio frontend implementations
│   ├── base.py            # BaseFrontend ABC
│   ├── model_frontend.py  # BaseModelFrontend template
│   ├── dummy_frontend.py  # Dummy implementation
│   ├── qwen2_audio_frontend.py      # Qwen2-Audio adapter (local)
│   ├── qwen3_omni_frontend.py       # Qwen3-Omni adapter (local)
│   ├── openai_compatible_frontend.py # OpenAI-compatible API frontend
│   └── mimo_frontend.py             # Xiaomi MiMo API frontend
├── planner/               # Planner implementations
│   ├── base.py            # BasePlanner ABC
│   ├── model_planner.py   # BaseModelPlanner template
│   ├── dummy_planner.py   # Dummy implementation
│   ├── qwen25_planner.py  # Qwen2.5 planner adapter
│   ├── openai_compatible_planner.py # OpenAI-compatible API planner
│   └── mimo_planner.py    # Xiaomi MiMo API planner
├── tools/                 # Tool system
│   ├── base.py            # BaseTool ABC
│   ├── registry.py        # ToolRegistry (internal + MCP tools)
│   ├── executor.py        # ToolExecutor
│   ├── visibility.py      # Planner-visible tool scope filtering
│   ├── dummy_tools.py     # Dummy tools
│   ├── mcp/               # MCP (Model Context Protocol) infrastructure
│   │   ├── client.py      # MCP client
│   │   ├── server_manager.py  # MCP server lifecycle
│   │   ├── tool_adapter.py    # MCP to BaseTool adapter
│   │   └── schemas.py     # MCP data models
│   └── catalog/           # MCP tool catalog
│       ├── loader.py      # Auto-discovery and registration
│       ├── _template/     # Template for new tools
│       ├── asr_qwen3/     # Qwen3-ASR-1.7B speech recognition
│       ├── diarizen/      # Speaker diarization
│       ├── lv_chordia/    # Large-vocabulary chord recognition (ISMIR 2019)
│       ├── omni_captioner/ # Qwen3-Omni captioner
│       └── tempo_cnn/      # Tempo-CNN musical tempo estimation
├── fusion/                # Evidence fusion
│   ├── base.py           # BaseEvidenceFuser ABC
│   └── default_fuser.py  # Default implementation
├── graph/                 # LangGraph workflow
│   ├── builder.py        # Graph construction
│   ├── nodes.py          # Node functions
│   └── routing.py        # Routing logic
├── log/                   # Run logging
│   ├── __init__.py       # Exports: RunLogger, log_run
│   ├── logger.py         # RunLogger class for markdown logs
│   └── formatter.py      # Markdown formatting utilities
├── prompts/               # Markdown prompt files
│   ├── frontend_system.md           # Frontend system prompt
│   ├── frontend_user.md             # Frontend user instruction
│   ├── frontend_followup_system.md  # Targeted frontend follow-up system prompt
│   ├── frontend_followup_user.md    # Targeted frontend follow-up user instruction
│   ├── frontend_final_answer_system.md  # Frontend final answer system prompt
│   ├── frontend_final_answer_user.md    # Frontend final answer user instruction
│   ├── initial_prompt_system.md     # Planner prompt for frontend prompt generation
│   ├── initial_prompt_user.md       # Planner instruction for frontend prompt generation
│   ├── plan_system.md               # Planner planning system prompt
│   ├── plan_user.md                 # Planner planning user instruction
│   ├── decide_system.md             # Planner decision system prompt
│   ├── decide_user.md               # Planner decision user instruction
│   ├── decide_rules.md              # Planner decision rules
│   ├── format_check_system.md       # Format check system prompt
│   ├── format_check_user.md         # Format check user instruction
│   ├── evidence_summary_system.md   # Evidence summarization system prompt
│   ├── evidence_summary_user.md     # Evidence summarization user instruction
│   ├── task_oriented_caption_skill.md  # Caption skill reference
│   └── task_skills.yaml             # Task skill reference for initial planning
├── config/                # Configuration
│   ├── settings.py       # AgentConfig
│   └── planner_tool_inventory.yaml  # Planner-facing tool descriptions and boundaries
├── utils/                 # Utilities
│   ├── validation.py     # Validation helpers
│   ├── model_io.py       # Model I/O helpers
│   ├── prompt_io.py      # Prompt loading utilities
│   └── model_downloader.py  # Model download utility
├── examples/              # Example scripts
│   ├── demo_run.py                # Basic demo (local models)
│   ├── demo_run_auto_tools.py     # Demo with auto MCP tool discovery
│   ├── demo_run_real_asr.py       # Demo with real ASR tool
│   ├── demo_run_api_planner.py    # Demo with API planner + local frontend
│   ├── demo_run_api_full.py       # Demo with API frontend + API planner (no GPU)
│   └── demo_run_mimo.py           # Demo with MiMo API frontend + planner (no GPU)
└── tests/                 # Tests
    ├── test_state.py
    ├── test_registry.py
    ├── test_graph_smoke.py
    └── ...
```

## Installation

For a verified, copy-paste-ready setup sequence, see
[ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md). The summary below mirrors it.

### Prerequisites

- **Python 3.11** (the main framework env). Python 3.10 is only required by the
  diarizen tool, which runs in its own isolated conda env.
- **uv** — used to provision every tool except diarizen. Install via:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  (If your machine already has a system `uv` on `$PATH`, that is enough.)
- **conda** — only required if you plan to set up the `diarizen` tool. Any
  miniconda/anaconda install on `$PATH` works; export `CONDA_SH=/path/to/conda/etc/profile.d/conda.sh`
  to point at a non-standard install.
- **A HuggingFace token** is needed if you want the `whisperx` diarization path
  (uses gated pyannote models). Run `huggingface-cli login` after accepting the
  `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` user agreements.
- **API key for the API-based demo**: `DASHSCOPE_API_KEY` (Alibaba DashScope)
  for `demo_run_api_full.py`. `OPENAI_API_KEY` for OpenAI-style backends.

### Bootstrap

```bash
git clone <repo-url> AUDIO_AGENT && cd AUDIO_AGENT

# 1. Main framework env (uv-managed, in repo root)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e '.[api,dev,download]'

# 2. Models directory (defaults to <repo>/models; override if you want elsewhere)
export AUDIO_AGENT_MODELS_DIR="$PWD/models"

# 3. Build every MCP tool's isolated environment
./setup_all_tools.sh

# 4. Download every model the catalog needs
audio-agent-download-models --all

# 5. Verify every tool builds and imports
./verify_all_tools.sh
```

`setup_all_tools.sh` auto-discovers every tool with a `setup.sh` and runs them.
Pass tool names to set up just a subset (e.g. `./setup_all_tools.sh ffmpeg librosa`).

## Running the Demo

The API-based demo needs no local GPU for the frontend/planner — only the MCP
tools that themselves call local models need a GPU.

```bash
# API-based, fully cloud-hosted frontend + planner (uses DashScope by default)
export DASHSCOPE_API_KEY="sk-xxx"
export AUDIO_AGENT_MODELS_DIR="$PWD/models"
python -m audio_agent.examples.demo_run_api_full \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?"
```

For local-model frontends (Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni), see
[DEMO_ENVIRONMENT.md](./DEMO_ENVIRONMENT.md).

### Verifying Tool Environments

To verify all MCP tools are properly configured:

```bash
# Test all tools (assumes ./setup_all_tools.sh already ran)
./verify_all_tools.sh

# Or combined: setup then verify
./verify_all_tools.sh --setup
```

See also `demo_run_real_asr.py` for a demo with specific ASR tool configuration.

### API-Based Demos (No Local GPU Required)

If you don't have a local GPU or prefer to use API-based models:

```bash
# Setup MCP tools first (same as above)
./verify_all_tools.sh --setup

# Demo with API planner + local frontend (single audio)
export DASHSCOPE_API_KEY="sk-xxx"
python -m audio_agent.examples.demo_run_api_planner \
  --audio /path/to/audio.wav \
  --question "What is being said?"

# Demo with API frontend + API planner (fully API-based, no local models)
python -m audio_agent.examples.demo_run_api_full \
  --audio /path/to/audio.wav \
  --question "What is being said?" \
  --frontend-model "qwen3-omni-flash" \
  --planner-model "qwen3.5-plus"

# Multi-audio example with API (speaker verification)
python -m audio_agent.examples.demo_run_api_full \
  --audio /path/to/first_audio.wav --audio /path/to/second_audio.wav \
  --question "Is the speaker in the second audio the same as the first?" \
  --frontend-model "qwen3-omni-flash" \
  --planner-model "qwen3.5-plus"
```

### MiMo API Demo (No Local GPU Required)

If you have access to Xiaomi MiMo's API:

```bash
# Set MiMo API key
export MIMO_API_KEY="sk-xxx"

# Demo with MiMo frontend + MiMo planner (fully API-based)
python -m audio_agent.examples.demo_run_mimo \
  --audio /path/to/audio.wav \
  --question "What is being said?"
```

The `demo_run_mimo.py` script uses:
- `MimoFrontend` with `mimo-v2.5` for audio understanding
- `MimoPlanner` with `mimo-v2.5-pro` for decision making
- MiMo's OpenAI-compatible endpoint

The API-based demos are ideal for:
- Users without local GPU resources
- Quick prototyping and testing
- Deployments where model inference is handled externally

## Pre-downloading Models

Model weights live under the directory pointed to by `AUDIO_AGENT_MODELS_DIR`. If
that env var is unset, the framework defaults to `<repo>/models/` (which is in
`.gitignore`). All tool `config.yaml` files reference the directory via
`${AUDIO_AGENT_MODELS_DIR}`, so a single export covers every tool.

```bash
# Optional: store weights somewhere other than <repo>/models.
export AUDIO_AGENT_MODELS_DIR=/path/to/your/models
```

**Download all models (one-time setup):**

```bash
# Install with download support
pip install -e ".[download]"

# Download all registered models
audio-agent-download-models --all
```

**Download specific models:**

```bash
audio-agent-download-models --models qwen2-audio qwen2.5
```

**List available models and their status:**

```bash
audio-agent-download-models --list
```

**Available models:**
- `qwen2-audio` - Qwen/Qwen2-Audio-7B-Instruct (frontend, ~15GB)
- `qwen2.5-omni` - Qwen/Qwen2.5-Omni-7B (single-GPU unified frontend, ~16GB)
- `qwen3-omni` - Qwen/Qwen3-Omni-30B-A3B-Instruct (frontend, ~60GB)
- `qwen2.5` - Qwen/Qwen2.5-7B-Instruct (planner, ~15GB)
- `qwen3-asr` - Qwen/Qwen3-ASR-1.7B (ASR tool, ~4GB)
- `qwen3-aligner` - Qwen/Qwen3-ForcedAligner-0.6B (aligner tool, ~1.5GB)
- `diarizen` - BUT-FIT/diarizen-wavlm-large-s80-md (diarization, ~1GB)
- `sortformer-diar` - nvidia/diar_streaming_sortformer_4spk-v2 (diarization, ~450MB)
- `omni-captioner` - Qwen/Qwen3-Omni-30B-A3B-Captioner (captioner, ~60GB)
- `fireredasr` - FireRedTeam/FireRedASR-AED-L (Mandarin/English ASR, ~3GB)
- `fireredvad` - FireRedTeam/FireRedVAD (VAD + AED, ~200MB)
- `wespeaker` - Wespeaker/wespeaker-voxceleb-resnet34-LM (speaker embedding, ~30MB)
- `pyannote-diarization`, `pyannote-segmentation` - used by whisperx diarization
  pipeline. **Requires a HuggingFace token with accepted user agreements** for
  the pyannote models (run `huggingface-cli login`).

**Using HuggingFace Hub paths (fallback):**

If you prefer to use HuggingFace Hub paths directly (models will be downloaded to cache):

```bash
# Single audio
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio.wav \
  --question "What is being said?" \
  --frontend-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --planner-model-path Qwen/Qwen2.5-7B-Instruct

# Multiple audios for comparison
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio1.wav --audio /path/to/audio2.wav \
  --question "Compare the speakers in these two audio files" \
  --frontend-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --planner-model-path Qwen/Qwen2.5-7B-Instruct
```

## Running Tests

```bash
pytest audio_agent/tests/ -v
```

## Multi-Audio Support

The framework supports processing multiple audio files in a single run, enabling tasks like:
- **Speaker verification**: Compare if two audio files contain the same speaker
- **Audio comparison**: Compare content, quality, or characteristics across multiple files
- **Multi-source analysis**: Analyze audio from different sources together

When multiple audios are provided:
- Each audio is assigned an ID (`audio_0`, `audio_1`, `audio_2`, etc.)
- The frontend processes all audios and provides a caption for each
- Tools can reference specific audios by their ID
- The planner can reason about relationships between audios

### Example: Speaker Verification

```python
from audio_agent.main import create_api_full_agent

agent = create_api_full_agent(
    frontend_model="qwen3-omni-flash",
    planner_model="qwen3.5-plus",
)

# Pass multiple audio files as a list
result = agent.run(
    question="Is the speaker in the second audio any of the speakers in the first audio?",
    audio_paths=["/path/to/first_audio.wav", "/path/to/second_audio.wav"]
)

if agent.is_successful(result):
    print(result["final_answer"].answer)
```

## Design Principles

### Fail-Fast
- All inputs validated before processing
- Missing required fields raise explicit exceptions
- Invalid tool names, malformed outputs caught immediately
- No silent fallbacks or None returns

### Explicit Contracts
- Every component has well-defined input/output schemas
- Pydantic models with validation
- Type hints throughout

### Extensibility
- Abstract base classes for all major components
- Easy to replace dummy implementations with real ones
- Clean dependency injection via factory functions

### Separation of Concerns
- Frontend: initial audio understanding
- Planner: decision making
- Tools: external capabilities
- Fusion: evidence accumulation
- Graph: workflow orchestration

## Extending the Framework

### Adding a New Tool

```python
from audio_agent.tools.base import BaseTool
from audio_agent.core.schemas import ToolSpec, ToolCallRequest, ToolResult

class MyAudioTool(BaseTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="my_audio_tool",
            description="Does something with audio",
            input_schema={"type": "object", "properties": {...}},
            output_schema={"type": "object", "properties": {...}},
        )
    
    def invoke(self, request: ToolCallRequest) -> ToolResult:
        # Your implementation here
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            output={"result": "..."},
        )

# Register the tool
registry.register(MyAudioTool())
```

### Adding a Real Frontend

**Local Model Frontend:**

```python
from audio_agent.frontend.base import BaseFrontend
from audio_agent.core.schemas import FrontendOutput

class RealLALMFrontend(BaseFrontend):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
    
    @property
    def name(self) -> str:
        return "real_lalm"
    
    def run(self, question: str, audio_paths: list[str]) -> FrontendOutput:
        self.validate_inputs(question, audio_paths)
        # Your implementation here
        return FrontendOutput(
            question_guided_caption="...",
        )
```

**API-Based Frontend:**

Use the built-in `OpenAICompatibleFrontend` for API-based frontends:

```python
from audio_agent.frontend.openai_compatible_frontend import OpenAICompatibleFrontend

frontend = OpenAICompatibleFrontend(
    model="qwen3-omni-flash",  # Or any API model that supports audio
    api_key="sk-xxx",  # Or set DASHSCOPE_API_KEY env var
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

Or use the dedicated `MimoFrontend` for Xiaomi MiMo:

```python
from audio_agent.frontend.mimo_frontend import MimoFrontend

frontend = MimoFrontend(
    model="mimo-v2.5",
    api_key="sk-xxx",  # Or set MIMO_API_KEY env var
)
```

The API frontend sends audio as base64-encoded data and receives text captions.

### Adding a Real Planner

```python
from audio_agent.planner.base import BasePlanner
from audio_agent.core.schemas import PlannerDecision

class OpenAIPlanner(BasePlanner):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    @property
    def name(self) -> str:
        return "openai_planner"
    
    def plan(self, question: str) -> InitialPlan:
        # Generate initial plan from question only
        return InitialPlan(approach="Analyze audio content...")
    
    def decide(self, state, available_tools) -> PlannerDecision:
        self.validate_state(state)
        # Your implementation here
        return PlannerDecision(...)
    
    def summarize_evidence(self, state) -> str:
        # Summarize accumulated evidence into a neutral narrative
        return "Summary of evidence..."
    
    def check_format(self, proposed_answer, expected_format, question) -> FormatCheckResult:
        # Validate format compliance only
        return FormatCheckResult(passed=True)
```

### Using MCP Tools

MCP tools run in isolated processes with auto-discovery:

```python
import asyncio
from audio_agent.main import AudioAgent
from audio_agent.tools.catalog import register_all_mcp_tools
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.frontend.qwen2_audio_frontend import Qwen2AudioFrontend
from audio_agent.planner.qwen25_planner import Qwen25Planner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser

async def run_with_tools():
    # Create components
    frontend = Qwen2AudioFrontend()
    planner = Qwen25Planner()
    registry = ToolRegistry()
    fuser = DefaultEvidenceFuser()
    
    # Register all MCP tools from catalog
    server_manager = MCPServerManager()
    await register_all_mcp_tools(registry, server_manager, verbose=True)
    
    # Create agent and run
    agent = AudioAgent(frontend, planner, registry, fuser)
    result = await agent.arun(
        question="What is being said?",
        audio_paths=["/path/to/audio.wav"]
    )
    
    # Cleanup
    await server_manager.shutdown_all()
    return result

asyncio.run(run_with_tools())
```

### Customizing Prompts

All prompts are now externalized as markdown files in `audio_agent/prompts/`. You can customize the behavior of the planner and frontend by editing these files.

**Available prompt files:**

| File | Purpose | Variables |
|------|---------|-----------|
| `frontend_system.md` | Frontend system prompt | None |
| `frontend_user.md` | Frontend user instruction | `{question}`, `{audio_list}`, `{question_oriented_prompt}` |
| `frontend_followup_system.md` | Targeted frontend follow-up system prompt | None |
| `frontend_followup_user.md` | Targeted frontend follow-up user instruction | `{question}`, `{audio_list}`, `{followup_prompt}` |
| `frontend_final_answer_system.md` | Frontend final answer system prompt | None |
| `frontend_final_answer_user.md` | Frontend final answer user instruction | `{question}`, `{expected_output_format}`, `{initial_plan_text}`, `{frontend_initial_text}`, `{evidence_and_history_text}`, `{audio_summary}`, `{format_critique_section}` |
| `initial_prompt_system.md` | Planner system prompt for question-oriented frontend prompt generation | None |
| `initial_prompt_user.md` | Planner user prompt for question-oriented frontend prompt generation | `{question}`, `{caption_skills_reference}` |
| `plan_system.md` | Planner initial planning system prompt | None |
| `plan_user.md` | Planner initial planning user instruction | `{question}`, `{frontend_caption}` |
| `decide_system.md` | Planner decision system prompt | None |
| `decide_user.md` | Planner decision user instruction | `{question}`, `{frontend_caption}`, `{initial_plan}`, `{evidence_log}`, `{tool_call_history}`, `{available_tools}`, `{step_count}`, `{max_steps}` |
| `decide_rules.md` | Planner decision rules | None |
| `disabled_audio_output_decision_guidance.md` | Archived decision-stage audio-output guidance, not loaded by default | None |
| `format_check_system.md` | Format check system prompt | None |
| `format_check_user.md` | Format check user instruction | `{question}`, `{expected_format}`, `{proposed_answer}`, `{is_audio_output_task}` |
| `evidence_summary_system.md` | Evidence summarization system prompt | None |
| `evidence_summary_user.md` | Evidence summarization user instruction | `{question}`, `{frontend_caption}`, `{evidence_text}`, `{planner_trace_text}`, `{tool_history_text}`, `{clarified_intent}`, `{expected_output_format}` |
| `task_oriented_caption_skill.md` | Caption skill reference injected into initial prompt generation | None |
| `task_skills.yaml` | Task skill reference for initial planning | Rendered as markdown cookbook |

**Example: Customizing the frontend system prompt:**

Edit `audio_agent/prompts/frontend_system.md`:
```markdown
You are an expert audio analyst. Focus on identifying speakers, emotions, 
and background sounds relevant to the question.
Return ONLY the caption as plain text.
```

**Example: Adding decision rules:**

Edit `audio_agent/prompts/decide_rules.md` to add custom decision logic:
```markdown
1. If you have enough evidence to answer the question, use action='answer'.
2. If you need transcription, use action='call_tool' with an ASR tool.
3. If you need speaker information, use action='call_tool' with a diarization tool.
...
```

**Loading prompts programmatically:**

```python
from audio_agent.utils.prompt_io import load_prompt

# Load a prompt
system_prompt = load_prompt("frontend_system")
user_prompt = load_prompt("plan_user").format(question="What is being said?")
```

### Adding an MCP Tool

For new tool onboarding, use the **Harness-First Agent Workflow** (recommended):

```bash
# See tool_preparation/README.md for the complete workflow
cat tool_preparation/README.md
```

For manual tool development, see the [Tool Preparation Guide](./tool_preparation/README.md). Quick start:

```bash
# 1. Copy template
cp -r audio_agent/tools/catalog/_template audio_agent/tools/catalog/my_tool

# 2. Edit pyproject.toml, server.py, config.yaml

# 3. Create setup.sh and test_env.sh (see tool_preparation/playbooks/env_uv.md for templates)

# 4. Setup environment
cd audio_agent/tools/catalog/my_tool && ./setup.sh

# 5. Verify
./test_env.sh

# 6. Register and test
./verify_all_tools.sh
```

## License

MIT
