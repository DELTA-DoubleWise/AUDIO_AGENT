# Audio Agent Framework - Agent Guide

This document provides essential information for AI coding agents working on the Audio Agent Framework project.

## Project Overview

The Audio Agent Framework is a **LangGraph-based framework for audio understanding with iterative tool use**. It provides a clean, extensible architecture for building audio understanding agents that:

1. Process one or more audio files with a frontend LALM (Large Audio Language Model) - supports multi-audio comparison tasks
2. Use an LLM planner to reason about evidence and decide next actions
3. Invoke tools iteratively to gather more evidence
4. Accumulate and fuse evidence from multiple sources
5. Produce final answers with supporting evidence

### Runtime Architecture

```
START
  -> initial_prompt_node (planner generates question-oriented prompt from task skills)
  -> frontend_evidence_node (LALM processes audio using question-oriented prompt, generates structured caption evidence)
  -> initial_plan_node (audio-aware planning using question + frontend caption, generates approach and focus points)
  -> planner_decision_node (LLM decides action; on last step forces ANSWER)
  -> [conditional routing based on decision]
     - ANSWER -> final_answer_node (frontend omni model generates answer from audio + context)
       -> format_check_node (mandatory format validation) -> [conditional]
        * Format OK -> answer_node -> END
        * Format Failed -> planner_decision_node (loop with critique as evidence)
     - CALL_TOOL -> tool_executor_node (auto-injects audio_path)
       -> evidence_fusion_node -> planner_decision_node (loop)
     - CLARIFY -> intent_clarification_node -> planner_decision_node
     - FAIL -> failure_node -> END
     - EXHAUSTED (max_steps reached) -> failure_node -> END
```

**Key Behaviors:**
- **Initial Prompt Generation**: A dedicated `initial_prompt_node` uses the planner (text LLM) to craft a customized `question_oriented_prompt` from the user question, referencing `task_oriented_caption_skill.md`. This prompt guides the frontend model.
- **Frontend Evidence**: The `frontend_evidence_node` feeds the `question_oriented_prompt` to the LALM, which produces a richer, structured `question_guided_caption` containing: (1) general caption, (2) focus point, (3) proposed answer + confidence, and (4) uncertainties / verification needs.
- **Initial Planning**: Planner's `plan()` method generates a high-level approach using both the question and the frontend's structured caption, enabling audio-aware planning
- **Tool Execution**: Tool executor automatically resolves `audio_id` references to actual paths and injects audio paths for tools that need them
- **Intent Clarification**: When planner returns CLARIFY action, the intent_clarification_node refines the question before continuing
- **Frontend Final Answer**: When the planner returns ANSWER (or is forced on the final step), the `final_answer_node` invokes the frontend (audio-capable) model with all original audio files and accumulated context to generate the final answer.
- **Format Checking**: Mandatory format validation occurs before final answer. The planner (text LLM) checks if the proposed answer follows the expected output format. If format violations are found, the critique is added as evidence and planning continues.
- **Final Step**: On the last step (`step_count >= max_steps - 1`), the planner decision node forces `action=ANSWER` with `draft_answer=None`, delegating final answer generation to the frontend model.
- **Evidence Accumulation**: Frontend output, tool results, and format check critiques are fused into evidence_log for planner context

## Technology Stack

- **Language**: Python 3.11+
- **Core Framework**: LangGraph >=0.2.0
- **Core Dependencies**: langchain-core >=0.3.0, pydantic >=2.0.0
- **Optional ML Stack**: PyTorch, transformers (from source), accelerate, librosa, soundfile
- **Tool Infrastructure**: MCP (Model Context Protocol), PyYAML for tool configs
- **Environment Management**: `uv` for isolated tool environments
- **Development Tools**: pytest, black, ruff, mypy

## Environment Setup

### Conda Initialization

> **Important**: On this system, conda requires initialization before use. Always run:
> ```bash
> source /lihaoyu/.conda.path.sh
> ```
> before using any `conda` commands.

### Environment Options

1. **Default Environment**: For core framework and dummy components (no GPU required)
   - See: `DEFAULT_ENVIRONMENT.md`

2. **Demo Environment**: For running with real Qwen models (GPU required)
   - See: `DEMO_ENVIRONMENT.md`

### Quick Start

```bash
# Initialize conda
source /lihaoyu/.conda.path.sh

# Create and activate environment (choose one)
conda create -n audio_agent python=3.11
conda activate audio_agent

# Install package
pip install -e .
# Or with dev dependencies: pip install -e ".[dev]"
```

## Project Structure

```
audio_agent/
├── __init__.py
├── main.py                    # Main AudioAgent class and factory functions
├── core/                      # Core contracts and primitives
│   ├── state.py              # AgentState TypedDict definition
│   ├── schemas.py            # Pydantic v2 schemas for all component contracts
│   ├── errors.py             # Exception taxonomy (AudioAgentError hierarchy)
│   ├── constants.py          # AgentStatus enum and defaults
│   └── logging.py            # Structured logging utilities
├── graph/                     # LangGraph workflow assembly
│   ├── builder.py            # Graph construction and compilation
│   ├── nodes.py              # Node factory functions (pure functions)
│   └── routing.py            # Routing logic and decision functions
├── frontend/                  # Audio frontend implementations
│   ├── base.py               # BaseFrontend ABC
│   ├── model_frontend.py     # BaseModelFrontend template class
│   ├── dummy_frontend.py     # Mock implementation for testing
│   ├── qwen2_audio_frontend.py    # Qwen2-Audio adapter (local)
│   ├── qwen3_omni_frontend.py     # Qwen3-Omni adapter (local)
│   └── openai_compatible_frontend.py  # OpenAI-compatible API frontend
├── planner/                   # Planner implementations
│   ├── base.py               # BasePlanner ABC
│   ├── model_planner.py      # BaseModelPlanner template class
│   ├── dummy_planner.py      # Mock implementation for testing
│   ├── qwen25_planner.py     # Qwen2.5 planner adapter
│   └── openai_compatible_planner.py  # OpenAI-compatible API planner
├── tools/                     # Tool system
│   ├── base.py               # BaseTool ABC
│   ├── registry.py           # ToolRegistry for tool management (internal + MCP)
│   ├── executor.py           # ToolExecutor for running tools
│   ├── dummy_tools.py        # Dummy tool implementations
│   ├── mcp/                  # MCP (Model Context Protocol) infrastructure
│   │   ├── client.py         # MCP client for JSON-RPC communication
│   │   ├── server_manager.py # Manages MCP server lifecycle
│   │   ├── tool_adapter.py   # Adapts MCP tools to BaseTool interface
│   │   └── schemas.py        # MCP data models
│   └── catalog/              # MCP tool catalog (external tools)
│       ├── __init__.py       # Catalog exports
│       ├── loader.py         # Auto-discovers and registers MCP tools
│       ├── _template/        # Template for new MCP tools
│       ├── asr_qwen3/        # Qwen3-ASR-1.7B speech recognition
│       ├── diarizen/         # Speaker diarization tool
│       ├── ffmpeg/           # Audio processing with FFmpeg
│       ├── librosa/          # Audio analysis with librosa
│       ├── omni_captioner/   # Qwen3-Omni captioner (API)
│       ├── snakers4_silero-vad/  # Voice activity detection
│       └── evaluation_tool/  # Evaluation utilities
├── prompts/                   # Markdown-based prompt files
│   ├── frontend_system.md           # Frontend system prompt
│   ├── frontend_user.md             # Frontend user instruction template
│   ├── frontend_final_answer_system.md  # Frontend final answer system prompt
│   ├── frontend_final_answer_user.md    # Frontend final answer user instruction
│   ├── plan_system.md               # Planner: initial planning system prompt
│   ├── plan_user.md                 # Planner: initial planning user instruction
│   ├── decide_system.md             # Planner: decision system prompt
│   ├── decide_user.md               # Planner: decision user instruction template
│   ├── decide_rules.md              # Planner: decision rules (numbered list)
│   ├── answer_system.md             # Planner: answer generation system prompt (legacy)
│   ├── answer_user.md               # Planner: answer generation user instruction (legacy)
│   ├── clarify_system.md            # Planner: intent clarification system prompt
│   ├── clarify_user.md              # Planner: intent clarification user instruction
│   ├── verification_system.md       # Verification: system prompt for answer review (legacy)
│   ├── verification_user.md         # Verification: user instruction template (legacy)
│   ├── format_check_system.md       # Format check: system prompt for format validation
│   ├── format_check_user.md         # Format check: user instruction template
│   ├── evidence_summary_system.md   # Evidence summarization system prompt
│   ├── evidence_summary_user.md     # Evidence summarization user instruction
│   └── task_skills.yaml             # Task skill reference for initial planning
├── fusion/                    # Evidence fusion
│   ├── base.py               # BaseEvidenceFuser ABC
│   └── default_fuser.py      # DefaultEvidenceFuser implementation
├── log/                       # Run logging module
│   ├── __init__.py           # Exports: RunLogger, log_run
│   ├── logger.py             # RunLogger class for markdown logs
│   └── formatter.py          # Markdown formatting utilities
├── config/                    # Configuration
│   └── settings.py           # AgentConfig Pydantic model
├── utils/                     # Utilities
│   ├── validation.py         # State validation helpers
│   ├── model_io.py           # Shared model I/O helpers (JSON parsing)
│   ├── prompt_io.py          # Prompt loading from markdown files
│   └── model_downloader.py   # Model download utility
├── examples/                  # Example scripts
│   ├── demo_run.py                # Demo with local Qwen frontend/planner
│   ├── demo_run_auto_tools.py     # Demo with auto MCP tool discovery
│   ├── demo_run_real_asr.py       # Demo with real ASR tool (Qwen3-ASR-1.7B)
│   ├── demo_run_api_planner.py    # Demo with API planner + local frontend
│   └── demo_run_api_full.py       # Demo with API frontend + API planner (no local GPU needed)
└── tests/                     # Unit and smoke tests
    ├── test_state.py
    ├── test_registry.py
    ├── test_graph_smoke.py
    ├── test_frontend_model_base.py
    ├── test_model_io.py
    ├── test_model_planner_base.py
    ├── test_planner_stages.py
    ├── test_qwen25_planner.py
    ├── test_qwen2_audio_frontend.py
    └── test_qwen3_omni_frontend.py
```

tool_preparation/              # Harness-First Agent Workflow for tool onboarding
├── AGENTS.md                 # Master onboarding harness documentation
├── README.md                 # Tool preparation workflow guide
├── SYSTEM_PROMPT.md          # Agent system prompt
├── TOOL_INPUT_TEMPLATE.md    # TOOL_INPUT specification
├── policies/                 # Constitution, evidence priority, backend selection
├── playbooks/                # Environment strategies, failure taxonomy
├── contracts/                # Validation contracts
├── specs/                    # Wrapper contract, model spec templates
└── templates/                # Artifact templates

## Build and Run Commands

### Installation

```bash
# Using conda (recommended for GPU environments)
source /lihaoyu/.conda.path.sh
conda env create -f environment.yml
conda activate audio_agent_demo

# Or using pip
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest audio_agent/tests/ -v

# Run specific test files
pytest audio_agent/tests/test_graph_smoke.py -v
pytest audio_agent/tests/test_state.py audio_agent/tests/test_registry.py -q

# Run with coverage
pytest audio_agent/tests/ --cov=audio_agent
```

### Running the Demo

```bash
# Using the installed console script
audio-agent-demo

# Or directly with arguments
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?" \
  --max-steps 5

# Or using dummy components (no GPU required)
python -c "
from audio_agent.main import create_dummy_agent
agent = create_dummy_agent()
result = agent.run('What is in this audio?', '/fake/audio.wav')
print(result['final_answer'].answer if agent.is_successful(result) else 'Failed')
"
```

### API-Based Demos (No Local GPU)

For users without local GPU resources, use the fully API-based demo:

```bash
# Set API key
export DASHSCOPE_API_KEY="sk-xxx"

# Demo with API planner + local frontend
python -m audio_agent.examples.demo_run_api_planner \
  --audio /path/to/audio.wav \
  --question "What is being said?"

# Demo with API frontend + API planner (fully API-based)
python -m audio_agent.examples.demo_run_api_full \
  --audio /path/to/audio.wav \
  --question "What is being said?"
```

The `demo_run_api_full.py` script uses:
- `OpenAICompatibleFrontend` with qwen3-omni-flash for audio understanding
- `OpenAICompatiblePlanner` with qwen3.5-plus for decision making
- No local model downloads or GPU required

### Setting up MCP Tools

MCP tools require pre-created isolated environments. Each tool has its own `setup.sh` script:

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup individual tools
cd audio_agent/tools/catalog/asr_qwen3 && ./setup.sh && cd -
cd audio_agent/tools/catalog/diarizen && ./setup.sh && cd -
cd audio_agent/tools/catalog/omni_captioner && ./setup.sh && cd -

# 3. Verify a specific tool
cd audio_agent/tools/catalog/asr_qwen3 && ./test_env.sh

# 4. Verify all tools at once
./verify_all_tools.sh

# 5. Setup and verify all tools
./verify_all_tools.sh --setup
```

See [tool_preparation/playbooks/env_uv.md](./AUDIO_AGENT/tool_preparation/playbooks/env_uv.md) for detailed environment setup instructions and templates.

### Running with Real MCP Tools

```bash
# Demo with real ASR tool (requires asr_qwen3 setup)
python -m audio_agent.examples.demo_run_real_asr \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?" \
  --max-steps 5

# Demo with automatic MCP tool discovery
python -m audio_agent.examples.demo_run_auto_tools \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?"
```

### Pre-downloading Models

By default, the framework uses local model paths to avoid re-downloading models on every login. Models are stored in `/lihaoyu/workspace/AUDIO_AGENT/models/`.

**Download all models (one-time setup):**

```bash
# Install with download support
pip install -e ".[download]"

# Download all models
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
- `qwen3-omni` - Qwen/Qwen3-Omni-30B-A3B-Instruct (frontend, ~60GB)
- `qwen2.5` - Qwen/Qwen2.5-7B-Instruct (planner, ~15GB)
- `qwen3-asr` - Qwen/Qwen3-ASR-1.7B (ASR tool, ~4GB)
- `qwen3-aligner` - Qwen/Qwen3-ForcedAligner-0.6B (aligner tool, ~1.5GB)
- `diarizen` - BUT-FIT/diarizen-wavlm-large-s80-md (diarization tool, ~1GB)
- `omni-captioner` - Qwen/Qwen3-Omni-30B-A3B-Captioner (captioner, ~60GB)

**Using HuggingFace Hub paths (fallback):**

If you prefer to use HuggingFace Hub paths directly (models will be downloaded to cache):

```bash
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio.wav \
  --question "What is being said?" \
  --frontend-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --planner-model-path Qwen/Qwen2.5-7B-Instruct
```

**Custom model directory:**

To store models in a different location:

```bash
# Download to custom directory
audio-agent-download-models --all --models-dir /path/to/models

# Run with custom paths
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio.wav \
  --question "What?" \
  --frontend-model-path /path/to/models/Qwen2-Audio-7B-Instruct \
  --planner-model-path /path/to/models/Qwen2.5-7B-Instruct
```

## Code Style Guidelines

### General Principles

- **Fail-Fast**: All inputs validated before processing; explicit exceptions on invalid state
- **Explicit Contracts**: Pydantic models with strict validation; type hints throughout
- **Extensibility**: Abstract base classes for all major components
- **Separation of Concerns**: Frontend, planner, tools, fusion, and graph are cleanly separated

### Style Configuration

The project uses:
- **Black**: line-length = 100, target-version = py311
- **Ruff**: line-length = 100, target-version = py311
  - Selected rules: E, F, W, I, N, UP, B, C4
  - Ignored: E501 (line too long - handled by Black)
- **mypy**: python_version = 3.11, warn_return_any = true, disallow_untyped_defs = true

### Type Hints

- Use Python 3.11+ type hint syntax
- Use `|` for unions (e.g., `str | None`)
- Use `Annotated` for LangGraph state reducers
- All function parameters and return types must be annotated

### Error Handling

- Raise custom exceptions from `audio_agent.core.errors` hierarchy
- Never return `None` when a structured object is expected
- Include context in exceptions via `details` dict
- Wrap external library exceptions into domain exceptions

## Testing Instructions

### Test Organization

- **test_state.py**: AgentState creation and validation tests
- **test_registry.py**: ToolRegistry tests (registration, lookup, duplicate handling, MCP tools)
- **test_graph_smoke.py**: End-to-end smoke tests with dummy components
- **test_frontend_model_base.py**: Frontend template/dispatch/normalization tests
- **test_model_io.py**: Model I/O helper tests (JSON parsing)
- **test_model_planner_base.py**: Model planner base class tests
- **test_planner_stages.py**: Planner stage tests (plan, decide, answer, clarify)
- **test_qwen25_planner.py**: Qwen2.5 planner adapter tests
- **test_qwen*_frontend.py**: Qwen adapter tests (with mock model classes)

### Writing Tests

```python
# Example test pattern
class TestFeature:
    def test_descriptive_test_name(self):
        # Arrange
        agent = create_dummy_agent()
        
        # Act
        result = agent.run("question", "/audio.wav")
        
        # Assert
        assert agent.is_successful(result)
        assert result["final_answer"] is not None
```

### Test Data

- Use fake paths like `/fake/audio.wav` for tests that don't actually load audio
- Mock heavy model dependencies (see `test_qwen3_omni_frontend.py` for examples)

## Development Conventions

### Documentation Updates

**Whenever you make any changes to the codebase, remember to update the relevant documentation:**

1. **Update AGENTS.md** if you:
   - Add new modules or components
   - Change project structure
   - Modify configuration options
   - Add new features that agents should know about

2. **Update README.md** if you:
   - Add new user-facing features
   - Change the project structure diagram
   - Add new example scripts or commands
   - Modify installation or setup instructions

3. **Update .gitignore** if you:
   - Add new directories that should not be committed (e.g., logs/, output/)
   - Generate new temporary or artifact files

4. **Update docstrings** in:
   - Schema classes (describe all fields)
   - Function/method signatures (describe parameters and return values)
   - Module-level docstrings

### Documentation Update Matrix

When you modify code in these locations, update the corresponding documentation:

#### Core Framework Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `audio_agent/core/schemas.py` | AGENTS.md (schema descriptions), PROJECT_MAP.md |
| `audio_agent/core/state.py` | AGENTS.md, PROJECT_MAP.md |
| `audio_agent/core/errors.py` | AGENTS.md |
| `audio_agent/main.py` | README.md, AGENTS.md |
| `audio_agent/config/settings.py` | AGENTS.md, README.md |

#### Frontend Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `audio_agent/frontend/*.py` | AGENTS.md, PROJECT_MAP.md |
| New frontend adapter | README.md, AGENTS.md, PROJECT_MAP.md |

#### Planner Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `audio_agent/planner/*.py` | AGENTS.md, PROJECT_MAP.md |
| `audio_agent/graph/nodes.py` | AGENTS.md, PROJECT_MAP.md |
| `audio_agent/graph/routing.py` | AGENTS.md |
| `audio_agent/graph/builder.py` | AGENTS.md |
| New planner adapter | README.md, AGENTS.md, PROJECT_MAP.md |

#### Tool Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `audio_agent/tools/base.py` | AGENTS.md, PROJECT_MAP.md |
| `audio_agent/tools/registry.py` | AGENTS.md |
| `audio_agent/tools/executor.py` | AGENTS.md |
| `audio_agent/tools/mcp/*.py` | AGENTS.md, PROJECT_MAP.md |
| `audio_agent/tools/catalog/<tool>/` | README.md (add to tool list), AGENTS.md |
| New MCP tool | README.md, AGENTS.md, PROJECT_MAP.md, tool_preparation/*.md |

#### Fusion Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `audio_agent/fusion/*.py` | AGENTS.md, PROJECT_MAP.md |

#### Log / Output Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `audio_agent/log/*.py` | AGENTS.md, README.md, PROJECT_MAP.md |
| `audio_agent/core/logging.py` | AGENTS.md |
| Output audio handling | AGENTS.md, README.md |

#### Utility Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `audio_agent/utils/*.py` | AGENTS.md (if widely used) |

#### Example Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `audio_agent/examples/*.py` | README.md (if new demo features) |

#### Tool Preparation Changes
| Code Location | Documents to Update |
|---------------|---------------------|
| `tool_preparation/policies/*.md` | tool_preparation/README.md |
| `tool_preparation/playbooks/*.md` | tool_preparation/README.md |
| `tool_preparation/contracts/*.md` | tool_preparation/README.md |
| `tool_preparation/specs/*.md` | tool_preparation/README.md |

### Quick Reference: Change Type → Documentation

| Change Type | Primary Doc | Secondary Docs |
|-------------|-------------|----------------|
| Add new module | AGENTS.md | PROJECT_MAP.md, README.md |
| Add new package | AGENTS.md | PROJECT_MAP.md, README.md |
| Add new tool | README.md | AGENTS.md, PROJECT_MAP.md, tool_preparation/*.md |
| Modify schema | AGENTS.md | PROJECT_MAP.md |
| Add config option | AGENTS.md | README.md |
| Change graph logic | AGENTS.md | PROJECT_MAP.md |
| Modify environment setup | DEFAULT_ENVIRONMENT.md | DEMO_ENVIRONMENT.md, AGENTS.md |
| Add new example | README.md | AGENTS.md |

### Adding a New Component

1. **Frontend (Local Model)**: Subclass `BaseModelFrontend`, implement `initialize_model()`, `call_model()`, and `generate_final_answer()`. Prompts are loaded from markdown files via `load_prompt()`.
2. **Frontend (API)**: Use `OpenAICompatibleFrontend` or subclass it. Override `build_api_model_input()` to customize how audio is sent to the API, and `generate_final_answer()` for final answer generation.
3. **Planner (Local Model)**: Subclass `BaseModelPlanner`, implement `plan()`, `decide()`, and `clarify_intent()`. Prompts are loaded from markdown files.
4. **Planner (API)**: Use `OpenAICompatiblePlanner` with any OpenAI-compatible API.
5. **Tool**: Subclass `BaseTool`, implement `spec` property and `invoke()`
6. **Fuser**: Subclass `BaseEvidenceFuser`, implement `fuse()`

### Adding a Graph Node

1. Create node factory function in `audio_agent/graph/nodes.py`
2. Add routing logic in `audio_agent/graph/routing.py` if needed
3. Wire in `audio_agent/graph/builder.py`
4. Update `AgentState` in `audio_agent/core/state.py` if new fields needed
5. **Update logging module** in `audio_agent/log/` if the node produces results that should be logged:
   - Add formatter function in `formatter.py` (e.g., `format_<node_name>_result()`)
   - Update `logger.py` to include the new section in `_build_markdown()`
   - See existing examples: `format_frontend_final_answer()`, `format_format_check_result()`

### Customizing Prompts

All prompts are externalized as markdown files in `audio_agent/prompts/`. This allows easy customization without code changes.

**Prompt File Organization:**

| File | Purpose | Variables |
|------|---------|-----------|
| `frontend_system.md` | Frontend system prompt | None |
| `frontend_user.md` | Frontend user instruction | `{question}`, `{audio_paths}` |
| `frontend_final_answer_system.md` | Frontend final answer system prompt | None |
| `frontend_final_answer_user.md` | Frontend final answer user instruction | `{question}`, `{expected_output_format}`, `{initial_plan_text}`, `{evidence_text}`, `{planner_trace_text}`, `{tool_history_text}`, `{audio_summary}`, `{format_critique_section}` |
| `plan_system.md` | Planner initial planning system prompt | None |
| `plan_user.md` | Planner initial planning user instruction | `{question}` |
| `decide_system.md` | Planner decision system prompt | None |
| `decide_user.md` | Planner decision user instruction | `{question}`, `{frontend_caption}`, `{initial_plan}`, `{evidence_log}`, `{tool_call_history}`, `{available_tools}`, `{step_count}`, `{max_steps}` |
| `decide_rules.md` | Planner decision rules (numbered list) | None |
| `answer_system.md` | Planner answer system prompt | None |
| `answer_user.md` | Planner answer user instruction | `{question}`, `{evidence_text}` |
| `clarify_system.md` | Planner clarify system prompt | None |
| `clarify_user.md` | Planner clarify user instruction | `{question}`, `{clarified_intent}`, `{expected_format}`, `{evidence_text}` |
| `verification_system.md` | Verification: system prompt for answer review | None |
| `verification_user.md` | Verification: user instruction template | `{question}`, `{proposed_answer}` |
| `format_check_system.md` | Format check: system prompt for format validation | None |
| `format_check_user.md` | Format check: user instruction template | `{question}`, `{expected_format}`, `{proposed_answer}` |
| `evidence_summary_system.md` | Evidence summarization system prompt | None |
| `evidence_summary_user.md` | Evidence summarization user instruction | `{question}`, `{frontend_caption}`, `{evidence_text}`, `{planner_trace_text}`, `{tool_history_text}`, `{clarified_intent}`, `{expected_output_format}` |
| `task_skills.yaml` | Task skill reference for initial planning | Rendered as markdown cookbook |

**Loading Prompts:**

```python
from audio_agent.utils.prompt_io import load_prompt

# Load a prompt file
system_prompt = load_prompt("frontend_system")  # Loads frontend_system.md
user_prompt = load_prompt("plan_user").format(question="What is being said?")
```

**Template Variables:**

- Use Python's `str.format()` syntax: `{variable_name}`
- Variables are substituted at runtime when building model inputs
- Missing variables will raise `KeyError` (fail-fast behavior)

**Decision Rules Format:**

The `decide_rules.md` file uses a numbered list format:
```markdown
1. If you have enough evidence to answer the question, use action='answer'.
2. If you need more information, use action='call_tool'.
3. If the question is unclear, use action='clarify'.
...
```

Rules are parsed from the markdown and injected into the decision payload as a list.

**Best Practices:**

1. Keep system prompts focused on role definition and high-level behavior
2. Use user prompts for task-specific instructions and variable content
3. Decision rules should be clear, actionable, and mutually exclusive
4. Test prompt changes with real model calls to verify behavior

### Adding an MCP Tool

#### Option 1: Harness-First Agent Workflow (Recommended for New Tools)

For automated tool onboarding, use the harness-first workflow:

```bash
# See the complete automated workflow
cat tool_preparation/README.md
```

This workflow handles discovery, environment setup, validation, and wrapper generation automatically.

#### Option 2: Manual Tool Development

MCP tools run in isolated processes and communicate via JSON-RPC:

1. **Copy the template:**
   ```bash
   cp -r audio_agent/tools/catalog/_template audio_agent/tools/catalog/my_tool
   cd audio_agent/tools/catalog/my_tool
   ```

2. **Define dependencies in `pyproject.toml`:**
   ```toml
   [project]
   name = "my-tool"
   version = "1.0.0"
   requires-python = ">=3.11"
   dependencies = ["requests>=2.28.0"]
   ```

3. **Implement `server.py`** - MCP server with handlers for:
   - `initialize` - Server initialization
   - `tools/list` - Return available tools
   - `tools/call` - Execute tool logic
   - `shutdown` - Cleanup

4. **Configure `config.yaml`:**
   ```yaml
   name: my_tool
   server:
     command: [".venv/bin/python", "server.py"]  # Use explicit venv path
     working_dir: "."
     env:
       MODEL_PATH: "/path/to/model"
     resources:
       memory_gb: 4
       gpu: false
   ```

5. **Add to model downloader** (if using ML models):
   - Edit `audio_agent/utils/model_downloader.py`
   - Add model to `MODELS` dict

6. **Create setup.sh and test_env.sh:**
   - Copy templates from `tool_preparation/playbooks/env_uv.md`
   - Customize for your tool's requirements

7. **Setup environment:**
   ```bash
   cd audio_agent/tools/catalog/my_tool
   ./setup.sh        # Creates .venv and installs dependencies
   ./test_env.sh     # Verifies the environment
   ```

8. **Test the tool:**
   ```python
   import asyncio
   from audio_agent.tools.catalog import load_mcp_server_config
   from audio_agent.tools.mcp import MCPClient
   
   async def test():
       config = load_mcp_server_config("my_tool")
       client = MCPClient(config.command, config.working_dir)
       await client.start()
       tools = await client.list_tools()
       print(f"Available: {[t.name for t in tools]}")
       await client.stop()
   
   asyncio.run(test())
   ```

9. **Verify all tools:**
   ```bash
   ./verify_all_tools.sh
   ```

See [tool_preparation/README.md](./AUDIO_AGENT/tool_preparation/README.md) for the complete harness-first workflow guide.

### Schema Changes

1. Update Pydantic models in `audio_agent/core/schemas.py`
2. Use `extra="forbid"` for strict validation
3. Add `@model_validator(mode="after")` for cross-field validation
4. Update related tests

### Validation Pattern

```python
from audio_agent.utils.validation import validate_state_has_fields

validate_state_has_fields(
    state,
    ["question", "initial_frontend_output"],
    context="planner_decision_node",
)
```

## Security Considerations

1. **Input Validation**: All inputs validated before processing (fail-fast)
2. **No Silent Failures**: Errors are explicit and logged
3. **Tool Execution**: Tool names and args are validated before execution
4. **Model Output Parsing**: JSON parsing is strict with clear error messages

## Environment Requirements

### For Basic Development (No GPU)

- Python 3.11+
- Core dependencies only (langgraph, langchain-core, pydantic)
- Can run dummy agent and most tests

### For Full Demo (GPU Required)

- Linux recommended
- NVIDIA GPU with CUDA 12.1+
- 32GB+ RAM
- 40-60GB disk space for models
- PyTorch with CUDA support
- transformers from GitHub (not PyPI) for Qwen2-Audio support

## Key Configuration Files

- **pyproject.toml**: Package metadata, dependencies, tool configs (pytest, black, ruff, mypy)
- **environment.yml**: Conda environment with PyTorch + CUDA
- **requirements.txt**: Pip dependencies (subset of pyproject.toml)

## Important Notes

1. **Qwen Models**: The demo uses real Qwen models that require significant GPU resources. Both frontend (Qwen2-Audio-7B) and planner (Qwen2.5-7B) load in one process.

2. **Two-Stage Prompting for Frontend Captioning**: The workflow now includes an `initial_prompt_node` that generates a task-specific prompt for the frontend model. This prompt decomposes the question into focus points and tasks, helping the frontend produce a more structured and actionable caption.

3. **Audio-Aware Initial Planning**: The `initial_plan_node` now receives the frontend's structured question-guided caption. The planner is instructed to identify uncertain or ambiguous statements in the caption and to plan tool-based verification (e.g., ASR, diarization, librosa) for those aspects.

2. **Multi-Audio Support**: The framework supports processing multiple audio files in a single run. Use `audio_paths: list[str]` instead of a single path. Each audio is assigned an ID (`audio_0`, `audio_1`, etc.) for tool reference. This enables speaker verification, audio comparison, and other multi-source analysis tasks.

3. **Transformers Version**: Must install transformers from GitHub source for Qwen2-Audio support: `pip install git+https://github.com/huggingface/transformers`

4. **MCP Tools**: The framework supports MCP (Model Context Protocol) tools that run in isolated processes. Each tool has its own `setup.sh` and `test_env.sh` for environment management. Use `./verify_all_tools.sh` to bulk-verify all tools.

5. **Async Support**: When using MCP tools, use `agent.arun()` instead of `agent.run()` for asynchronous execution.

6. **Planner/Tools Status**: Core architecture is complete with real Qwen2.5 planner. Tools include both dummy implementations and real MCP-based tools (asr_qwen3, diarizen, omni_captioner, ffmpeg, librosa, snakers4_silero-vad).

7. **Prompt System**: All prompts are externalized in `audio_agent/prompts/` as markdown files. The system uses `load_prompt()` from `audio_agent/utils/prompt_io.py` to load prompts at runtime. This enables easy customization without code changes.

8. **API Frontend**: The framework now includes `OpenAICompatibleFrontend` for API-based audio understanding (e.g., qwen3-omni-flash via DashScope). This enables fully API-based deployments without local GPU requirements.

9. **API Planner**: The framework includes `OpenAICompatiblePlanner` for API-based planning (e.g., qwen3.5-plus, kimi-k2.5 via DashScope or OpenAI). Use `create_openai_planner()` helper function.

10. **Intent Clarification**: The planner can return a CLARIFY action when the question is unclear. This triggers the intent_clarification_node which refines the question before continuing.

11. **Frontend Final Answer Generation**: When the planner decides to ANSWER (or is forced on the final step), the `final_answer_node` invokes the frontend model with all original audio files and accumulated context. The frontend generates the final answer directly, ensuring it is grounded in the actual audio content. The context includes:
    - `evidence_log` - all accumulated evidence items
    - `planner_trace` - all previous planner decisions
    - `tool_call_history` - record of all tool invocations
    - `initial_plan`, `clarified_intent`, `expected_output_format`
    - `format_critique` - if a previous format check failed

12. **Format Checking**: Mandatory format validation occurs before final answer. The planner (text LLM) checks if the proposed answer follows the expected output format (from `initial_plan.expected_output_format`). This is different from answer generation:
    - Format check validates structure/format compliance only, NOT content correctness
    - Format check uses the text LLM (planner), not the audio model
    - If format violations are found, the critique is added as evidence and planning continues
    Configure via `AgentConfig(enable_format_check=True, max_format_checks=2)`.

13. **AgentConfig Fields**: Some fields (`planner_name`, `frontend_name`, `fail_on_tool_error`) exist in config but are not fully wired into orchestration logic yet.

14. **Checkpointer Support**: `build_graph_with_config()` exists but is not used by default `AudioAgent` constructor.

15. **Conda Initialization**: Remember to run `source /lihaoyu/.conda.path.sh` before using conda commands on this system.

16. **Model Output Retry**: `BaseModelPlanner` and `BaseModelFrontend` automatically retry model calls when the output fails schema validation or JSON parsing. This handles transient API instability without failing the entire agent run. Configure via `max_retries` (default 3, can be set to 0 to disable). The retry uses exponential backoff (0.5s, 1s, 2s). `AgentConfig` exposes `max_model_output_retries` for documentation purposes; wire it into your planner/frontend constructor as needed.

16. **Run Logging**: The framework automatically logs each run to a Markdown file in the `logs/` directory. This includes:
    - Complete AgentState with all evidence, tool calls, and planner decisions
    - Frontend output and initial plan
    - Final answer with output audio information
    - Format check results and critiques
    - All errors and metadata
    Configure via `AgentConfig(log_dir="./logs", enable_run_logging=True)`.

17. **Audio Output Handling**: For tasks that produce audio files (e.g., trimming, conversion), the framework:
    - Detects audio output requirements at planning stage (`requires_audio_output` flag)
    - Tracks generated audio files in `audio_list`
    - Copies output audio to a dedicated `output/` directory
    - Includes structured output audio info in `FinalAnswer.output_audio`
    - Prompts guide LLMs to reference audio by ID, not raw file paths

## Reading Order for New Agents

1. `README.md` - Architecture overview
2. `DEFAULT_ENVIRONMENT.md` - Default environment setup
3. `DEMO_ENVIRONMENT.md` - Demo environment with real models
4. `tool_preparation/README.md` - Harness-first tool onboarding workflow
5. `audio_agent/main.py` - Entry point and AudioAgent class
6. `audio_agent/core/state.py` and `audio_agent/core/schemas.py` - Core contracts
7. `audio_agent/graph/builder.py`, `nodes.py`, `routing.py` - Workflow orchestration
8. `audio_agent/frontend/base.py` and `model_frontend.py` - Frontend patterns
9. `audio_agent/frontend/openai_compatible_frontend.py` - API frontend implementation
10. `audio_agent/planner/base.py` and `openai_compatible_planner.py` - Planner interface
11. `audio_agent/prompts/` - Markdown prompt files
12. `audio_agent/tests/test_graph_smoke.py` - Usage examples
