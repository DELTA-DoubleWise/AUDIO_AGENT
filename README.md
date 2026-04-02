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
  -> frontend_evidence_node (LALM processes audio)
  -> planner_node (LLM decides action)
  -> [routing based on decision]
     - ANSWER -> answer_node -> END
     - CALL_TOOL -> tool_executor_node -> evidence_fusion_node -> planner_node (loop)
     - FAIL -> failure_node -> END
```

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
│   └── openai_compatible_frontend.py # OpenAI-compatible API frontend
├── planner/               # Planner implementations
│   ├── base.py            # BasePlanner ABC
│   ├── model_planner.py   # BaseModelPlanner template
│   ├── dummy_planner.py   # Dummy implementation
│   └── qwen25_planner.py  # Qwen2.5 planner adapter
├── tools/                 # Tool system
│   ├── base.py            # BaseTool ABC
│   ├── registry.py        # ToolRegistry (internal + MCP tools)
│   ├── executor.py        # ToolExecutor
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
│       └── omni_captioner/ # Qwen3-Omni captioner
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
│   ├── frontend_system.md # Frontend system prompt
│   ├── frontend_user.md   # Frontend user instruction
│   ├── plan_system.md     # Planner planning system prompt
│   ├── plan_user.md       # Planner planning user instruction
│   ├── decide_system.md   # Planner decision system prompt
│   ├── decide_user.md     # Planner decision user instruction
│   ├── decide_rules.md    # Planner decision rules
│   ├── answer_system.md   # Planner answer system prompt
│   ├── answer_user.md     # Planner answer user instruction
│   ├── clarify_system.md  # Planner clarify system prompt
│   └── clarify_user.md    # Planner clarify user instruction
├── config/                # Configuration
│   └── settings.py       # AgentConfig
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
│   └── demo_run_api_full.py       # Demo with API frontend + API planner (no GPU)
└── tests/                 # Tests
    ├── test_state.py
    ├── test_registry.py
    ├── test_graph_smoke.py
    └── ...
```

## Installation

```bash
# Create and activate conda environment
conda create -n audio_agent python=3.11
conda activate audio_agent

# Install the package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Running the Demo

The demo uses real models (Qwen2-Audio frontend, Qwen2.5 planner) with automatic MCP tool discovery:

```bash
# Setup MCP tools first (requires uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup individual tools
cd audio_agent/tools/catalog/asr_qwen3 && ./setup.sh && cd -
cd audio_agent/tools/catalog/diarizen && ./setup.sh && cd -
cd audio_agent/tools/catalog/omni_captioner && ./setup.sh && cd -

# Or use the helper script to setup all tools
./verify_all_tools.sh --setup

# Download models
audio-agent-download-models --models qwen2-audio qwen2.5 qwen3-asr

# Run the demo with auto tool discovery
python -m audio_agent.examples.demo_run_auto_tools \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?"
```

### Verifying Tool Environments

To verify all MCP tools are properly configured:

```bash
# Test all tools
./verify_all_tools.sh

# Setup and test all tools
./verify_all_tools.sh --setup
```

See also `demo_run_real_asr.py` for a demo with specific ASR tool configuration.

### API-Based Demos (No Local GPU Required)

If you don't have a local GPU or prefer to use API-based models:

```bash
# Setup MCP tools first (same as above)
./verify_all_tools.sh --setup

# Demo with API planner + local frontend
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
```

The `demo_run_api_full.py` script is ideal for:
- Users without local GPU resources
- Quick prototyping and testing
- Deployments where model inference is handled externally

## Pre-downloading Models

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
- `diarizen` - BUT-FIT/diarizen-wavlm-large-s80-md (diarization, ~1GB)
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

## Running Tests

```bash
pytest audio_agent/tests/ -v
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
    
    def run(self, question: str, audio_path_or_uri: str) -> FrontendOutput:
        self.validate_inputs(question, audio_path_or_uri)
        # Your implementation here
        return FrontendOutput(
            caption="...",
            confidence=0.9,
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
    
    def answer(self, state) -> str:
        # Generate final answer from accumulated evidence
        return "Based on the evidence..."
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
        audio_path_or_uri="/path/to/audio.wav"
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
| `frontend_user.md` | Frontend user instruction | `{question}`, `{audio_path_or_uri}` |
| `plan_system.md` | Planner initial planning system prompt | None |
| `plan_user.md` | Planner initial planning user instruction | `{question}` |
| `decide_system.md` | Planner decision system prompt | None |
| `decide_user.md` | Planner decision user instruction | `{question}`, `{frontend_caption}`, `{initial_plan}`, `{evidence_log}`, `{tool_call_history}`, `{available_tools}`, `{step_count}`, `{max_steps}` |
| `decide_rules.md` | Planner decision rules | None |
| `answer_system.md` | Planner answer system prompt | None |
| `answer_user.md` | Planner answer user instruction | `{question}`, `{evidence_text}` |
| `clarify_system.md` | Planner clarify system prompt | None |
| `clarify_user.md` | Planner clarify user instruction | `{question}`, `{clarified_intent}`, `{expected_format}`, `{evidence_text}` |

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
