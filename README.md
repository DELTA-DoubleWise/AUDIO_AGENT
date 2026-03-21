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
│   ├── base.py           # BaseFrontend ABC
│   └── dummy_frontend.py # Dummy implementation
├── planner/               # Planner implementations
│   ├── base.py           # BasePlanner ABC
│   └── dummy_planner.py  # Dummy implementation
├── tools/                 # Tool system
│   ├── base.py           # BaseTool ABC
│   ├── registry.py       # ToolRegistry
│   ├── executor.py       # ToolExecutor
│   └── dummy_tools.py    # Dummy tools
├── fusion/                # Evidence fusion
│   ├── base.py           # BaseEvidenceFuser ABC
│   └── default_fuser.py  # Default implementation
├── graph/                 # LangGraph workflow
│   ├── builder.py        # Graph construction
│   ├── nodes.py          # Node functions
│   └── routing.py        # Routing logic
├── config/                # Configuration
│   └── settings.py       # AgentConfig
├── utils/                 # Utilities
│   └── validation.py     # Validation helpers
├── examples/              # Example scripts
│   └── demo_run.py       # Runnable demo
└── tests/                 # Tests
    ├── test_state.py
    ├── test_registry.py
    └── test_graph_smoke.py
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

## Quick Start

```python
from audio_agent.main import create_dummy_agent

# Create agent with dummy components
agent = create_dummy_agent()

# Run on a query
result = agent.run(
    question="What is being discussed in this audio?",
    audio_path_or_uri="/path/to/audio.wav",
)

# Check result
if agent.is_successful(result):
    answer = agent.get_answer(result)
    print(answer.answer)
```

## Running the Demo

```bash
# Using the installed script
audio-agent-demo

# Or directly
python -m audio_agent.examples.demo_run
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
    
    def decide(self, state, available_tools) -> PlannerDecision:
        self.validate_state(state)
        # Your implementation here
        return PlannerDecision(...)
```

## License

MIT
