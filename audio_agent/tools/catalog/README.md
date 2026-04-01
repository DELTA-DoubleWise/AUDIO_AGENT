# Audio Agent Tool Catalog

This directory contains all available MCP (Model Context Protocol) tools for the Audio Agent Framework.

## ⚠️ CRITICAL: Environment Pre-Creation Required

**All tools in this catalog require pre-created environments!**

The MCP servers use explicit venv paths (`.venv/bin/python`) and will **FAIL** if the environment doesn't exist.

---

## Tool Onboarding (Recommended)

For **new tool onboarding**, use the **Harness-First Agent Workflow**:

```bash
# See the complete automated workflow
cat tool_preparation/README.md
```

This workflow automates the entire process of discovering, setting up, and validating new tools.

---

## Manual Tool Setup

For **existing tools** or **manual development**:

### Quick Start

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup a tool (MUST do this before using)
cd audio_agent/tools/catalog/asr_qwen3
./setup.sh

# 3. Verify it's ready
./test_env.sh

# 4. Use in your agent
python -c "
from audio_agent.tools.catalog import load_mcp_server_config
config = load_mcp_server_config('asr_qwen3')
print(f'Ready to use: {config.command}')
"
```

### Setup All Tools

```bash
# Setup all tools at once
./verify_all_tools.sh --setup

# Verify without setup
./verify_all_tools.sh
```

---

## Overview

Tools in this catalog are organized as MCP (Model Context Protocol) servers that run in separate processes with isolated environments. This provides:

- **Dependency Isolation**: Each tool has its own dependencies
- **Reproducibility**: Locked environments via uv
- **Fail-Fast**: Clear errors if environment is missing

---

## Available Tools

| Tool | Description | Resources | Status |
|------|-------------|-----------|--------|
| [asr_qwen3](./asr_qwen3/) | Speech recognition using Qwen3-ASR-1.7B | 8GB RAM, GPU optional | Ready |
| [diarizen](./diarizen/) | Speaker diarization using DiariZen | 8GB RAM, GPU optional | Ready |
| [ffmpeg](./ffmpeg/) | Audio processing with FFmpeg | Minimal | Ready |
| [librosa](./librosa/) | Audio analysis with librosa | Minimal | Ready |
| [omni_captioner](./omni_captioner/) | Audio captioning via Qwen3-Omni API | API key required | Ready |
| [snakers4_silero-vad](./snakers4_silero-vad/) | Voice activity detection | Minimal | Ready |

### Tool Categories

#### Speech Processing
- **ASR** (Automatic Speech Recognition): Transcribe speech to text
  - [asr_qwen3](./asr_qwen3/) - Qwen3-ASR-1.7B based ASR with 52 language support

#### Speaker Analysis
- **Diarization**: Identify who speaks when
  - [diarizen](./diarizen/) - Speaker diarization with WavLM

#### Audio Processing
- **Utilities**: Format conversion, analysis
  - [ffmpeg](./ffmpeg/) - Audio format conversion and processing
  - [librosa](./librosa/) - Audio feature extraction
  - [snakers4_silero-vad](./snakers4_silero-vad/) - Voice activity detection

#### Captioning
- **Omni Captioner**: Generate audio descriptions
  - [omni_captioner](./omni_captioner/) - API-based captioning

---

## Environment Management

### Setup a Specific Tool

```bash
cd audio_agent/tools/catalog/<tool_name>
./setup.sh      # Create environment
./test_env.sh   # Verify environment
```

### Force Recreate Environment

```bash
cd audio_agent/tools/catalog/<tool_name>
rm -rf .venv
./setup.sh
```

### Verify All Tools

```bash
./verify_all_tools.sh
```

---

## Why Explicit Venv Paths?

We use explicit venv paths (`.venv/bin/python`) instead of `uv run` because:

1. **No Runtime Setup**: `uv run` can trigger environment creation during inference
2. **Predictable Latency**: Pre-created environments start immediately
3. **Fail-Fast**: Clear error if environment is missing
4. **Production-Ready**: Explicit dependencies are easier to audit

---

## Tool Structure

Each tool has this structure:

```
asr_qwen3/
├── config.yaml          # Tool configuration (uses .venv/bin/python)
├── server.py            # MCP server implementation
├── pyproject.toml       # Tool dependencies
├── setup.sh             # Environment setup script
├── test_env.py          # Environment verification (Python)
├── test_env.sh          # Environment verification (shell)
├── README.md            # Tool documentation
└── .venv/               # Isolated environment (created by setup.sh)
    └── bin/python
```

---

## Adding a New Tool

### Option 1: Harness-First Agent Workflow (Recommended)

For automated tool onboarding, use the harness-first workflow:

```bash
# See complete workflow documentation
cat tool_preparation/README.md
```

This workflow handles discovery, environment setup, validation, and wrapper generation automatically.

### Option 2: Manual Tool Development

#### 1. Copy the Template

```bash
cp -r _template my_new_tool
cd my_new_tool
```

#### 2. Define Dependencies (pyproject.toml)

```toml
[project]
name = "my-new-tool"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.28.0",
    "librosa>=0.10.0",
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["server", "model"]
```

#### 3. Configure Server (config.yaml)

```yaml
name: my_new_tool
server:
  command: [".venv/bin/python", "server.py"]
  working_dir: "."
  env:
    MODEL_PATH: "model/name"
  resources:
    memory_gb: 4
    gpu: false
  lifecycle: "session"
```

#### 4. Implement server.py

Follow the MCP protocol:
- Handle `initialize` request
- Handle `tools/list` request
- Handle `tools/call` request

See `_template/server.py` for a complete example.

#### 5. Create setup.sh and test_env.sh

Copy templates from `tool_preparation/playbooks/env_uv.md` or use the `_template` versions.

#### 6. Setup Environment

```bash
./setup.sh      # Create environment
./test_env.sh   # Verify environment
```

#### 7. Test

```python
import asyncio
from audio_agent.tools.mcp import MCPClient

async def test():
    client = MCPClient(
        command=[".venv/bin/python", "server.py"],
        working_dir="/path/to/my_new_tool"
    )
    await client.start()
    tools = await client.list_tools()
    print(f"Tools: {[t.name for t in tools]}")
    await client.stop()

asyncio.run(test())
```

---

## Using Tools in Your Agent

```python
from audio_agent.tools.catalog import load_mcp_server_config
from audio_agent.tools.mcp import MCPServerManager, MCPToolAdapter

# Load config (resolves paths)
config = load_mcp_server_config("asr_qwen3")

# Register with server manager
manager = MCPServerManager()
manager.register_config("asr_qwen3", config)

# Discover and register tools
client = await manager.get_client("asr_qwen3")
tools = await client.list_tools()

for tool_info in tools:
    adapter = MCPToolAdapter(
        server_name="asr_qwen3",
        tool_info=tool_info,
        server_manager=manager,
    )
    registry.register_mcp(adapter)
```

---

## Configuration Reference

### config.yaml

```yaml
name: tool_name
description: "What this tool does"
version: "1.0.0"

server:
  command: [".venv/bin/python", "server.py"]
  working_dir: "."  # Resolved relative to tool directory
  
  env:
    KEY: "value"
  
  resources:
    memory_gb: 8
    gpu: true
    cpu_cores: 4
  
  lifecycle: "session"  # per_call | session | persistent
  startup_timeout_sec: 300
```

### Lifecycle Modes

- **`session`**: Keep server alive during agent session (recommended)
- **`per_call`**: Spawn new server for each invocation (stateless, slow)
- **`persistent`**: Always running, managed externally (for production)

---

## Troubleshooting

### "Virtual environment not found" error

You forgot to setup the environment:
```bash
cd audio_agent/tools/catalog/<tool_name>
./setup.sh
```

### Setup fails

Check that pyproject.toml exists and is valid:
```bash
cd audio_agent/tools/catalog/<tool_name>
cat pyproject.toml
```

### Import errors when running

The environment may be outdated. Recreate it:
```bash
cd audio_agent/tools/catalog/<tool_name>
rm -rf .venv
./setup.sh
```

### Verify fails but setup succeeds

There might be a Python version mismatch. Check:
```bash
cd audio_agent/tools/catalog/<tool_name>
.venv/bin/python --version
```

---

## Contributing

When adding a new tool:

1. **Recommended**: Use the Harness-First Agent Workflow (`tool_preparation/`)
2. **Manual**: Copy `_template/` and customize
3. Define minimal dependencies in `pyproject.toml`
4. Use explicit venv path in `config.yaml` (`.venv/bin/python`)
5. Create `setup.sh` and `test_env.sh` scripts
6. **Test setup**: `./setup.sh`
7. **Test verify**: `./test_env.sh`
8. Document resource requirements accurately
9. Update this README with the new tool

---

## See Also

- [MCP Protocol Documentation](https://modelcontextprotocol.io/)
- [Template Tool](./_template/) - Starting point for new tools
- [Tool Preparation Workflow](../../tool_preparation/README.md) - Automated tool onboarding
- [ASR Qwen3 Example](./asr_qwen3/) - Complete working example
- [uv Documentation](https://docs.astral.sh/uv/)
