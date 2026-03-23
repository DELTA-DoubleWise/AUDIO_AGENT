# Audio Agent Tool Catalog

This directory contains all available tools for the audio agent framework.

## ⚠️ CRITICAL: Environment Pre-Creation Required

**All tools in this catalog require pre-created environments!**

The MCP servers use explicit venv paths (`.venv/bin/python`) and will **FAIL** if the environment doesn't exist.

### Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup a tool (MUST do this before using)
python -m audio_agent.tools.catalog.setup_tool asr_qwen3

# 3. Verify it's ready
python -m audio_agent.tools.catalog.setup_tool asr_qwen3 --verify

# 4. Use in your agent
python -c "
from audio_agent.tools.catalog import load_mcp_server_config
config = load_mcp_server_config('asr_qwen3')
print(f'Ready to use: {config.command}')
"
```

## Overview

Tools in this catalog are organized as MCP (Model Context Protocol) servers that run in separate processes with isolated environments. This provides:

- **Dependency Isolation**: Each tool has its own dependencies
- **Reproducibility**: Locked environments via uv
- **Fail-Fast**: Clear errors if environment is missing

## Available Tools

| Tool | Description | Resources | Status |
|------|-------------|-----------|--------|
| [asr_qwen3](./asr_qwen3/) | Speech recognition using Qwen3-ASR-1.7B | 8GB RAM, GPU optional | Ready |

### Tool Categories

#### Speech Processing
- **ASR** (Automatic Speech Recognition): Transcribe speech to text
  - [asr_qwen3](./asr_qwen3/) - Qwen3-ASR-1.7B based ASR with 52 language support

## Environment Management Commands

```bash
# Setup a specific tool (creates .venv)
python -m audio_agent.tools.catalog.setup_tool <tool_name>

# Setup all tools
python -m audio_agent.tools.catalog.setup_tool --all

# Verify a tool is ready (dry-run check)
python -m audio_agent.tools.catalog.setup_tool <tool_name> --verify

# Verify all tools
python -m audio_agent.tools.catalog.setup_tool --verify-all

# List all tool statuses
python -m audio_agent.tools.catalog.setup_tool --list

# Force recreate environment
python -m audio_agent.tools.catalog.setup_tool <tool_name> --force
```

## Why Explicit Venv Paths?

We use explicit venv paths (`.venv/bin/python`) instead of `uv run` because:

1. **No Runtime Setup**: `uv run` can trigger environment creation during inference
2. **Predictable Latency**: Pre-created environments start immediately
3. **Fail-Fast**: Clear error if environment is missing
4. **Production-Ready**: Explicit dependencies are easier to audit

## Tool Structure

Each tool has this structure:

```
asr_qwen3/
├── config.yaml          # Tool configuration (uses .venv/bin/python)
├── server.py            # MCP server implementation
├── pyproject.toml       # Tool dependencies
├── README.md            # Tool documentation
└── .venv/               # Isolated environment (created by setup_tool)
    └── bin/python
```

## Adding a New Tool

### 1. Copy the Template

```bash
cp -r _template my_new_tool
cd my_new_tool
```

### 2. Define Dependencies (pyproject.toml)

```toml
[project]
name = "my-new-tool"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.28.0",
    "librosa>=0.10.0",
]
```

### 3. Configure Server (config.yaml)

```yaml
name: my_new_tool
server:
  # IMPORTANT: Must run setup_tool first!
  command: [".venv/bin/python", "server.py"]
  working_dir: "."
  env:
    MODEL_PATH: "model/name"
  resources:
    memory_gb: 4
    gpu: false
  lifecycle: "session"
```

### 4. Implement server.py

Follow the MCP protocol:
- Handle `initialize` request
- Handle `tools/list` request
- Handle `tools/call` request

See `_template/server.py` for a complete example.

### 5. Setup Environment

```bash
python -m audio_agent.tools.catalog.setup_tool my_new_tool
python -m audio_agent.tools.catalog.setup_tool my_new_tool --verify
```

### 6. Test

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

## Configuration Reference

### config.yaml

```yaml
name: tool_name
description: "What this tool does"
version: "1.0.0"

server:
  # MUST pre-create: python -m audio_agent.tools.catalog.setup_tool tool_name
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

## Troubleshooting

### "Virtual environment not found" error

You forgot to setup the environment:
```bash
python -m audio_agent.tools.catalog.setup_tool <tool_name>
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
python -m audio_agent.tools.catalog.setup_tool <tool_name> --force
```

### Verify fails but setup succeeds

There might be a Python version mismatch. Check:
```bash
cd audio_agent/tools/catalog/<tool_name>
.venv/bin/python --version
```

## Contributing

When adding a new tool:

1. Copy `_template/` and customize
2. Define minimal dependencies in `pyproject.toml`
3. Use explicit venv path in `config.yaml` (`.venv/bin/python`)
4. **Test setup**: `python -m audio_agent.tools.catalog.setup_tool <name>`
5. **Test verify**: `python -m audio_agent.tools.catalog.setup_tool <name> --verify`
6. Document resource requirements accurately
7. Update this README with the new tool

## See Also

- [MCP Protocol Documentation](https://modelcontextprotocol.io/)
- [Template Tool](./_template/) - Starting point for new tools
- [ASR Qwen3 Example](./asr_qwen3/) - Complete working example
- [uv Documentation](https://docs.astral.sh/uv/)
