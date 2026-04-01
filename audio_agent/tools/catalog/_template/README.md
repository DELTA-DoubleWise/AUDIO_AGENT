# Template Tool

This is a template for creating new MCP tools for the audio agent framework.

## ⚠️ IMPORTANT: Environment Setup Required

**You MUST set up the environment BEFORE running this tool!**

The tool will FAIL if the environment is not pre-created.

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv
```

### 2. Setup Environment (REQUIRED)

```bash
# Setup this tool's environment (recommended)
./setup.sh

# Verify it's ready
./test_env.sh
```

### 3. Test the Server

```bash
cd audio_agent/tools/catalog/template_tool
.venv/bin/python server.py
```

## Overview

Brief description of what this tool does and when to use it.

## Capabilities

- Feature 1: Description
- Feature 2: Description
- Feature 3: Description

## Environment Management

### Project Structure

```
template_tool/
├── config.yaml       # Tool configuration (uses .venv/bin/python)
├── server.py         # MCP server implementation
├── pyproject.toml    # Python dependencies
└── README.md         # This file
```

### Managing Dependencies

Edit `pyproject.toml` to add/modify dependencies:

```toml
[project]
dependencies = [
    "requests>=2.28.0",
    "your-new-dependency>=1.0.0",
]
```

Then recreate the environment:

```bash
rm -rf .venv
./setup.sh
```

### Checking Environment Status

```bash
# Verify all tools
./verify_all_tools.sh

# Verify this tool is ready
./test_env.sh
```

## Usage

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "input_text": {
      "type": "string",
      "description": "The input text to process"
    },
    "options": {
      "type": "object",
      "description": "Optional processing options"
    }
  },
  "required": ["input_text"]
}
```

### Example

```python
# Example of how to use this tool
request = {
    "input_text": "Hello world",
    "options": {"language": "en"}
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model | `"default/model"` |
| `DEVICE` | Device to use (`cuda` or `cpu`) | `"cpu"` |
| `LOG_LEVEL` | Logging level | `"INFO"` |

### Resource Requirements

- **Memory**: 4GB RAM
- **GPU**: Not required
- **CPU**: 2 cores

## Testing with MCP Client

```python
import asyncio
from audio_agent.tools.mcp import MCPClient

async def test():
    # The server expects pre-created environment
    client = MCPClient(
        command=[".venv/bin/python", "server.py"],
        working_dir="/path/to/template_tool"
    )
    await client.start()
    
    tools = await client.list_tools()
    print(f"Available: {[t.name for t in tools]}")
    
    result = await client.call_tool("example_tool", {"input_text": "Hello"})
    print(f"Result: {result}")
    
    await client.stop()

asyncio.run(test())
```

## Troubleshooting

### "Virtual environment not found" error

You forgot to setup the environment:
```bash
./setup.sh
```

### "Python executable failed" error

The environment may be corrupted. Recreate it:
```bash
rm -rf .venv
./setup.sh
```

### Import errors

Ensure dependencies are correctly specified in `pyproject.toml` and the environment is up to date:
```bash
./test_env.sh
```

## See Also

- [MCP Protocol Documentation](https://modelcontextprotocol.io/)
- [Tool Catalog README](../README.md)
- [uv Documentation](https://docs.astral.sh/uv/)
