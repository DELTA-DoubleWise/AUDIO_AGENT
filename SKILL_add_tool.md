# Skill: Add New Tools to Audio Agent Framework

## Overview

This skill guides you through adding a new MCP-based tool to the Audio Agent Framework. Tools run in isolated environments using `uv` and communicate via the Model Context Protocol (MCP).

## Prerequisites

- Understanding of Python 3.11+
- Familiarity with the MCP protocol (JSON-RPC over stdio)
- `uv` installed for environment management

## Quick Reference

```bash
# 1. Create tool directory from template
cp -r audio_agent/tools/catalog/_template audio_agent/tools/catalog/my_tool

# 2. Edit pyproject.toml with dependencies

# 3. Implement server.py

# 4. Update config.yaml

# 5. If using ML models, add to model_downloader.py

# 6. Setup environment
python -m audio_agent.tools.catalog.setup_tool my_tool

# 7. Test
python -m audio_agent.tools.catalog.setup_tool my_tool --verify
```

---

## Detailed Steps

### Step 1: Create Tool Directory Structure

Create a new directory under `audio_agent/tools/catalog/`:

```
audio_agent/tools/catalog/my_tool/
├── __init__.py          # (optional) Package marker
├── config.yaml          # Tool configuration and metadata
├── server.py            # MCP server implementation
├── pyproject.toml       # Tool dependencies
└── README.md            # Documentation
```

**Copy from template:**
```bash
cp -r audio_agent/tools/catalog/_template audio_agent/tools/catalog/my_tool
cd audio_agent/tools/catalog/my_tool
```

---

### Step 2: Define Dependencies (pyproject.toml)

Create `pyproject.toml` with minimal, specific dependencies:

```toml
[project]
name = "my-tool"
version = "1.0.0"
description = "Brief description of what this tool does"
requires-python = ">=3.11"
dependencies = [
    # List ONLY what this tool needs
    # Examples:
    "requests>=2.28.0",
    "numpy>=1.24.0",
    # For ML models:
    # "torch>=2.0.0",
    # "transformers>=4.40.0",
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

**Important:** Keep dependencies minimal. Don't include packages the tool doesn't directly use.

---

### Step 3: Implement MCP Server (server.py)

The server must implement the MCP protocol with these handlers:

```python
#!/usr/bin/env python3
"""MCP Server for my_tool."""

from __future__ import annotations

import json
import os
import sys
from typing import Any


class MyToolServer:
    """MCP Server implementation."""
    
    def __init__(self):
        self._initialized = False
        # Load config from environment
        self._config_value = os.environ.get("CONFIG_KEY", "default")
        
        # Define available tools
        self._tools = [
            {
                "name": "my_tool_action",
                "description": "What this tool does",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input_param": {
                            "type": "string",
                            "description": "Description of parameter"
                        }
                    },
                    "required": ["input_param"]
                }
            }
        ]
    
    def run(self) -> None:
        """Run the server, reading from stdin."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response:
                    self._send_response(response)
            except json.JSONDecodeError as e:
                self._send_error(None, -32700, f"Parse error: {e}")
            except Exception as e:
                request_id = request.get("id") if isinstance(request, dict) else None
                self._send_error(request_id, -32603, f"Internal error: {e}")
    
    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle JSON-RPC request."""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return self._handle_tools_call(request_id, params)
        elif method == "shutdown":
            return self._handle_shutdown(request_id)
        else:
            return self._error_response(request_id, -32601, f"Method not found: {method}")
    
    def _handle_initialize(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "my-tool-server",
                    "version": "1.0.0"
                }
            }
        }
    
    def _handle_tools_list(self, request_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": self._tools}
        }
    
    def _handle_tools_call(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle tools/call request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            result = self._execute_tool(tool_name, arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                    "error": str(e)
                }
            }
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Execute the requested tool."""
        if tool_name == "my_tool_action":
            return self._my_tool_action(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _my_tool_action(self, arguments: dict) -> dict[str, Any]:
        """Implement your tool logic here."""
        input_param = arguments.get("input_param", "")
        
        # Your tool logic here
        result = f"Processed: {input_param}"
        
        return {
            "content": [{"type": "text", "text": result}],
            "isError": False
        }
    
    def _handle_shutdown(self, request_id: Any) -> dict[str, Any]:
        """Handle shutdown request."""
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    
    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message}
        }
    
    def _send_response(self, response: dict[str, Any]) -> None:
        """Send response to stdout."""
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    
    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        """Send error response."""
        self._send_response(self._error_response(request_id, code, message))


def main():
    """Main entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    server = MyToolServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
```

---

### Step 4: Configure Tool (config.yaml)

```yaml
name: my_tool
description: "Brief description of what this tool does"
version: "1.0.0"
author: "Your Name"
tags: ["category", "tag"]

server:
  # CRITICAL: Must use explicit venv path
  # Run setup_tool first: python -m audio_agent.tools.catalog.setup_tool my_tool
  command: [".venv/bin/python", "server.py"]
  working_dir: "."
  
  env:
    # Environment variables passed to the server
    CONFIG_KEY: "value"
    MODEL_PATH: "/path/to/model"  # or HuggingFace repo
  
  resources:
    memory_gb: 4
    gpu: false
    cpu_cores: 2
  
  lifecycle: "session"
  startup_timeout_sec: 60

# Tool definitions (for documentation)
tools:
  - name: "my_tool_action"
    description: "What this tool does"
    input_schema:
      type: object
      properties:
        input_param:
          type: string
          description: "Description of parameter"
      required: ["input_param"]
```

**Critical Points:**
- Use explicit venv path `.venv/bin/python` (NOT `uv run`)
- Environment variables are set at server startup
- `working_dir: "."` resolves to the tool's directory

---

### Step 5: Add to Model Downloader (If Using ML Models)

If your tool uses HuggingFace models, add them to `audio_agent/utils/model_downloader.py`:

```python
MODELS: dict[str, dict[str, Any]] = {
    # ... existing models ...
    "my-model": {
        "repo_id": "Organization/Model-Name",
        "description": "Description of the model",
        "subdir": "Model-Name",
    },
}

# Add convenience constant
DEFAULT_MY_MODEL_PATH = str(DEFAULT_MODELS_DIR / MODELS["my-model"]["subdir"])
```

Then update `config.yaml` to use the local path:
```yaml
server:
  env:
    MODEL_PATH: "/lihaoyu/workspace/AUDIO_AGENT/models/Model-Name"
```

**To download models:**
```bash
audio-agent-download-models --models my-model
```

---

### Step 6: Setup Tool Environment

```bash
# Setup the tool's isolated environment
python -m audio_agent.tools.catalog.setup_tool my_tool

# Verify it's ready
python -m audio_agent.tools.catalog.setup_tool my_tool --verify

# If you need to recreate:
python -m audio_agent.tools.catalog.setup_tool my_tool --force
```

**What this does:**
1. Creates `.venv/` with Python 3.11
2. Installs dependencies from `pyproject.toml`
3. Verifies the environment works

---

### Step 7: Write Documentation (README.md)

```markdown
# My Tool

Brief description of what this tool does.

## ⚠️ Setup Required

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Download Models (if using ML)
```bash
audio-agent-download-models --models my-model
```

### 3. Setup Environment
```bash
python -m audio_agent.tools.catalog.setup_tool my_tool
python -m audio_agent.tools.catalog.setup_tool my_tool --verify
```

## Usage

### Tool: my_tool_action

Description of what it does.

**Input:**
```json
{
  "input_param": "value"
}
```

**Output:**
```json
{
  "content": [{"type": "text", "text": "result"}],
  "isError": false
}
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_KEY` | What this config does | `default` |

## Troubleshooting

### Environment not found
Run setup: `python -m audio_agent.tools.catalog.setup_tool my_tool`

### Model download fails
Check HuggingFace access: `huggingface-cli login`
```

---

### Step 8: Test the Tool

**Manual test:**
```bash
cd audio_agent/tools/catalog/my_tool
.venv/bin/python server.py
```

Then in another terminal:
```python
import asyncio
from audio_agent.tools.mcp import MCPClient

async def test():
    client = MCPClient(
        command=[".venv/bin/python", "server.py"],
        working_dir="/path/to/audio_agent/tools/catalog/my_tool"
    )
    await client.start()
    
    tools = await client.list_tools()
    print(f"Tools: {[t.name for t in tools]}")
    
    result = await client.call_tool(
        "my_tool_action",
        {"input_param": "test"}
    )
    print(f"Result: {result}")
    
    await client.stop()

asyncio.run(test())
```

---

## Integration with Agent

The tool is automatically discovered by the catalog loader. To use it in your agent:

```python
from audio_agent.tools.catalog import load_mcp_server_config
from audio_agent.tools.mcp import MCPServerManager, MCPToolAdapter

# Load config
config = load_mcp_server_config("my_tool")

# Register with server manager
manager = MCPServerManager()
manager.register_config("my_tool", config)

# Discover and register tools
client = await manager.get_client("my_tool")
tools = await client.list_tools()

for tool_info in tools:
    adapter = MCPToolAdapter(
        server_name="my_tool",
        tool_info=tool_info,
        server_manager=manager,
    )
    registry.register_mcp(adapter)
```

---

## Common Issues and Solutions

### Issue: "Virtual environment not found"
**Solution:** Run `python -m audio_agent.tools.catalog.setup_tool my_tool`

### Issue: "No such file or directory" for audio files
**Solution:** The caller must provide absolute paths. In demo scripts:
```python
audio_path = str(Path(args.audio).resolve())
```

### Issue: Model loads from HuggingFace instead of local
**Solution:** Check `config.yaml` uses local path and model is downloaded:
```bash
audio-agent-download-models --models my-model
```

### Issue: Import errors in server
**Solution:** Ensure dependencies are in `pyproject.toml` and environment is recreated:
```bash
python -m audio_agent.tools.catalog.setup_tool my_tool --force
```

### Issue: Tool returns "Empty output"
**Solution:** Check server stderr logs. Common causes:
- Missing environment variables
- Model not downloaded
- Bug in tool implementation

---

## Checklist

Before considering a tool complete:

- [ ] `pyproject.toml` with minimal dependencies
- [ ] `server.py` implements MCP protocol correctly
- [ ] `config.yaml` uses explicit venv path (`.venv/bin/python`)
- [ ] `config.yaml` has correct environment variables
- [ ] `README.md` with setup instructions
- [ ] If using ML models: added to `model_downloader.py`
- [ ] Environment setup works: `setup_tool my_tool --verify`
- [ ] Tool tested manually with MCPClient
- [ ] Tool integrated into agent and tested end-to-end

---

## See Also

- [MCP Protocol Documentation](https://modelcontextprotocol.io/)
- [Tool Catalog README](../audio_agent/tools/catalog/README.md)
- [Template Tool](../audio_agent/tools/catalog/_template/)
