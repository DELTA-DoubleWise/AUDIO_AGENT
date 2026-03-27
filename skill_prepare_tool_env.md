# Skill: Prepare Tool Environment

> **Purpose**: Universal guide for setting up isolated Python environments for MCP tools in the Audio Agent Framework.

---

## 1. Environment Tool Selection

### Decision Matrix

| Tool | Python Version | Use When |
|------|---------------|----------|
| **uv** | 3.11 | Standard tools, no special version requirements |
| **conda** | 3.10 (or specific) | Tools requiring specific Python versions (e.g., DiariZen) |

### Key Differences

**uv (Recommended for standard cases)**
- Fast, lightweight
- Good integration with `pyproject.toml`
- Requires persistent storage configuration (`UV_PYTHON_INSTALL_DIR`)

**conda (Required for version-specific tools)**
- Can install specific Python versions
- Heavier but more flexible
- Self-contained environments

---

## 2. Standard Directory Structure

Every tool should follow this structure:

```
audio_agent/tools/catalog/<tool_name>/
├── .venv/                  # Virtual environment (isolated)
│   ├── bin/
│   │   ├── python          # Python interpreter
│   │   └── ...
│   └── lib/python3.X/site-packages/
│
├── server.py               # MCP server entry point
├── model.py                # Tool implementation/model wrapper
├── setup.sh                # Environment setup script ⭐
├── test_env.py             # Environment test (Python) ⭐
├── test_env.sh             # Environment test (shell wrapper) ⭐
├── pyproject.toml          # Package configuration ⭐
├── config.yaml             # MCP tool configuration
└── README.md               # Tool documentation
```

**Files marked with ⭐ are required for every tool.**

---

## 3. Setup.sh Templates

### Template A: uv-based (Python 3.11)

Use for standard tools without special Python version requirements.

```bash
#!/bin/bash
# Setup script for <tool_name>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find uv - check persistent location first, then PATH
if [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv" ]; then
    UV="/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Please install uv first."
    exit 1
fi

echo "Using uv: $UV"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Detect CUDA and install PyTorch
echo "Installing PyTorch..."
CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")

if [[ "$CUDA_VERSION" == 12.* ]]; then
    if [[ "$CUDA_VERSION" > "12.3" ]]; then
        $UV pip install --python .venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    else
        $UV pip install --python .venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    fi
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    $UV pip install --python .venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
else
    $UV pip install --python .venv/bin/python torch==2.4.0
fi

# Install package
echo "Installing <tool_name> package..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
```

### Template B: conda-based (Python 3.10)

Use when tool requires specific Python version (e.g., DiariZen requires 3.10).

```bash
#!/bin/bash
# Setup script for <tool_name> (requires Python 3.10)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda
source /lihaoyu/.conda.path.sh

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.10
echo "Creating virtual environment with Python 3.10..."
conda create --prefix ./.venv python=3.10 -y

# Install PyTorch (MUST be before other packages if they depend on it)
echo "Installing PyTorch..."
.venv/bin/pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install package and dependencies
echo "Installing <tool_name> package..."
.venv/bin/pip install -e .

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
```

**Critical**: Always use `--python .venv/bin/python` with uv pip install to ensure packages go into the venv, not the base environment.

---

## 4. Pyproject.toml Requirements

### Minimum Required Structure

```toml
[project]
name = "<tool-name>"
version = "1.0.0"
description = "Description for the Audio Agent Framework"
requires-python = ">=3.11"  # or ">=3.10" for conda-based tools
authors = [
    {name = "Audio Agent Team"}
]

dependencies = [
    "torch==2.4.0",
    "transformers>=4.40.0",
    # ... other dependencies
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

# ⭐ CRITICAL: Exclude test files from package discovery
[tool.setuptools]
py-modules = ["server", "model"]  # List only your package modules
```

### Critical Section: [tool.setuptools]

**Without this section**, setuptools will auto-discover all `.py` files as package modules, including `test_env.py`, causing:

```
error: Multiple top-level modules discovered in a flat-layout: ['server', 'test_env'].
```

**Always include** `[tool.setuptools]` with explicit `py-modules` listing.

---

## 5. Test Scripts

Every tool must have `test_env.py` and `test_env.sh` for quick verification.

### test_env.py Template

```python
#!/usr/bin/env python3
"""Quick environment test for <tool_name>."""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    # Add other critical imports...
    
    return True


def test_torch_cuda():
    """Test PyTorch CUDA availability."""
    print("\nTesting PyTorch CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠ CUDA not available (CPU only)")
    
    return True


def test_model_load():
    """Test that the model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        from model import <ModelClass>
        print("  ✓ <ModelClass> imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("<Tool Name> Environment Test")
    print("=" * 60)
    print()
    
    results = [
        ("Imports", test_imports()),
        ("PyTorch CUDA", test_torch_cuda()),
        ("Model Loading", test_model_load()),
    ]
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print()
    print("✓ All tests passed!" if all_passed else "✗ Some tests failed.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

### test_env.sh Template

```bash
#!/bin/bash
# Quick environment test wrapper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python interpreter not found at $PYTHON_EXE"
    echo "Please run setup.sh first."
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
echo "Python version: $($PYTHON_EXE --version)"
echo ""

$PYTHON_EXE test_env.py "$@"
```

---

## 6. Common Issues & Solutions

### Issue 1: "No module named 'torch'" (or any package)

**Symptoms**: 
- Package is installed but import fails
- test_env.py shows `✗ torch: No module named 'torch'`

**Root Cause**: 
Packages installed in base conda environment, not in `.venv`.

**Diagnosis**:
```bash
ls .venv/lib/python3.X/site-packages/ | grep torch
# Should show torch packages
```

**Solution**:
- For uv: Always use `--python .venv/bin/python` flag
- For conda: Use `.venv/bin/pip` instead of just `pip`

### Issue 2: "Multiple top-level modules discovered"

**Symptoms**:
```
error: Multiple top-level modules discovered in a flat-layout: ['server', 'test_env'].
```

**Root Cause**:
Missing `[tool.setuptools]` section in `pyproject.toml`.

**Solution**:
Add to `pyproject.toml`:
```toml
[tool.setuptools]
py-modules = ["server", "model"]  # Exclude test_env.py
```

### Issue 3: Python Version Mismatch

**Symptoms**:
- Setup succeeds but test shows wrong Python version
- `Python 3.11.15` when expecting `3.10`

**Root Cause**:
- Wrong Python used for venv creation
- `requires-python` in pyproject.toml doesn't match

**Solution**:
- For uv: Use `--python 3.11` (or `--python 3.10`)
- For conda: Use `python=3.10` in create command
- Update `requires-python` in pyproject.toml

### Issue 4: CUDA Not Available

**Symptoms**:
- PyTorch installs but `torch.cuda.is_available()` returns False

**Root Cause**:
- Installed CPU-only PyTorch
- Wrong CUDA version index URL

**Solution**:
Install with correct CUDA index:
```bash
# CUDA 12.4+
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

### Issue 5: Empty venv After Setup

**Symptoms**:
- `.venv` exists but `site-packages` is nearly empty
- Only `_virtualenv.py` present

**Root Cause**:
Setup script didn't actually install packages into venv.

**Solution**:
Verify setup script uses correct pip/python:
```bash
# Wrong (installs in base env):
pip install torch

# Correct (installs in venv):
.venv/bin/pip install torch
# or
uv pip install --python .venv/bin/python torch
```

---

## 7. Verification Checklist

Before considering a tool environment ready:

- [ ] `.venv/` directory exists
- [ ] `.venv/bin/python` exists and is executable
- [ ] Python version matches requirement (`python --version`)
- [ ] `setup.sh` completed without errors
- [ ] `./test_env.sh` passes all tests
- [ ] Key imports work (torch, transformers, etc.)
- [ ] CUDA available (if GPU expected)
- [ ] Model can be imported/instantiated
- [ ] `server.py` can start without errors

### Bulk Verification

To verify all tools at once from the project root:

```bash
# Verify all tools
./verify_all_tools.sh

# Setup and verify all tools (useful after server restart)
./verify_all_tools.sh --setup
```

This script finds all `test_env.sh` files in the catalog and runs them, providing a summary of which tools passed/failed.

---

## 8. Quick Reference

### Create New Tool Environment

1. Create directory: `audio_agent/tools/catalog/<tool_name>/`
2. Copy templates:
   - `setup.sh` (choose uv or conda template)
   - `test_env.py` + `test_env.sh`
   - `pyproject.toml` (update `py-modules`!)
3. Run `./setup.sh`
4. Run `./test_env.sh` to verify
5. Test server: `.venv/bin/python server.py`

### Post-Server-Restart Recovery

```bash
# Reactivate persistent uv (for uv-based tools)
source /lihaoyu/workspace/AUDIO_AGENT/.uv/activate.sh

# Or reactivate conda (for conda-based tools)
source /lihaoyu/.conda.path.sh
conda activate ./.venv

# Verify a single tool still works
./test_env.sh

# Or verify all tools at once
./verify_all_tools.sh
```

---

## Related Documents

- `PERSISTENT_UV_SETUP.md` - uv configuration for persistence
- Individual tool `SETUP_PROCEDURE.md` files - tool-specific instructions
- `experience/RULES.md` and `experience/LESSONS_LEARNED.md` - collected wisdom

---

**Last Updated**: 2024
**Skill Version**: 1.0
