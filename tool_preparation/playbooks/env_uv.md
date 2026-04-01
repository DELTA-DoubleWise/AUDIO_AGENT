# UV Environment Strategy Playbook

## 何时使用 UV

| 条件 | 使用 UV |
|------|---------|
| 纯 Python 依赖 | ✅ 推荐 |
| 主要依赖为 PyPI 包 | ✅ 推荐 |
| 无复杂 C++ 扩展 | ✅ 推荐 |
| 无系统库依赖 | ✅ 推荐 |
| 无 CUDA 编译需求 | ✅ 推荐 |

## 环境创建方式

```bash
cd audio_agent/tools/catalog/{tool_name}

# 创建环境
/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv venv --python=python3.11

# 安装依赖
/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv pip install --python .venv/bin/python -e .
```

> **Critical**: Always use `--python .venv/bin/python` with uv pip install to ensure packages go into the venv, not the base environment.

## Lock/Sync 约定

```bash
# 导出精确依赖
/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv pip freeze --python .venv/bin/python > requirements.lock

# 从 lock 恢复
/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv pip install --python .venv/bin/python -r requirements.lock
```

## 常见失败和修复

### 1. 系统 PyTorch 与虚拟环境冲突

**症状**: `ModuleNotFoundError: No module named 'torch._utils'`

**原因**: UV 环境隔离导致无法访问系统已安装的 PyTorch

**修复**:
```bash
# 在虚拟环境中重新安装 PyTorch
/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv pip install --python .venv/bin/python torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. NumPy 版本冲突

**症状**: `AttributeError: np.sctypes was removed in the NumPy 2.0`

**原因**: NeMo 等库不支持 NumPy 2.0

**修复**:
```bash
/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv pip install --python .venv/bin/python numpy==1.26.4
```

### 3. Torchvision 版本不匹配

**症状**: `RuntimeError: operator torchvision::nms does not exist`

**修复**:
```bash
# 安装与 PyTorch 匹配的 torchvision
/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv pip install --python .venv/bin/python torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. "Multiple top-level modules discovered"

**症状**:
```
error: Multiple top-level modules discovered in a flat-layout: ['server', 'test_env'].
```

**原因**: Missing `[tool.setuptools]` section in `pyproject.toml`.

**修复**:
Add to `pyproject.toml`:
```toml
[tool.setuptools]
py-modules = ["server", "model"]  # Exclude test_env.py
```

## 验证命令

```bash
# 验证 Python 版本
.venv/bin/python --version

# 验证关键包
.venv/bin/python -c "import torch; print(torch.__version__)"
.venv/bin/python -c "import numpy; print(numpy.__version__)"
```

## 参考

- For complete setup.sh templates, see Section 3 below.
- For persistent uv configuration, see project root `setup_tools_uv_persistent.sh`.
