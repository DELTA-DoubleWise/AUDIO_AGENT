# DiariZen Environment Setup Procedure

> **⚠️ CRITICAL**: This tool has **very specific requirements** that differ from other tools. Read this entire document before proceeding.

---

## ⚡ Quick Summary

- **Python Version**: 3.10 (NOT 3.11!)
- **Environment**: Use conda (not uv)
- **Installation Time**: ~30 minutes
- **Success Rate**: High if following this procedure exactly

---

## 🚨 The Python Version Conflict

### The Problem

| Tool | Python Version | Environment Path |
|------|---------------|------------------|
| asr_qwen3 | 3.11 | `.venv/` |
| omni_captioner | 3.11 | `.venv/` |
| **diarizen** | **3.10** | `.venv/` or conda env |

**DiariZen strictly requires Python 3.10** due to dependencies on pyannote.audio and specific NumPy APIs.

### The Solution

You have two options:

#### Option A: Use Conda Named Environment (Recommended for isolation)
```bash
conda create -n diarizen python=3.10
conda activate diarizen
```
- Pros: Complete isolation, standard conda practice
- Cons: Different from other tools' `.venv` pattern

#### Option B: Use `.venv` with Python 3.10 (Consistent path)
```bash
conda create --prefix ./.venv python=3.10
```
- Pros: Same `.venv` path as other tools
- Cons: Must remember it's Python 3.10, not 3.11

**This procedure uses Option B** for path consistency.

---

## 📋 Pre-requisites

### 1. Conda Must Be Available
```bash
source /lihaoyu/.conda.path.sh
conda --version  # Should show version
```

### 2. Git Submodules Must Be Initialized

**CRITICAL**: DiariZen includes a modified `pyannote-audio` as a git submodule.

```bash
cd /lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/audio_agent/tools/catalog/diarizen

# Check if submodules are initialized
ls diarizen_src/pyannote-audio/

# If empty or doesn't exist, initialize:
git submodule update --init --recursive

# Verify pyannote-audio has content
ls diarizen_src/pyannote-audio/pyannote/
```

If the submodule directory is empty, you **cannot proceed**.

---

## 🔧 Step-by-Step Installation

### Step 1: Create Environment with Python 3.10

```bash
cd /lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/audio_agent/tools/catalog/diarizen

# Remove old environment if exists
rm -rf .venv

# Create new environment with Python 3.10 (NOT 3.11!)
source /lihaoyu/.conda.path.sh
conda create --prefix ./.venv python=3.10 -y

# Activate the environment
conda activate ./.venv

# Verify Python version
python --version  # Should show 3.10.x
```

**⚠️ WARNING**: If this shows Python 3.11, something went wrong. Do not proceed.

---

### Step 2: Install PyTorch

**CRITICAL**: PyTorch MUST be installed BEFORE DiariZen.

```bash
# Detect CUDA version and install appropriate PyTorch
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")

if [[ "$CUDA_VERSION" == 12.* ]]; then
    if [[ "$CUDA_VERSION" > "12.3" ]]; then
        echo "Installing PyTorch for CUDA 12.4+"
        pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    else
        echo "Installing PyTorch for CUDA 12.1"
        pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    fi
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    echo "Installing PyTorch for CUDA 11.8"
    pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected, installing CPU-only PyTorch"
    pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
fi
```

---

### Step 3: Install DiariZen

```bash
cd diarizen_src
pip install -e .
```

This installs the main DiariZen package in editable mode.

---

### Step 4: Install pyannote-audio Submodule (CRITICAL!)

> **🚨 CRITICAL WARNING**: You MUST use the submodule version, NOT the PyPI version!
>
> ❌ WRONG: `pip install pyannote.audio`
> ✅ CORRECT: `pip install -e .` from the submodule directory

```bash
cd pyannote-audio
pip install -e .
```

**Why?** The submodule contains modifications that DiariZen depends on. The PyPI version (3.1.1) is incompatible.

---

### Step 5: Lock NumPy Version (CRITICAL!)

> **🚨 CRITICAL**: NumPy MUST be exactly 1.26.4

```bash
pip install numpy==1.26.4
```

**Why?** pyannote.audio (submodule) uses NumPy 1.x APIs that are incompatible with NumPy 2.x.

If you skip this step, you'll get errors like:
```
AttributeError: module 'numpy' has no attribute '_core'
```

---

### Step 6: Install Missing Dependencies

```bash
pip install psutil accelerate
```

These are runtime dependencies not listed in the main requirements.

---

### Step 7: Install Additional Dependencies from pyproject.toml

```bash
cd /lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/audio_agent/tools/catalog/diarizen

# Install remaining deps (excluding torch which is already installed)
pip install huggingface-hub>=0.20.0

# Install the local package (server code)
pip install -e . --no-deps
```

---

## ✅ Verification

### Test 1: Import Test

```bash
cd /lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/audio_agent/tools/catalog/diarizen
.venv/bin/python -c "
from model import DiariZenModel
m = DiariZenModel()
print('✓ DiariZen model loaded successfully')
"
```

Expected output:
```
✓ DiariZen model loaded successfully
```

### Test 2: NumPy Version Check

```bash
.venv/bin/python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

Expected: `NumPy version: 1.26.4`

### Test 3: PyTorch CUDA Check

```bash
.venv/bin/python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠ PyTorch CUDA not available (CPU only)')
"
```

### Test 4: Full Pipeline Test

**⚠️ WARNING**: Use audio files >= 30 seconds. Short audio triggers a bug in pyannote.audio.

```bash
# If you have a test audio file (>= 30s)
.venv/bin/python -c "
from model import DiariZenModel
import librosa

model = DiariZenModel()
audio, sr = librosa.load('path/to/audio.wav', sr=16000)
result = model.diarize(audio, sr)
print(f'Detected {result[\"num_speakers\"]} speakers')
"
```

---

## 🛑 Critical Rules (DO NOT BREAK)

### Rule 1: Never Modify Source Code

```bash
# ❌ NEVER DO THIS:
vim diarizen_src/pyannote-audio/pyannote/audio/inference.py
# ... edit code ...

# ✅ If something doesn't work, check:
# - Installation order
# - Version compatibility
# - Audio file requirements
```

### Rule 2: Strict Installation Order

The order in this procedure is **mandatory**:
1. ✅ Python 3.10 environment
2. ✅ PyTorch FIRST
3. ✅ DiariZen package
4. ✅ pyannote-audio submodule
5. ✅ NumPy 1.26.4
6. ✅ Other dependencies

Reversing steps 2 and 3 will cause conflicts.

### Rule 3: Use Submodule pyannote-audio

```bash
# ❌ WRONG:
pip install pyannote.audio

# ✅ CORRECT:
cd diarizen_src/pyannote-audio
pip install -e .
```

### Rule 4: Lock NumPy to 1.26.4

```bash
# ❌ WRONG:
pip install numpy  # Gets 2.x

# ✅ CORRECT:
pip install numpy==1.26.4
```

---

## 🔧 Troubleshooting

### Error: `'DiariZenPipeline' object has no attribute '_segmentation_model'`

**Cause**: Using PyPI pyannote.audio instead of submodule.

**Fix**:
```bash
pip uninstall pyannote.audio -y
cd diarizen_src/pyannote-audio
pip install -e .
```

### Error: `numpy._core.multiarray` attribute error

**Cause**: NumPy 2.x installed instead of 1.26.4.

**Fix**:
```bash
pip install numpy==1.26.4 --force-reinstall
```

### Error: `UnboundLocalError: cannot access local variable 'segmentations'`

**Cause**: Audio file too short (< 16 seconds).

**Fix**: Use audio files >= 30 seconds for testing.

### Error: `ModuleNotFoundError: No module named 'psutil'`

**Cause**: Missing runtime dependency.

**Fix**:
```bash
pip install psutil accelerate
```

---

## 📝 Summary Checklist

Before considering setup complete, verify:

- [ ] Python 3.10 (not 3.11) in `.venv/bin/python --version`
- [ ] Git submodule `pyannote-audio/` is not empty
- [ ] PyTorch installed BEFORE DiariZen
- [ ] pyannote-audio installed from submodule (`pip install -e .` from submodule dir)
- [ ] NumPy exactly 1.26.4 (`pip show numpy`)
- [ ] psutil and accelerate installed
- [ ] Import test passes
- [ ] NumPy version check shows 1.26.4

---

## 🔄 Post-Restart Verification

After server restart:

```bash
cd /lihaoyu/workspace/AUDIO_AGENT/AUDIO_AGENT/audio_agent/tools/catalog/diarizen

# Reactivate environment
source /lihaoyu/.conda.path.sh
conda activate ./.venv

# Verify Python still works
.venv/bin/python --version  # Should be 3.10.x

# Quick import test
.venv/bin/python -c "from model import DiariZenModel; print('OK')"
```

**Note**: Unlike uv-based tools, conda environments with `--copies` or standalone Python are fully self-contained and should survive restarts without issues.

---

## 📚 References

- `diarizen_src/requirements.txt` - Official dependencies
- `diarizen_src/pyannote-audio/requirements.txt` - Submodule dependencies
- `experience/RULES.md` - Detailed rules from colleague
- `experience/LESSONS_LEARNED.md` - Debugging history

---

**Last Updated**: 2024
**Procedure Version**: 1.0
**Based on**: DiariZen + pyannote.audio submodule experience
