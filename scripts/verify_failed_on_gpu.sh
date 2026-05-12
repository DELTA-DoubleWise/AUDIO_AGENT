#!/bin/bash
# Re-verify the two tools that failed in the original GPU run.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TMPDIR="${TMPDIR:-/itet-stor/yuchwang/net_scratch/tmp}"
export AUDIO_AGENT_MODELS_DIR="${AUDIO_AGENT_MODELS_DIR:-$REPO_ROOT/models}"
export HF_HOME="${HF_HOME:-$AUDIO_AGENT_MODELS_DIR/.hf_cache}"

ARTIFACTS="$REPO_ROOT/.artifacts/verify_runs/retry_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARTIFACTS"

nvidia-smi -L | tee -a "$ARTIFACTS/run.log"
echo "Hostname: $(hostname)" | tee -a "$ARTIFACTS/run.log"

for tool in fireredvad sortformer_diarization; do
    echo "[$(date +%H:%M:%S)] Testing $tool ..." | tee -a "$ARTIFACTS/run.log"
    log="$ARTIFACTS/${tool}.log"
    if (cd "audio_agent/tools/catalog/$tool" && bash test_env.sh) >"$log" 2>&1; then
        echo "  PASS" | tee -a "$ARTIFACTS/run.log"
    else
        echo "  FAIL (see $log)" | tee -a "$ARTIFACTS/run.log"
        echo "--- last 25 lines of $tool.log ---" | tee -a "$ARTIFACTS/run.log"
        tail -25 "$log" | tee -a "$ARTIFACTS/run.log"
    fi
done

echo "Artifacts in $ARTIFACTS" | tee -a "$ARTIFACTS/run.log"
