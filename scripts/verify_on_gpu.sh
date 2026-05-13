#!/bin/bash
# GPU verification harness for AUDIO_AGENT after a fresh setup.
#
# What it does (all on a single GPU allocation):
#   1. Sanity-checks GPU + Python + repo env.
#   2. Runs each MCP tool's test_env.sh under its own .venv (real model loads
#      where weights are downloaded).
#   3. If DASHSCOPE_API_KEY (or OPENAI_API_KEY) is set, runs one targeted
#      demo_run_api_full.py question per catalog tool.
#
# Submit interactively (adapt partition / exclude list / time to your cluster):
#   srun --mem=32GB --gres=gpu:1 --pty bash -i
#   then: ./scripts/verify_on_gpu.sh
#
# Or non-interactively:
#   sbatch --mem=32GB --gres=gpu:1 --time=02:00:00 \
#          -o .artifacts/verify_runs/%j.out \
#          --wrap='./scripts/verify_on_gpu.sh'

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Cache/temp directories. Each defaults to a local path inside the repo so the
# script works without any pre-configured cluster-specific env vars. Override
# TMPDIR ahead of time if your cluster has a dedicated scratch tmp.
export TMPDIR="${TMPDIR:-$REPO_ROOT/.artifacts/tmp}"
export AUDIO_AGENT_MODELS_DIR="${AUDIO_AGENT_MODELS_DIR:-$REPO_ROOT/models}"
export HF_HOME="${HF_HOME:-$AUDIO_AGENT_MODELS_DIR/.hf_cache}"
mkdir -p "$TMPDIR" "$AUDIO_AGENT_MODELS_DIR"

ARTIFACTS="$REPO_ROOT/.artifacts/verify_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARTIFACTS"

log() { printf "[%(%H:%M:%S)T] %s\n" -1 "$*" | tee -a "$ARTIFACTS/run.log"; }
section() { echo; echo "=========================================="; echo "  $*"; echo "=========================================="; echo; }

section "Preflight"
log "Hostname: $(hostname)"
log "SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
log "REPO_ROOT=$REPO_ROOT"
log "AUDIO_AGENT_MODELS_DIR=$AUDIO_AGENT_MODELS_DIR"

if ! nvidia-smi -L 2>/dev/null; then
    log "WARN: nvidia-smi missing GPU info — this script expects a GPU allocation."
fi

# Discover and source a conda hook if conda is not already on PATH.
# Search order: $CONDA_SH (explicit override), then common install roots.
if ! command -v conda >/dev/null 2>&1; then
    for candidate in \
        "${CONDA_SH:-}" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            # shellcheck source=/dev/null
            source "$candidate"
            break
        fi
    done
fi

# Locate the main audio_agent env. Resolution order:
#   1. $AUDIO_AGENT_MAIN_ENV (explicit override, expected to point at conda env root)
#   2. $CONDA_PREFIX if the user already activated a conda env
#   3. <repo>/.venv (uv venv created at repo root)
if [ -n "${AUDIO_AGENT_MAIN_ENV:-}" ] && [ -x "$AUDIO_AGENT_MAIN_ENV/bin/python" ]; then
    MAIN_ENV="$AUDIO_AGENT_MAIN_ENV"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    MAIN_ENV="$CONDA_PREFIX"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    MAIN_ENV="$REPO_ROOT/.venv"
else
    log "ERROR: cannot locate the audio_agent main env."
    log "  Set AUDIO_AGENT_MAIN_ENV=/path/to/env (conda prefix or venv root),"
    log "  or activate your conda env before running this script,"
    log "  or build the uv venv at \$REPO_ROOT/.venv per ENVIRONMENT_SETUP.md."
    exit 1
fi
log "Main env: $MAIN_ENV"
"$MAIN_ENV/bin/python" --version

section "Per-tool env verification"
PASSED=()
FAILED=()
for tool_dir in $(find "$REPO_ROOT/audio_agent/tools/catalog" -mindepth 2 -maxdepth 2 -name test_env.sh ! -path "*/_template/*" -printf '%h\n' | sort); do
    tool="$(basename "$tool_dir")"
    log_file="$ARTIFACTS/${tool}.log"
    log "Testing $tool ..."
    if (cd "$tool_dir" && bash test_env.sh) >"$log_file" 2>&1; then
        PASSED+=("$tool")
        log "  PASS"
    else
        FAILED+=("$tool")
        log "  FAIL (see $log_file)"
    fi
done

section "Per-tool summary"
log "Passed: ${#PASSED[@]} / $((${#PASSED[@]} + ${#FAILED[@]}))"
[ "${#PASSED[@]}" -gt 0 ] && log "  passed: ${PASSED[*]}"
[ "${#FAILED[@]}" -gt 0 ] && log "  failed: ${FAILED[*]}"

# Stage a tiny fixture WAV for demo runs.
FIXTURE="$ARTIFACTS/fixture.wav"
"$MAIN_ENV/bin/python" - "$FIXTURE" <<'PY'
import sys, numpy as np, soundfile as sf
path = sys.argv[1]
sr = 16000
# 3 seconds: 1s silence, 1s 440 Hz tone, 1s 880 Hz tone.
t1 = np.linspace(0, 1, sr, endpoint=False)
audio = np.concatenate([
    np.zeros(sr, dtype="float32"),
    (0.25 * np.sin(2*np.pi*440*t1)).astype("float32"),
    (0.25 * np.sin(2*np.pi*880*t1)).astype("float32"),
])
sf.write(path, audio, sr)
print(f"wrote {path}")
PY

section "demo_run_api_full.py (only if DASHSCOPE_API_KEY is set)"
if [ -z "${DASHSCOPE_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    log "Skipping API-driven demo runs: no DASHSCOPE_API_KEY / OPENAI_API_KEY in env."
    log "Re-run with:  DASHSCOPE_API_KEY=sk-... $0"
    exit 0
fi

# Per-tool targeted questions for the API demo.
declare -A QUESTIONS=(
    [ffmpeg]="Use the audio_stats tool to report this file's sample rate and duration, then summarize."
    [librosa]="Use the get_audio_info tool to report basic audio metadata, then summarize."
    [snakers4_silero-vad]="Use the vad_predict tool to list any speech regions. There is no speech, so report that."
    [fireredvad]="Use the fireredvad_predict tool to list speech regions, then summarize."
    [asr_qwen3]="Use the transcribe_qwenasr tool to transcribe this audio. Report the transcript or that there is no speech."
    [fireredasr2s]="Use the transcribe_fireredasr tool to transcribe this audio."
    [whisperx]="Use the transcribe_whisperx tool to transcribe this audio."
    [diarizen]="Use the diarize tool to list speaker turns. There is no speech, so report that."
    [sortformer_diarization]="Use the diarize_sortformer tool to list speaker turns. There is no speech, so report that."
    [tempo_cnn]="Use the estimate_tempo_cnn_mirex tool to estimate this clip's BPM."
    [lv_chordia]="Use the recognize_chords_large_vocab tool to label any chord progression in this audio."
    [autochord]="Use the recognize_chords tool to label the chord progression in this audio."
    [omni_captioner]="Use the inspect_audio_plots tool to describe visible structure in this audio's plots."
)

for tool in "${!QUESTIONS[@]}"; do
    log "demo_run_api_full --tool $tool"
    out="$ARTIFACTS/demo_${tool}.log"
    "$MAIN_ENV/bin/python" -m audio_agent.examples.demo_run_api_full \
        --audio "$FIXTURE" \
        --question "${QUESTIONS[$tool]}" \
        --max-steps 5 \
        >"$out" 2>&1 \
        && log "  done -> $out" \
        || log "  FAILED -> $out (continuing)"
done

section "All verification artifacts in $ARTIFACTS"
ls -la "$ARTIFACTS"
