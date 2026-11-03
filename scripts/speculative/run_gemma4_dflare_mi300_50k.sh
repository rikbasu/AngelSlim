#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/rbasu/AngelSlim}
PYTHON=${PYTHON:-/home/rbasu/venv-dflare/bin/python}
MODEL=${MODEL:-/home/rbasu/models/gemma-4-E4B-it}
DATA_DIR=${DATA_DIR:-/home/rbasu/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/rbasu/outputs}
JOB_DIR=${JOB_DIR:-/home/rbasu/jobs/dflare-50k}
NUM_GPUS=${NUM_GPUS:-8}

PROMPTS=$DATA_DIR/dflare_public_50k_prompts.jsonl
EXISTING=$DATA_DIR/dflare_public_50k_existing6144.jsonl
DISTILLED=$DATA_DIR/dflare_public_50k_gemma_distilled.jsonl
TRAIN_OUTPUT=$OUTPUT_ROOT/dflare-online-public50k-6ep

mkdir -p "$JOB_DIR" "$OUTPUT_ROOT"
exec 9>"$JOB_DIR/pipeline.lock"
if ! flock -n 9; then
  echo "50K pipeline is already running"
  exit 0
fi
if [[ -f "$JOB_DIR/PIPELINE_DONE" ]]; then
  echo "50K pipeline already completed"
  exit 0
fi
rm -f "$JOB_DIR/PIPELINE_FAILED"
trap 'date -Is > "$JOB_DIR/PIPELINE_FAILED"' ERR

export PATH="/home/rbasu/venv-dflare/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=/home/rbasu/cache/huggingface
export TMPDIR=/home/rbasu/tmp
export HSA_OVERRIDE_GFX_VERSION=9.4.2

if [[ ! -f "$JOB_DIR/DISTILLATION_DONE" ]]; then
  torchrun --standalone --nproc-per-node="$NUM_GPUS" \
    "$ROOT_DIR/tools/generate_gemma4_self_distillation_hf_distributed.py" \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output "$DISTILLED" \
    --resume-file "$EXISTING" \
    --max-model-length 1024 \
    --max-new-tokens 256 \
    --batch-size 4 \
    --temperature 0.6 \
    --seed 0
  date -Is > "$JOB_DIR/DISTILLATION_DONE"
fi

NUM_GPUS="$NUM_GPUS" \
TARGET_MODEL_PATH="$MODEL" \
TRAIN_DATA_PATH="$DISTILLED" \
OUTPUT_DIR="$TRAIN_OUTPUT" \
NUM_EPOCHS=6 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=8 \
MAX_LENGTH=1024 \
MIN_LOSS_TOKENS=16 \
LEARNING_RATE=6e-4 \
BLOCK_SIZE=16 \
NUM_ANCHORS=512 \
LOSS_DECAY_GAMMA=4.5 \
GAMMA_WARMUP=1 \
GAMMA_WARMUP_STEP=1 \
DATA_NUM_PROC=8 \
ATTENTION_BACKEND=eager \
"$ROOT_DIR/scripts/speculative/run_gemma4_dflare_online.sh"

date -Is > "$JOB_DIR/TRAINING_DONE"
date -Is > "$JOB_DIR/PIPELINE_DONE"
