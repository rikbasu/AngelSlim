#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR=${1:?usage: $0 CACHE_DIR OUTPUT_DIR [EPOCHS]}
OUTPUT_DIR=${2:?usage: $0 CACHE_DIR OUTPUT_DIR [EPOCHS]}
EPOCHS=${3:-6}

ROOT_DIR=${ROOT_DIR:-/scratch/rbasu/AngelSlim}
PYTHON=${PYTHON:-/scratch/rbasu/venv-dflare/bin/python}
TARGET_COMPONENTS=${TARGET_COMPONENTS:-/scratch/rbasu/models/gemma-target-components}
JOB_DIR=${JOB_DIR:-/scratch/rbasu/jobs/dflare-public10k}
BLOCK_SIZE=${BLOCK_SIZE:-16}
NUM_ANCHORS=${NUM_ANCHORS:-32}
LR=${LR:-1e-4}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7}
INIT_DRAFT_CHECKPOINT=${INIT_DRAFT_CHECKPOINT:-}

mkdir -p "$JOB_DIR"
exec 9>"$JOB_DIR/training.lock"
if ! flock -n 9; then
  echo "Training is already running."
  exit 0
fi
if [[ -f "$JOB_DIR/TRAINING_DONE" ]]; then
  echo "Training already completed."
  exit 0
fi
rm -f "$JOB_DIR/TRAINING_FAILED"
trap 'date -Is > "$JOB_DIR/TRAINING_FAILED"' ERR

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=/scratch/rbasu/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=/scratch/rbasu/tmp

# A reboot can invoke this script before /scratch and the GPU are ready.
for _ in {1..120}; do
  if [[ -d "$CACHE_DIR" && -x "$PYTHON" ]] \
    && [[ -z "$INIT_DRAFT_CHECKPOINT" || -d "$INIT_DRAFT_CHECKPOINT" ]] \
    && "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'
  then
    break
  fi
  sleep 30
done
[[ -d "$CACHE_DIR" && -x "$PYTHON" ]]
[[ -z "$INIT_DRAFT_CHECKPOINT" || -d "$INIT_DRAFT_CHECKPOINT" ]]
"$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'

TRAIN_ARGS=(
  "$ROOT_DIR/tools/train_dflash_offline.py"
  --target_model_name_or_path "$TARGET_COMPONENTS"
  --draft_model_config_path "$ROOT_DIR/configs/gemma4_dflare.json"
  --train_hidden_path "$CACHE_DIR"
  --output_dir "$OUTPUT_DIR"
  --num_train_epochs "$EPOCHS"
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate "$LR"
  --warmup_ratio 0.04
  --max_grad_norm 1.0
  --attention_backend eager
  --block_size "$BLOCK_SIZE"
  --num_anchors "$NUM_ANCHORS"
  --loss_decay_gamma "$LOSS_DECAY_GAMMA"
  --logging_steps 100
  --save_strategy steps
  --save_steps 1000
  --save_total_limit 3
  --bf16
  --cache_in_memory
  --report_to none
)
if [[ -n "$INIT_DRAFT_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--init_draft_checkpoint "$INIT_DRAFT_CHECKPOINT")
fi

"$PYTHON" "${TRAIN_ARGS[@]}"

date -Is > "$JOB_DIR/TRAINING_DONE"
