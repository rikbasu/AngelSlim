#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/rbasu/AngelSlim}
PYTHON=${PYTHON:-/home/rbasu/venv-dflare/bin/python}
TARGET_MODEL=${TARGET_MODEL:-/home/rbasu/models/gemma-4-E4B-it}
CACHE_DIR=${CACHE_DIR:-/home/rbasu/data/offline-cache-public10k-vllm}
INITIAL_DRAFT=${INITIAL_DRAFT:-/home/rbasu/outputs/dflare-vllm-aligned-public50k-1ep}
OUTPUT_DIR=${OUTPUT_DIR:-/home/rbasu/outputs/dflare-strix023-cache-aligned-10k-1ep}
JOB_DIR=${JOB_DIR:-/home/rbasu/jobs/dflare-strix023-cache-alignment}
NUM_GPUS=${NUM_GPUS:-8}

mkdir -p "$JOB_DIR" "$OUTPUT_DIR"
exec 9>"$JOB_DIR/training.lock"
if ! flock -n 9; then
  echo "Strix-vLLM-cache alignment is already running"
  exit 0
fi
if [[ -f "$JOB_DIR/TRAINING_DONE" ]]; then
  echo "Strix-vLLM-cache alignment already completed"
  exit 0
fi
rm -f "$JOB_DIR/TRAINING_FAILED"
trap 'date -Is > "$JOB_DIR/TRAINING_FAILED"' ERR

export PATH="/home/rbasu/venv-dflare/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=/home/rbasu/cache/huggingface
export TMPDIR=/home/rbasu/tmp
export HSA_OVERRIDE_GFX_VERSION=9.4.2

torchrun --standalone --nproc-per-node="$NUM_GPUS" \
  "$ROOT_DIR/tools/train_dflash_offline.py" \
  --target_model_name_or_path "$TARGET_MODEL" \
  --draft_model_config_path "$ROOT_DIR/configs/gemma4_dflare.json" \
  --init_draft_checkpoint "$INITIAL_DRAFT" \
  --train_hidden_path "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.04 \
  --max_grad_norm 1.0 \
  --attention_backend eager \
  --block_size 16 \
  --num_anchors 512 \
  --loss_decay_gamma 4.5 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 2 \
  --eval_strategy no \
  --bf16 \
  --report_to none

date -Is > "$JOB_DIR/TRAINING_DONE"
