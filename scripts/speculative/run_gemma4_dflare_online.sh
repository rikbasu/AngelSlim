#!/usr/bin/env bash
set -euo pipefail

# Gemma-4-E4B DFlare online training.
#
# Required:
#   TARGET_MODEL_PATH=/path/to/google/gemma-4-E4B-it
#   TRAIN_DATA_PATH=/path/to/conversations.jsonl
# Optional:
#   NUM_GPUS=1, OUTPUT_DIR=..., SAMPLE_NUM=5000
#
# The target model must be loadable by Transformers 5.14.1 or newer and the
# training data must use the standard {"id", "conversations"} schema.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

NUM_GPUS="${NUM_GPUS:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-eager}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/gemma4_dflare.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/gemma4-e4b-dflare-b16}"
SAMPLE_NUM="${SAMPLE_NUM:-}"
DATA_NUM_PROC="${DATA_NUM_PROC:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-3072}"

if [[ -z "$TARGET_MODEL_PATH" || -z "$TRAIN_DATA_PATH" ]]; then
  echo "TARGET_MODEL_PATH and TRAIN_DATA_PATH must be set" >&2
  exit 2
fi

SAMPLE_ARGS=()
if [[ -n "$SAMPLE_NUM" ]]; then
  SAMPLE_ARGS+=(--sample_num "$SAMPLE_NUM")
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  DISTRIBUTED_ARGS=(--standalone --nproc_per_node "$NUM_GPUS")
else
  DISTRIBUTED_ARGS=(--nproc_per_node 1)
fi

torchrun "${DISTRIBUTED_ARGS[@]}" \
  "$ROOT_DIR/tools/train_dflash_online.py" \
  --target_model_name_or_path "$TARGET_MODEL_PATH" \
  --draft_model_config_path "$DRAFT_CONFIG_PATH" \
  --draft_arch dflare \
  --train_data_path "$TRAIN_DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --modal_type DFlash \
  --training_mode online \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --learning_rate 6e-4 \
  --warmup_ratio 0.04 \
  --max_grad_norm 1.0 \
  --model_max_length "$MAX_LENGTH" \
  --chat_template_type gemma4 \
  --attention_backend "$ATTENTION_BACKEND" \
  --block_size 16 \
  --num_anchors 512 \
  --loss_decay_gamma 7 \
  --num_proc "$DATA_NUM_PROC" \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 500 \
  --bf16 \
  --lr_scheduler_type cosine \
  --dataloader_drop_last
